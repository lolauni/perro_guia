#!/usr/bin/env python3
"""
percepcion_fusion.py  -  Supergiz: percepción unificada

Sustituye a percepcion_completa.py + percepcion_3d.py en un solo nodo con tres capas:

  CAPA RÁPIDA (a la frecuencia de la cámara, en el callback de profundidad)
      Plano del suelo (RANSAC con prior de inclinación) -> obstáculos / desniveles / suelo.
      Es la capa de seguridad: nunca espera a ninguna red neuronal.

  CAPA LENTA, hilo 1 (YOLOv8-seg, ~10 Hz)
      Detecta objetos y, con la profundidad alineada, calcula distancia y ángulo de cada uno.

  CAPA LENTA, hilo 2 (SegFormer, ~2 Hz)
      Clasifica el terreno (suelo, acera, hierba, calzada, paso de cebra...) y lo traduce a un
      COSTE por clase (COSTE_CLASE, más abajo) que es lo que luego puede ponderar el movimiento.

Salidas:
  /percepcion/mascara_final        mono8  255 = navegable, 0 = no
  /percepcion/mapa_coste           mono8  0 = bloqueado, 1..255 = cuanto mayor, más preferible
  /percepcion/distancia_obstaculo  Float32  distancia al obstáculo en el pasillo del robot (-1 = libre)
  /percepcion/desnivel             Bool     escalón/hueco delante (con antirrebote)
  /percepcion/geometria_ok         Bool     False si no hay plano de suelo fiable (el movimiento debería parar)
  /percepcion/sectores             Float32MultiArray  distancia mínima de obstáculo por sector, izq -> der
  /percepcion/objetos              String (JSON) lista de objetos con clase, distancia, x lateral y ángulo
  /percepcion/imagen_visu          Image (solo si publicar_visu:=True)

Ejemplo:
  python3 percepcion_fusion.py --ros-args -p cam_pitch_deg:=5.0 -p seg_hz:=1.0 -p debug_view:=False
(los parámetros float se pasan con decimales: 2.0 y no 2; yolo_hz:=0.0 o seg_hz:=0.0 apagan ese hilo)
"""
import json
import math
import os
import threading
import time
from collections import deque

import cv2
import numpy as np
import torch
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import Float32, Float32MultiArray, Bool, String
from cv_bridge import CvBridge
from ultralytics import YOLO
from transformers import SegformerForSemanticSegmentation

# ---------------------------------------------------------------------------
# Constantes (editar a gusto)
# ---------------------------------------------------------------------------
SEGFORMER_ADE20K = "nvidia/segformer-b0-finetuned-ade-512-512"

# COSTE por clase de terreno (nombre ADE20K, primer término). 0 = no transitable.
# Cualquier clase que no esté aquí vale 0. 'road' no aparece a propósito: la calzada solo se
# permite en el paso de cebra (COSTE_CEBRA). Si quieres que sea "transitable en emergencia",
# añade 'road': 30.
COSTE_CLASE = {
    'floor': 255, 'sidewalk': 255,
    'rug': 230, 'path': 230,
    'earth': 190, 'land': 190, 'dirt track': 190,
    'grass': 150, 'field': 150,
    'sand': 120,
}
COSTE_CEBRA = 200     # paso de cebra (blanco sobre calzada)
COSTE_GEO = 200       # coste cuando solo hay geometría (sin semántica fresca)
ROAD_NAME = 'road'

# Clases COCO que cuentan como obstáculo (persona, bici, coche, moto, bus, camión, semáforo,
# boca de incendios, stop, banco, pájaro, gato, perro, silla).
OBSTACLE_CLASSES = [0, 1, 2, 3, 5, 7, 9, 10, 11, 13, 14, 15, 16, 56]

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

YOLO_CANDIDATOS = [
    'yolov8n-seg_ncnn_model',       # mejor opción en Raspberry Pi
    'yolov8n-seg_openvino_model',   # PC Intel
    'yolov8n-seg.pt',               # PyTorch, funciona siempre
]

WIN = 'Supergiz - Percepcion fusionada'
FONT = cv2.FONT_HERSHEY_SIMPLEX


def nombre_base(etiqueta):
    return etiqueta.split(',')[0].strip().lower()


# ---------------------------------------------------------------------------
# Geometría (sin ROS, para poder probarla aparte)
# ---------------------------------------------------------------------------
class SueloRansac:
    """
    Ajusta el plano del suelo en coordenadas ópticas de la cámara (x derecha, y abajo, z delante).
    Plano: n·P + d = 0, con n apuntando hacia arriba, de modo que h = n·P + d es la altura sobre el suelo.

    Mejoras respecto a la versión anterior:
      - Prior de inclinación: solo se aceptan planos cuya normal esté a menos de max_tilt_deg de la
        normal esperada según cam_pitch_deg (pitch > 0 = cámara mirando hacia abajo). Así una pared
        o un mueble nunca pueden ganar el RANSAC aunque dominen la imagen.
      - Seguimiento: una vez hay plano, solo se usan los puntos cercanos a él.
      - Refinado en dos pasadas (SVD) con tolerancia decreciente.
    """

    def __init__(self, iters=100, tol=0.05, roi_top=0.45, z_max=3.0,
                 pitch_deg=0.0, max_tilt_deg=25.0, min_span=0.5, seed=None):
        self.iters, self.tol, self.roi_top, self.z_max = iters, tol, roi_top, z_max
        self.min_span = min_span      # m de profundidad que deben cubrir los inliers para fiarse de la inclinación
        th = math.radians(pitch_deg)
        self.n_prior = np.array([0.0, -math.cos(th), -math.sin(th)])
        self.cos_tilt = math.cos(math.radians(max_tilt_deg))
        self.rng = np.random.default_rng(seed)
        self.plano = None
        self.fallos = 0
        self.fallos_seguidos = 0

    def _refinar(self, Q):
        c = Q.mean(axis=0)
        _, _, vt = np.linalg.svd(Q - c, full_matrices=False)
        n = vt[-1]
        if n @ self.n_prior < 0:
            n = -n
        return n, -float(n @ c)

    def ajustar(self, X, Y, Z, valid):
        hs = Z.shape[0]
        cand = valid & (Z < self.z_max)
        cand[:int(hs * self.roi_top), :] = False

        # Seguimiento: con plano previo, solo puntos cerca de él
        if self.plano is not None and self.fallos_seguidos < 3:
            n, d = self.plano
            hprev = float(n[0]) * X + float(n[1]) * Y + float(n[2]) * Z + float(d)
            cand &= np.abs(hprev) < 0.12 + 0.02 * Z

        P = np.stack([X[cand], Y[cand], Z[cand]], axis=1)
        N = len(P)
        nuevo = None
        if N >= 150:
            if N > 3000:
                P = P[self.rng.choice(N, 3000, replace=False)]
                N = 3000
            idx = self.rng.integers(0, N, size=(self.iters, 3))
            p0, p1, p2 = P[idx[:, 0]], P[idx[:, 1]], P[idx[:, 2]]
            nrm = np.cross(p1 - p0, p2 - p0)
            norma = np.linalg.norm(nrm, axis=1, keepdims=True)
            nrm = nrm / np.maximum(norma, 1e-9)
            nrm = nrm * np.where(nrm @ self.n_prior < 0, -1.0, 1.0)[:, None]
            ok = (norma[:, 0] > 1e-6) & (nrm @ self.n_prior > self.cos_tilt)
            dd = -np.einsum('kj,kj->k', nrm, p0)
            dist = np.abs(P @ nrm.T + dd)                     # (N, K)
            inl = (dist < self.tol).sum(axis=0)
            inl[~ok] = 0
            mejor = int(inl.argmax())

            if inl[mejor] >= max(80, 0.12 * N):
                Q = P[dist[:, mejor] < self.tol].astype(np.float64)
                n, d = self._refinar(Q)
                Q2 = P[np.abs(P @ n + d) < 0.5 * self.tol].astype(np.float64)
                if len(Q2) >= 60:
                    Q = Q2
                    n, d = self._refinar(Q2)
                if float(np.ptp(Q[:, 2])) < self.min_span:
                    # Franja de suelo demasiado corta (pared cercana, etc.): la inclinación no se
                    # puede estimar bien. Se conserva la normal anterior (o el prior) y solo se
                    # ajusta la altura de la cámara.
                    n = self.plano[0] if self.plano is not None else self.n_prior
                    d = -float(np.median(Q @ n))
                if n @ self.n_prior > self.cos_tilt:
                    nuevo = (n, d)

        if nuevo is None:
            self.fallos += 1
            self.fallos_seguidos += 1
            if self.fallos_seguidos > 10:     # plano demasiado viejo: no fiarse
                self.plano = None
            return self.plano

        self.fallos_seguidos = 0
        if self.plano is None:
            self.plano = nuevo
        else:
            a = 0.3                           # suavizado temporal
            n = a * nuevo[0] + (1 - a) * self.plano[0]
            n = n / np.linalg.norm(n)
            self.plano = (n, a * nuevo[1] + (1 - a) * self.plano[1])
        return self.plano


def clasificar(plano, X, Y, Z, valid, suelo_tol, k_ruido, obst_min, obst_alto_max, desnivel_min):
    """
    Altura de cada punto sobre el suelo y clasificación. Las tolerancias crecen con z^2, que es
    como crece el error de una cámara estéreo. Devuelve (h, suelo, obstaculo, desnivel) booleanos.

    Ojo: 'suelo' es solo el suelo estricto (sobre el plano). Lo navegable es todo lo que no es
    obstáculo ni desnivel, así una pequeña irregularidad o un poco de ruido no abre agujeros.
    """
    n, d = plano
    h = float(n[0]) * X + float(n[1]) * Y + float(n[2]) * Z + float(d)
    r = k_ruido * Z * Z
    suelo = valid & (np.abs(h) < suelo_tol + r)
    obst = valid & (h > obst_min + r) & (h < obst_alto_max)
    desn = valid & (h < -(desnivel_min + r))
    return h, suelo, obst, desn


# ---------------------------------------------------------------------------
# Nodo
# ---------------------------------------------------------------------------
class PercepcionFusion(Node):
    def __init__(self):
        super().__init__('percepcion_fusion')
        self.bridge = CvBridge()
        self.stop = threading.Event()

        def par(nombre, defecto):
            self.declare_parameter(nombre, defecto)
            return self.get_parameter(nombre).value

        # --- Entradas / modelos ---
        base = par('base_path', '/home/lola/Documentos/ROS2/supergiz_ws')
        topico_color = par('topico_color', '/camera/camera/color/image_raw')
        topico_depth = par('topico_depth', '/camera/camera/aligned_depth_to_color/image_raw')
        topico_info = par('topico_info', '/camera/camera/aligned_depth_to_color/camera_info')
        seg_src = par('segformer', SEGFORMER_ADE20K)

        # --- Geometría ---
        self.stride = par('stride', 4)                    # 640x480 -> 160x120
        self.dist_min = par('dist_min', 0.25)             # m
        self.dist_max = par('dist_max', 4.0)              # m
        cam_pitch = par('cam_pitch_deg', 0.0)             # >0 = cámara mirando hacia abajo
        max_tilt = par('max_tilt_deg', 25.0)              # tolerancia respecto al pitch esperado
        roi_top = par('roi_top', 0.45)                    # fracción superior que NO se usa para el suelo
        iters = par('ransac_iters', 100)
        tol = par('ransac_tol', 0.05)
        self.suelo_tol = par('suelo_tol', 0.03)           # m
        self.k_ruido = par('k_ruido', 0.008)              # m/m^2 (ruido estéreo ~ k z^2)
        self.obst_min = par('obst_min', 0.08)             # m sobre el suelo
        self.obst_alto_max = par('obst_alto_max', 2.2)    # m
        self.desnivel_min = par('desnivel_min', 0.10)     # m bajo el suelo
        self.desnivel_dist_max = par('desnivel_dist_max', 2.5)   # m
        self.ancho_pasillo = par('ancho_pasillo', 0.6)    # m, ancho del robot + margen
        self.min_pix = par('min_pix', 12)
        self.n_sectores = par('n_sectores', 9)

        # --- IA ---
        self.imgsz = par('yolo_imgsz', 320)
        self.conf = par('yolo_conf', 0.35)
        yolo_hz = par('yolo_hz', 10.0)
        self.seg_w = par('seg_w', 320)                    # múltiplos de 32
        self.seg_h = par('seg_h', 256)
        seg_hz = par('seg_hz', 2.0)
        self.sem_max_edad = par('sem_max_edad', 2.0)      # s: más viejo -> solo geometría
        self.yolo_max_edad = par('yolo_max_edad', 0.5)    # s
        self.yolo_bloquea = par('yolo_bloquea', True)     # restar los objetos YOLO de la máscara
        self.coste_rescate = par('coste_rescate', 100)    # suelo plano que la semántica niega (0 = off)
        self.umbral_blanco = par('umbral_blanco', 185)
        self.min_area_frac = par('min_area_frac', 0.003)
        torch.set_num_threads(par('torch_threads', 2))

        # --- Salida / depuración ---
        self.debug_view = par('debug_view', True)
        self.publicar_visu = par('publicar_visu', False)

        # --- Geometría: objetos ---
        self.suelo = SueloRansac(iters, tol, roi_top, 3.0, cam_pitch, max_tilt)
        self.kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        self.kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self.kernel_cebra = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 9))
        self.kernel_erode = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        self.hist_desn = deque(maxlen=5)

        # --- 1. YOLOv8-seg ---
        self.yolo = None
        for nombre in YOLO_CANDIDATOS:
            ruta = os.path.join(base, nombre)
            if os.path.exists(ruta):
                self.get_logger().info(f'Cargando YOLOv8-seg desde: {nombre}')
                self.yolo = YOLO(ruta, task='segment')
                break
        if self.yolo is None:
            raise FileNotFoundError(f'No se encontró yolov8n-seg.pt en {base}')

        # --- 2. SegFormer + tabla de costes por clase ---
        self.get_logger().info(f'Cargando SegFormer desde: {seg_src}')
        self.segformer = SegformerForSemanticSegmentation.from_pretrained(seg_src).eval()
        id2label = {int(i): l for i, l in self.segformer.config.id2label.items()}
        self.lut = np.zeros(256, dtype=np.uint8)
        for i, etiqueta in id2label.items():
            nb = nombre_base(etiqueta)
            if nb in COSTE_CLASE:
                self.lut[i] = COSTE_CLASE[nb]
        if not self.lut.any():
            raise RuntimeError('Ninguna clase de SegFormer coincide con COSTE_CLASE.')
        self.road_id = next((i for i, l in id2label.items() if nombre_base(l) == ROAD_NAME), None)
        self.get_logger().info('Costes: ' + ', '.join(
            f'{id2label[int(i)].split(",")[0]}={self.lut[i]}' for i in np.nonzero(self.lut)[0])
            + f' | calzada: {self.road_id}')

        # --- Estado compartido entre hilos (asignaciones de referencia = atómicas en CPython) ---
        self.fx = self.fy = self.cx = self.cy = None
        self.rx = self.ry = None
        self.shape_full = None
        self.last_color = None
        self.depth_snap = None      # (stamp, depth uint16)
        self.sem = None             # (t_monotonic, coste_uint8 [seg_h x seg_w], calzada_bool)
        self.yolo_res = None        # (t_monotonic, mascara_full | None, lista_objetos)
        self.ms_yolo = 0.0
        self.ms_seg = 0.0

        # --- ROS 2 ---
        self.create_subscription(CameraInfo, topico_info, self.info_cb, qos_profile_sensor_data)
        self.create_subscription(Image, topico_depth, self.depth_cb, qos_profile_sensor_data)
        self.create_subscription(Image, topico_color, self.color_cb, qos_profile_sensor_data)

        self.pub_final = self.create_publisher(Image, '/percepcion/mascara_final', 1)
        self.pub_coste = self.create_publisher(Image, '/percepcion/mapa_coste', 1)
        self.pub_visu = self.create_publisher(Image, '/percepcion/imagen_visu', 1)
        self.pub_dist = self.create_publisher(Float32, '/percepcion/distancia_obstaculo', 1)
        self.pub_desn = self.create_publisher(Bool, '/percepcion/desnivel', 1)
        self.pub_ok = self.create_publisher(Bool, '/percepcion/geometria_ok', 1)
        self.pub_sect = self.create_publisher(Float32MultiArray, '/percepcion/sectores', 1)
        self.pub_obj = self.create_publisher(String, '/percepcion/objetos', 1)

        if self.debug_view:
            try:
                cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(WIN, 1280, 480)
            except cv2.error:
                self.get_logger().warn('No se pudo abrir la ventana; sigo sin ella.')
                self.debug_view = False

        self.t_log = time.perf_counter()
        self.t_acc, self.n_acc = 0.0, 0
        self.fps_txt = ''

        # --- Hilos de inferencia ---
        # hz <= 0 desactiva ese hilo (útil para probar la geometría sola)
        for paso, hz in ((self._paso_yolo, yolo_hz), (self._paso_seg, seg_hz)):
            if hz > 0:
                threading.Thread(target=self._bucle, args=(paso, hz), daemon=True).start()

        self.get_logger().info('>>> Percepción fusionada lista <<<')

    # ------------------------------------------------------------------
    # Callbacks ligeros
    # ------------------------------------------------------------------
    def info_cb(self, msg):
        if self.fx is None:
            self.fx, self.fy = msg.k[0], msg.k[4]
            self.cx, self.cy = msg.k[2], msg.k[5]
            self.get_logger().info(
                f'Intrínsecos: fx={self.fx:.1f} fy={self.fy:.1f} cx={self.cx:.1f} cy={self.cy:.1f}')

    def color_cb(self, msg):
        self.last_color = msg

    # ------------------------------------------------------------------
    # Hilos lentos
    # ------------------------------------------------------------------
    def _bucle(self, paso, hz):
        periodo = 1.0 / hz
        ultimo = None
        while not self.stop.is_set():
            msg = self.last_color
            if msg is None or msg is ultimo:
                time.sleep(0.005)
                continue
            ultimo = msg
            t0 = time.perf_counter()
            try:
                paso(msg)
            except Exception as e:
                self.get_logger().error(f'{paso.__name__}: {e}')
                time.sleep(0.5)
            resto = periodo - (time.perf_counter() - t0)
            if resto > 0:
                time.sleep(resto)

    # ---- SegFormer -> coste por clase ----
    def limpiar(self, mask):
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel_open)
        n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if n > 1:
            keep = np.zeros(n, dtype=bool)
            keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= self.min_area_frac * mask.size
            mask = keep[lab].astype(np.uint8) * 255
        return mask

    def calcular_semantica(self, frame):
        sw, sh = self.seg_w, self.seg_h
        small = cv2.resize(frame, (sw, sh), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        x = torch.from_numpy(((rgb - MEAN) / STD).transpose(2, 0, 1)).unsqueeze(0)

        with torch.inference_mode():
            cls = self.segformer(pixel_values=x).logits[0].argmax(0).numpy().astype(np.uint8)
        cls = cv2.resize(cls, (sw, sh), interpolation=cv2.INTER_NEAREST)

        coste = self.lut[cls]                                   # (sh, sw) uint8, copia
        coste[self.limpiar((coste > 0).astype(np.uint8) * 255) == 0] = 0

        calzada = np.zeros((sh, sw), dtype=bool)
        if self.road_id is not None:
            calzada = cls == self.road_id
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            blanco = ((gray > self.umbral_blanco) & calzada).astype(np.uint8) * 255
            cebra = cv2.morphologyEx(blanco, cv2.MORPH_CLOSE, self.kernel_cebra)
            coste[(cebra > 0) & calzada] = COSTE_CEBRA
        return coste, calzada

    def _paso_seg(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        t = time.perf_counter()
        coste, calzada = self.calcular_semantica(frame)
        self.ms_seg = 1000.0 * (time.perf_counter() - t)
        self.sem = (time.monotonic(), coste, calzada)

    # ---- YOLOv8-seg -> objetos con distancia ----
    def _paso_yolo(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        snap = self.depth_snap                     # la profundidad más reciente en este instante
        t = time.perf_counter()
        res = self.yolo(frame, imgsz=self.imgsz, conf=self.conf,
                        classes=OBSTACLE_CLASSES, verbose=False)[0]
        self.ms_yolo = 1000.0 * (time.perf_counter() - t)

        h, w = frame.shape[:2]
        depth = snap[1] if (snap is not None and snap[1].shape[:2] == (h, w)) else None
        objetos, mask_full = [], None

        if res.masks is not None and len(res.masks.xyn):
            mask_full = np.zeros((h, w), dtype=np.uint8)
            clases = res.boxes.cls.cpu().numpy().astype(int)
            confs = res.boxes.conf.cpu().numpy()
            esc = np.array([w, h], dtype=np.float32)
            for poly, c, conf in zip(res.masks.xyn, clases, confs):
                if len(poly) < 3:
                    continue
                pts = (poly * esc).astype(np.int32)
                pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
                pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)
                cv2.fillPoly(mask_full, [pts], 255)

                x, y, bw, bh = cv2.boundingRect(pts)
                obj = {'clase': res.names[int(c)], 'conf': round(float(conf), 2),
                       'caja': [int(x), int(y), int(x + bw), int(y + bh)]}

                if depth is not None and self.fx is not None:
                    m = np.zeros((bh, bw), dtype=np.uint8)
                    cv2.fillPoly(m, [pts - np.array([x, y], dtype=np.int32)], 1)
                    if min(bw, bh) > 12:
                        m = cv2.erode(m, self.kernel_erode)    # fuera los bordes (mezclan fondo)
                    zs = depth[y:y + bh, x:x + bw][m > 0]
                    zs = zs[zs > 0]
                    if zs.size >= 20:
                        z = float(np.percentile(zs, 25)) * 0.001   # percentil bajo: lo más cercano
                        u = x + float(np.nonzero(m)[1].mean())
                        X = (u - self.cx) / self.fx * z
                        obj['dist'] = round(z, 2)
                        obj['x'] = round(X, 2)                      # + = derecha del robot
                        obj['ang'] = round(math.degrees(math.atan2(X, z)), 1)
                objetos.append(obj)

        objetos.sort(key=lambda o: o.get('dist', 99.0))
        self.yolo_res = (time.monotonic(), mask_full, objetos)
        self.pub_obj.publish(String(data=json.dumps(
            {'t': time.time(), 'objetos': objetos}, ensure_ascii=False)))

    # ------------------------------------------------------------------
    # Capa rápida: profundidad
    # ------------------------------------------------------------------
    def depth_cb(self, msg):
        if self.fx is None:
            return
        try:
            t0 = time.perf_counter()
            depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')   # uint16, mm
            self.depth_snap = (msg.header.stamp, depth)
            H, W = depth.shape[:2]
            s = self.stride

            if self.rx is None or self.shape_full != (H, W):
                self.shape_full = (H, W)
                u = np.arange(0, W, s, dtype=np.float32)
                v = np.arange(0, H, s, dtype=np.float32)
                self.rx = ((u - self.cx) / self.fx)[None, :]
                self.ry = ((v - self.cy) / self.fy)[:, None]

            z = depth[::s, ::s].astype(np.float32) * 0.001
            hs, ws = z.shape
            valid = (z > self.dist_min) & (z < self.dist_max)
            X = self.rx * z
            Y = self.ry * z

            plano = self.suelo.ajustar(X, Y, z, valid)
            self.pub_ok.publish(Bool(data=plano is not None))
            if plano is None:
                return

            _, suelo, obst, desn = clasificar(
                plano, X, Y, z, valid, self.suelo_tol, self.k_ruido,
                self.obst_min, self.obst_alto_max, self.desnivel_min)
            suelo_u8 = suelo.astype(np.uint8) * 255
            obst_u8 = cv2.morphologyEx(obst.astype(np.uint8) * 255, cv2.MORPH_OPEN, self.kernel3)
            desn_u8 = cv2.morphologyEx(desn.astype(np.uint8) * 255, cv2.MORPH_OPEN, self.kernel3)
            bloq_geo = (obst_u8 > 0) | (desn_u8 > 0)

            # Pasillo del robot medido en METROS (no en tercios de imagen)
            en_pasillo = np.abs(X) < self.ancho_pasillo / 2
            c_obst = (obst_u8 > 0) & en_pasillo
            dist = -1.0
            if int(c_obst.sum()) >= self.min_pix:
                dist = float(np.percentile(z[c_obst], 5))
            c_desn = (desn_u8 > 0) & en_pasillo & (z < self.desnivel_dist_max)
            self.hist_desn.append(int(c_desn.sum()) >= self.min_pix)
            hay_desnivel = sum(self.hist_desn) >= 3        # antirrebote: 3 de los últimos 5 frames

            # Distancia mínima por sector (izquierda -> derecha)
            bordes = np.linspace(0, ws, self.n_sectores + 1).astype(int)
            min_sec = max(4, self.min_pix // 3)
            sect = []
            for i in range(self.n_sectores):
                a, b = bordes[i], bordes[i + 1]
                sel = bloq_geo[:, a:b]
                sect.append(float(np.percentile(z[:, a:b][sel], 5)) if int(sel.sum()) >= min_sec else -1.0)

            self.pub_dist.publish(Float32(data=dist))
            self.pub_desn.publish(Bool(data=hay_desnivel))
            self.pub_sect.publish(Float32MultiArray(data=sect))

            # --- Fusión: coste = semántica (o geometría) menos lo bloqueado ---
            ahora_m = time.monotonic()
            bloq = bloq_geo.copy()
            yres = self.yolo_res
            if (self.yolo_bloquea and yres is not None and yres[1] is not None
                    and ahora_m - yres[0] < self.yolo_max_edad):
                bloq |= cv2.resize(yres[1], (ws, hs), interpolation=cv2.INTER_NEAREST) > 0

            sem = self.sem
            usa_sem = sem is not None and ahora_m - sem[0] < self.sem_max_edad
            if usa_sem:
                coste = cv2.resize(sem[1], (ws, hs), interpolation=cv2.INTER_NEAREST)
                if self.coste_rescate > 0:
                    # Suelo plano que la semántica niega (p. ej. una alfombra mal clasificada):
                    # se rescata con coste bajo, salvo que sea calzada.
                    calz = cv2.resize(sem[2].astype(np.uint8), (ws, hs),
                                      interpolation=cv2.INTER_NEAREST) > 0
                    coste[(coste == 0) & suelo & ~calz] = self.coste_rescate
            else:
                libre = (valid & ~bloq).astype(np.uint8)
                libre = cv2.morphologyEx(libre, cv2.MORPH_CLOSE, self.kernel3)   # solo agujeros de 1 px
                coste = libre * COSTE_GEO
            coste[bloq] = 0

            # Métricas (sin contar la salida a resolución completa)
            self.t_acc += time.perf_counter() - t0
            self.n_acc += 1
            ahora = time.perf_counter()
            if ahora - self.t_log >= 2.0:
                fps = self.n_acc / (ahora - self.t_log)
                ms = 1000.0 * self.t_acc / self.n_acc
                self.fps_txt = (f'{fps:.1f} FPS | 3D {ms:.1f} ms | YOLO {self.ms_yolo:.0f} ms | '
                                f'Seg {self.ms_seg:.0f} ms | fallos plano {self.suelo.fallos}')
                self.get_logger().info(self.fps_txt)
                self.t_log, self.t_acc, self.n_acc = ahora, 0.0, 0

            # --- Salidas a resolución completa (solo si alguien las usa) ---
            quiere_vis = self.debug_view or self.publicar_visu
            if (self.pub_final.get_subscription_count() > 0
                    or self.pub_coste.get_subscription_count() > 0 or quiere_vis):
                bin_s = (coste > 0).astype(np.uint8) * 255
                final = cv2.threshold(cv2.resize(bin_s, (W, H), interpolation=cv2.INTER_LINEAR),
                                      127, 255, cv2.THRESH_BINARY)[1]
                coste_full = cv2.resize(coste, (W, H), interpolation=cv2.INTER_NEAREST)
                coste_full[final == 0] = 0

                if self.pub_final.get_subscription_count() > 0:
                    m = self.bridge.cv2_to_imgmsg(final, encoding='mono8')
                    m.header = msg.header
                    self.pub_final.publish(m)
                if self.pub_coste.get_subscription_count() > 0:
                    m = self.bridge.cv2_to_imgmsg(coste_full, encoding='mono8')
                    m.header = msg.header
                    self.pub_coste.publish(m)
                if quiere_vis:
                    self.dibujar(msg, obst_u8, desn_u8, final, coste_full,
                                 dist, hay_desnivel, usa_sem, H, W)

        except Exception as e:
            self.get_logger().error(f'Error en percepción: {e}')

    # ------------------------------------------------------------------
    def dibujar(self, msg, obst_u8, desn_u8, final, coste_full, dist, hay_desnivel, usa_sem, H, W):
        col = self.last_color
        if col is not None:
            vis = self.bridge.imgmsg_to_cv2(col, 'bgr8')
            if vis.shape[:2] != (H, W):
                vis = cv2.resize(vis, (W, H))
            vis = vis.copy()
        else:
            vis = np.zeros((H, W, 3), dtype=np.uint8)

        def pintar(m, color, alfa=0.5):
            vis[m] = ((1 - alfa) * vis[m] + alfa * np.array(color)).astype(np.uint8)

        pintar(final > 0, (0, 255, 0), 0.35)                                   # verde: navegable
        pintar(cv2.resize(obst_u8, (W, H), interpolation=cv2.INTER_NEAREST) > 0, (0, 0, 255))
        pintar(cv2.resize(desn_u8, (W, H), interpolation=cv2.INTER_NEAREST) > 0, (255, 0, 255))

        yres = self.yolo_res
        if yres is not None and time.monotonic() - yres[0] < 1.0:
            for o in yres[2]:
                x0, y0, x1, y1 = o['caja']
                cv2.rectangle(vis, (x0, y0), (x1, y1), (255, 200, 0), 2)
                et = o['clase'] + (f" {o['dist']:.1f} m" if 'dist' in o else '')
                cv2.putText(vis, et, (x0, max(y0 - 6, 14)), FONT, 0.5, (255, 200, 0), 2)

        txt = f'Obstaculo: {dist:.2f} m' if dist > 0 else 'Camino libre'
        if hay_desnivel:
            txt += ' | DESNIVEL'
        txt += ' | semantica' if usa_sem else ' | solo geometria'
        cv2.putText(vis, self.fps_txt, (10, 30), FONT, 0.5, (0, 255, 255), 2)
        cv2.putText(vis, txt, (10, 60), FONT, 0.7, (255, 255, 255), 2)

        if self.publicar_visu and self.pub_visu.get_subscription_count() > 0:
            mv = self.bridge.cv2_to_imgmsg(vis, encoding='bgr8')
            mv.header = msg.header
            self.pub_visu.publish(mv)

        if self.debug_view:
            panel = cv2.applyColorMap(coste_full, cv2.COLORMAP_VIRIDIS)
            panel[coste_full == 0] = 0
            cv2.putText(panel, 'MAPA DE COSTE (amarillo = preferido)', (10, 30),
                        FONT, 0.6, (255, 255, 255), 2)
            cv2.imshow(WIN, np.hstack((vis, panel)))
            cv2.waitKey(1)


def main(args=None):
    rclpy.init(args=args)
    nodo = PercepcionFusion()
    try:
        rclpy.spin(nodo)
    except KeyboardInterrupt:
        pass
    finally:
        nodo.stop.set()
        cv2.destroyAllWindows()
        nodo.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()