import os
import threading
import time
from collections import deque

import cv2
import numpy as np
import torch
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import Float32, Bool
from cv_bridge import CvBridge
from ultralytics import YOLO
from transformers import SegformerForSemanticSegmentation

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------
SEGFORMER_ADE20K = "nvidia/segformer-b0-finetuned-ade-512-512"

# Clases transitables por NOMBRE (valen para ADE20K y Cityscapes)
WALKABLE_NAMES = {'floor', 'sidewalk', 'earth', 'terrain'}
ROAD_NAME = 'road'   # sobre ella se busca el paso de cebra

# Clases COCO que cuentan como obstáculo: ["person", "bicycle", "car", "motorcycle", "bus", "truck", 
# "traffic light", "stop sign", "bench", "cat", "dog", "suitcase", "chair", "dining table"]
OBSTACLE_CLASSES = [0, 1, 2, 3, 5, 7, 9, 11, 13, 15, 16, 28, 56, 60]

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

YOLO_CANDIDATOS = [
    'yolov8n-seg_ncnn_model',
    'yolov8n-seg_openvino_model',
    'yolov8n-seg.pt',
]

WIN = 'Supergiz - Percepcion'
FONT = cv2.FONT_HERSHEY_SIMPLEX
# 0 nada | 1 navegable (verde) | 2 obstáculo 3D (rojo) | 3 desnivel (magenta)  [BGR]
PALETA = np.array([[0, 0, 0], [0, 255, 0], [0, 0, 255], [255, 0, 255]], dtype=np.uint8)

# Una sola imagen en cola: menos latencia si el nodo no da abasto
QOS_IMG = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                     history=HistoryPolicy.KEEP_LAST, depth=1)


def nombre_base(etiqueta):
    return etiqueta.split(',')[0].strip().lower()


class PercepcionCompleta(Node):
    """
    Dos velocidades en un solo nodo:
      - Hilo principal (a la velocidad de la cámara): geometría con profundidad
        (plano del suelo, obstáculos, desniveles), fusión de máscaras y ventana.
      - Hilo en segundo plano: YOLOv8-seg + SegFormer, que actualizan sus resultados
        cuando terminan.

    Publica:
      /percepcion/mascara_navegable    (mono8)   acera según SegFormer, menos obstáculos y desniveles
      /percepcion/imagen_visu          (bgr8)    imagen de demo (solo si alguien se suscribe)
      /percepcion/distancia_obstaculo  (Float32) metros al obstáculo más cercano en el pasillo central;
                                                 -1.0 = no hay; NaN = sin datos 3D fiables
      /percepcion/desnivel             (Bool)    escalón / bordillo / hueco delante
    """

    def __init__(self):
        super().__init__('percepcion_completa')
        self.bridge = CvBridge()

        # ---------------- Parámetros (--ros-args -p nombre:=valor) ----------------
        d = self.declare_parameter
        d('base_path', '/home/lola/Documentos/ROS2/supergiz_ws')
        d('topico_color', '/camera/camera/color/image_raw')
        d('topico_depth', '/camera/camera/aligned_depth_to_color/image_raw')
        d('topico_info', '/camera/camera/aligned_depth_to_color/camera_info')
        d('segformer', SEGFORMER_ADE20K)   # id del hub o ruta local
        d('seg_w', 320)                    # múltiplos de 32
        d('seg_h', 256)
        d('seg_every_n', 8)                # SegFormer cada N ciclos de IA
        d('neural_cada', 2)                # un ciclo de IA cada N frames de cámara
        d('yolo_imgsz', 320)
        d('yolo_conf', 0.35)
        d('torch_threads', 4)
        d('umbral_blanco', 185)            # brillo mínimo de las franjas del paso de cebra
        d('min_area_frac', 0.003)
        d('debug_view', True)              # ventana de demo
        # --- geometría 3D ---
        d('stride', 4)                     # 640x480 -> 160x120
        d('dist_min', 0.25)
        d('dist_max', 4.0)
        d('ransac_iters', 64)
        d('ransac_tol', 0.05)
        d('min_up', 0.5)                   # cos del ángulo máx. del plano respecto a la vertical
        d('altura_min', 0.1)               # m: altura mínima de la cámara sobre el suelo
        d('altura_max', 1.8)               # m: altura máxima de la cámara sobre el suelo
        d('obst_min', 0.08)                # m sobre el suelo para contar como obstáculo
        d('obst_alto_max', 2.2)
        d('desnivel_min', 0.10)            # m bajo el suelo para contar como desnivel
        d('min_pix', 12)                   # píxeles mínimos (obstáculo en el pasillo)
        d('min_pix_desnivel', 80)          # píxeles mínimos (desnivel)
        d('desn_frames', 3)                # frames seguidos para confirmar un desnivel

        g = lambda n: self.get_parameter(n).value
        base = g('base_path')
        self.seg_w, self.seg_h = g('seg_w'), g('seg_h')
        self.seg_every_n = g('seg_every_n')
        self.neural_cada = max(1, g('neural_cada'))
        self.imgsz, self.conf = g('yolo_imgsz'), g('yolo_conf')
        self.torch_threads = g('torch_threads')
        self.umbral_blanco = g('umbral_blanco')
        self.min_area_frac = g('min_area_frac')
        self.debug_view = g('debug_view')
        self.stride = g('stride')
        self.dist_min, self.dist_max = g('dist_min'), g('dist_max')
        self.ransac_iters, self.ransac_tol = g('ransac_iters'), g('ransac_tol')
        self.min_up = g('min_up')
        self.altura_min, self.altura_max = g('altura_min'), g('altura_max')
        self.obst_min, self.obst_alto_max = g('obst_min'), g('obst_alto_max')
        self.desnivel_min = g('desnivel_min')
        self.min_pix, self.min_pix_desnivel = g('min_pix'), g('min_pix_desnivel')
        self.desn_frames = g('desn_frames')

        # ---------------- Modelos ----------------
        self.yolo = None
        for nombre in YOLO_CANDIDATOS:
            ruta = os.path.join(base, nombre)
            if os.path.exists(ruta):
                self.get_logger().info(f'Cargando YOLOv8-seg desde: {nombre}')
                self.yolo = YOLO(ruta, task='segment')
                break
        if self.yolo is None:
            raise FileNotFoundError(f'No se encontró yolov8n-seg.pt en {base}')

        seg_src = g('segformer')
        self.get_logger().info(f'Cargando SegFormer desde: {seg_src}')
        self.segformer = SegformerForSemanticSegmentation.from_pretrained(seg_src).eval()

        id2label = {int(i): l for i, l in self.segformer.config.id2label.items()}
        n_cls = len(id2label)
        walkable_ids = [i for i, l in id2label.items() if nombre_base(l) in WALKABLE_NAMES]
        self.road_id = next((i for i, l in id2label.items() if nombre_base(l) == ROAD_NAME), None)
        if not walkable_ids:
            raise RuntimeError('El modelo SegFormer no tiene clases transitables conocidas.')
        self.get_logger().info(
            'Transitables: ' + ', '.join(f'{i}={id2label[i]}' for i in walkable_ids)
            + f' | calzada: {self.road_id}')

        self.walk_idx = torch.tensor(walkable_ids, dtype=torch.long)
        self.nonwalk_idx = torch.tensor(
            [i for i in range(n_cls) if i not in walkable_ids], dtype=torch.long)
        if self.road_id is not None:
            self.nonroad_idx = torch.tensor(
                [i for i in range(n_cls) if i != self.road_id], dtype=torch.long)

        self.kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self.kernel_cebra = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 9))
        self.kernel_geo = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

        # ---------------- Estado geometría ----------------
        self.rng = np.random.default_rng()
        self.fx = self.fy = self.cx = self.cy = None
        self.rx = self.ry = None
        self.shape_depth = None
        self.plano = None
        self.fallos = 0
        self.fallos_seguidos = 0
        self.depth_msg = None
        self.depth_t = 0.0
        self.hist_dist = deque(maxlen=5)
        self.cnt_desn = 0

        # ---------------- Estado hilo de IA ----------------
        self._lock = threading.Lock()
        self._ev = threading.Event()
        self._frame_neural = None
        self.sem_low = None          # máscara SegFormer (seg_h x seg_w, 0/255)
        self.polys = []              # [(polígono normalizado, clase)]
        self.names = {}
        self.yolo_ms = 0.0
        self.seg_ms = 0.0
        self.n_neural = 0
        self.n_ia_prev = 0
        self._running = True
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

        # ---------------- ROS 2 ----------------
        self.create_subscription(CameraInfo, g('topico_info'), self.info_cb, QOS_IMG)
        self.create_subscription(Image, g('topico_depth'), self.depth_cb, QOS_IMG)
        self.create_subscription(Image, g('topico_color'), self.color_cb, QOS_IMG)
        self.pub_mascara = self.create_publisher(Image, '/percepcion/mascara_navegable', 1)
        self.pub_visu = self.create_publisher(Image, '/percepcion/imagen_visu', 1)
        self.pub_dist = self.create_publisher(Float32, '/percepcion/distancia_obstaculo', 1)
        self.pub_desn = self.create_publisher(Bool, '/percepcion/desnivel', 1)

        # ---------------- Métricas y ventana ----------------
        self.n_in = 0
        self.n_frames = 0
        self.t_acc = 0.0
        self.t_log = time.perf_counter()
        self.hud1 = ''
        self.hud2 = ''

        if self.debug_view:
            try:
                cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(WIN, 1280, 480)
            except cv2.error:
                self.get_logger().warn('No se pudo abrir la ventana; sigo sin ella.')
                self.debug_view = False

        self.get_logger().info('>>> Percepción completa lista <<<')

    # =====================================================================
    # Callbacks sencillos
    # =====================================================================
    def info_cb(self, msg):
        if self.fx is None:
            self.fx, self.fy = msg.k[0], msg.k[4]
            self.cx, self.cy = msg.k[2], msg.k[5]
            self.get_logger().info(f'Intrínsecos: fx={self.fx:.1f} fy={self.fy:.1f} '
                                   f'cx={self.cx:.1f} cy={self.cy:.1f}')

    def depth_cb(self, msg):
        self.depth_msg = msg                  # solo se guarda; se procesa en color_cb
        self.depth_t = time.monotonic()

    # =====================================================================
    # Hilo de IA: YOLOv8-seg + SegFormer
    # =====================================================================
    def limpiar(self, mask):
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel_open)
        n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if n > 1:
            keep = np.zeros(n, dtype=bool)
            keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= self.min_area_frac * mask.size
            mask = keep[lab].astype(np.uint8) * 255
        return mask

    def calcular_suelo(self, frame):
        """SegFormer -> máscara transitable a baja resolución (seg_h x seg_w, 0/255)."""
        sw, sh = self.seg_w, self.seg_h
        small = cv2.resize(frame, (sw, sh), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        rgb = (rgb - MEAN) / STD
        x = torch.from_numpy(rgb.transpose(2, 0, 1)).unsqueeze(0)

        with torch.inference_mode():
            lg = self.segformer(pixel_values=x).logits[0]
            walk = (lg[self.walk_idx].max(dim=0).values
                    - lg[self.nonwalk_idx].max(dim=0).values)
            road = None
            if self.road_id is not None:
                road = lg[self.road_id] - lg[self.nonroad_idx].max(dim=0).values

        walk = cv2.resize(walk.numpy(), (sw, sh), interpolation=cv2.INTER_LINEAR)
        mask = self.limpiar((walk > 0).astype(np.uint8) * 255)

        if road is not None:
            road = cv2.resize(road.numpy(), (sw, sh), interpolation=cv2.INTER_LINEAR) > 0
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            blanco = ((gray > self.umbral_blanco) & road).astype(np.uint8) * 255
            cebra = cv2.morphologyEx(blanco, cv2.MORPH_CLOSE, self.kernel_cebra)
            mask[(cebra > 0) & road] = 255
        return mask

    def _worker(self):
        # Los hilos de torch se fijan DENTRO de este hilo (en otros hilos no se hereda)
        torch.set_num_threads(self.torch_threads)
        ciclo = 0
        while self._running:
            if not self._ev.wait(timeout=0.5):
                continue
            self._ev.clear()
            with self._lock:
                frame = self._frame_neural
            if frame is None:
                continue
            try:
                t = time.perf_counter()
                res = self.yolo(frame, imgsz=self.imgsz, conf=self.conf,
                                classes=OBSTACLE_CLASSES, verbose=False)[0]
                polys = []
                if res.masks is not None:
                    clases = res.boxes.cls.cpu().numpy().astype(int)
                    for poly, c in zip(res.masks.xyn, clases):
                        if len(poly) >= 3:
                            polys.append((poly.astype(np.float32), int(c)))
                y_ms = (time.perf_counter() - t) * 1000.0

                sem, s_ms = None, None
                if self.sem_low is None or ciclo % self.seg_every_n == 0:
                    t = time.perf_counter()
                    sem = self.calcular_suelo(frame)
                    s_ms = (time.perf_counter() - t) * 1000.0
                ciclo += 1

                with self._lock:
                    self.polys = polys
                    self.names = res.names
                    self.yolo_ms = y_ms
                    if sem is not None:
                        self.sem_low = sem
                        self.seg_ms = s_ms
                    self.n_neural += 1
            except Exception as e:
                self.get_logger().error(f'Error en el hilo de IA: {e}')

    # =====================================================================
    # Geometría con profundidad
    # =====================================================================
    def ajustar_suelo(self, X, Y, Z, valid, hs):
        """RANSAC vectorizado sobre la mitad inferior. Devuelve (normal, altura_camara)."""
        cand = valid.copy()
        cand[:int(hs * 0.4), :] = False
        cand &= (Z < 3.5)
        P = np.stack([X[cand], Y[cand], Z[cand]], axis=1)
        N = len(P)

        nuevo = None
        if N >= 150:
            if N > 3000:
                P = P[self.rng.choice(N, 3000, replace=False)]
                N = 3000
            K = self.ransac_iters
            idx = self.rng.integers(0, N, size=(K, 3))
            p0, p1, p2 = P[idx[:, 0]], P[idx[:, 1]], P[idx[:, 2]]
            nrm = np.cross(p1 - p0, p2 - p0)
            norma = np.linalg.norm(nrm, axis=1, keepdims=True)
            nrm = nrm / np.maximum(norma, 1e-9)
            nrm[nrm[:, 1] > 0] *= -1                       # normal hacia arriba (-Y óptico)
            dd = -np.einsum('kj,kj->k', nrm, p0)           # altura de la cámara sobre el plano
            ok = ((norma[:, 0] > 1e-6) & (-nrm[:, 1] > self.min_up)
                  & (dd > self.altura_min) & (dd < self.altura_max))
            dist = np.abs(P @ nrm.T + dd)                  # (N, K)
            inl = (dist < self.ransac_tol).sum(axis=0)
            inl[~ok] = 0
            mejor = int(inl.argmax())

            if inl[mejor] >= 0.2 * N:
                Q = P[dist[:, mejor] < self.ransac_tol].astype(np.float64)
                c = Q.mean(axis=0)
                _, _, vt = np.linalg.svd(Q - c, full_matrices=False)
                n = vt[-1]
                if n[1] > 0:
                    n = -n
                altura = -float(n @ c)
                if -n[1] > self.min_up and self.altura_min < altura < self.altura_max:
                    nuevo = (n, altura)

        if nuevo is None:
            self.fallos += 1
            self.fallos_seguidos += 1
            if self.fallos_seguidos > 10:                  # plano demasiado viejo
                self.plano = None
            return self.plano

        self.fallos_seguidos = 0
        if self.plano is None:
            self.plano = nuevo
        else:
            a = 0.3                                        # suavizado temporal
            n = a * nuevo[0] + (1 - a) * self.plano[0]
            n /= np.linalg.norm(n)
            self.plano = (n, a * nuevo[1] + (1 - a) * self.plano[1])
        return self.plano

    def geometria(self):
        """Devuelve (obst_u8, desn_u8, dist_cruda, desnivel_crudo) o None si no hay datos fiables."""
        msg = self.depth_msg
        if msg is None or self.fx is None or (time.monotonic() - self.depth_t) > 0.3:
            return None

        depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')   # uint16, mm
        H, W = depth.shape[:2]
        s = self.stride
        if self.rx is None or self.shape_depth != (H, W):
            self.shape_depth = (H, W)
            u = np.arange(0, W, s, dtype=np.float32)
            v = np.arange(0, H, s, dtype=np.float32)
            self.rx = ((u - self.cx) / self.fx)[None, :]
            self.ry = ((v - self.cy) / self.fy)[:, None]

        z = depth[::s, ::s].astype(np.float32) * 0.001
        hs, ws = z.shape
        valid = (z > self.dist_min) & (z < self.dist_max)
        X = self.rx * z
        Y = self.ry * z

        plano = self.ajustar_suelo(X, Y, z, valid, hs)
        if plano is None:
            return None
        nx, ny, nz = (float(v) for v in plano[0])
        h = nx * X + ny * Y + nz * z + float(plano[1])      # altura sobre el suelo (m)

        # Umbrales que crecen con la distancia (el ruido de la D435 también)
        obst = valid & (h > self.obst_min + 0.03 * z) & (h < self.obst_alto_max)
        desn = valid & (h < -(self.desnivel_min + 0.03 * z))
        obst_u8 = cv2.morphologyEx(obst.astype(np.uint8) * 255, cv2.MORPH_OPEN, self.kernel_geo)
        desn_u8 = cv2.morphologyEx(desn.astype(np.uint8) * 255, cv2.MORPH_OPEN, self.kernel_geo)

        c0, c1 = ws // 3, 2 * ws // 3                        # pasillo central
        corr = obst_u8[:, c0:c1] > 0
        dist = -1.0
        if int(corr.sum()) >= self.min_pix:
            dist = float(np.percentile(z[:, c0:c1][corr], 5))
        desn_raw = int((desn_u8[hs // 2:, c0:c1] > 0).sum()) >= self.min_pix_desnivel
        return obst_u8, desn_u8, dist, desn_raw

    # =====================================================================
    # Bucle principal: llega un frame de color
    # =====================================================================
    def color_cb(self, msg):
        try:
            t0 = time.perf_counter()
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            H, W = frame.shape[:2]
            self.n_in += 1

            # 1) Pasar el frame al hilo de IA (solo cada N frames)
            if self.n_in % self.neural_cada == 0:
                with self._lock:
                    self._frame_neural = frame.copy()
                self._ev.set()

            # 2) Geometría (rápida, en cada frame)
            geo = self.geometria()
            if geo is None:
                dist_pub, desn_pub = float('nan'), False
                self.hist_dist.clear()
                self.cnt_desn = 0
                self.get_logger().warn('Sin profundidad o sin suelo fiable: distancia = NaN.',
                                       throttle_duration_sec=3.0)
            else:
                _, _, dist_raw, desn_raw = geo
                self.hist_dist.append(dist_raw if dist_raw > 0 else float('inf'))
                m = float(np.median(self.hist_dist))
                dist_pub = -1.0 if np.isinf(m) else m
                self.cnt_desn = self.cnt_desn + 1 if desn_raw else 0
                desn_pub = self.cnt_desn >= self.desn_frames
            self.pub_dist.publish(Float32(data=dist_pub))
            self.pub_desn.publish(Bool(data=bool(desn_pub)))

            # 3) Fusión: semántica AND NOT (obstáculo 3D | desnivel | objetos YOLO)
            quiere_visu = self.debug_view or self.pub_visu.get_subscription_count() > 0
            quiere_mask = self.pub_mascara.get_subscription_count() > 0
            final = None
            polys, names = [], {}
            if quiere_visu or quiere_mask:
                with self._lock:
                    sem, polys, names = self.sem_low, self.polys, self.names
                if sem is None:
                    final = np.zeros((H, W), dtype=np.uint8)     # aún sin SegFormer
                else:
                    final = cv2.resize(sem, (W, H), interpolation=cv2.INTER_LINEAR)
                    final = cv2.threshold(final, 127, 255, cv2.THRESH_BINARY)[1]
                if geo is not None:
                    bloq = cv2.resize(cv2.bitwise_or(geo[0], geo[1]), (W, H),
                                      interpolation=cv2.INTER_NEAREST)
                    final[bloq > 0] = 0
                esc = np.array([W, H], dtype=np.float32)
                for poly, _ in polys:
                    cv2.fillPoly(final, [(poly * esc).astype(np.int32)], 0)

                if quiere_mask:
                    mm = self.bridge.cv2_to_imgmsg(final, encoding='mono8')
                    mm.header = msg.header
                    self.pub_mascara.publish(mm)

            # 4) Métricas (sin contar el dibujo)
            self.t_acc += time.perf_counter() - t0
            self.n_frames += 1
            ahora = time.perf_counter()
            if ahora - self.t_log >= 2.0:
                dt = ahora - self.t_log
                fps = self.n_frames / dt
                ms = 1000.0 * self.t_acc / self.n_frames
                with self._lock:
                    y_ms, s_ms, n_ia = self.yolo_ms, self.seg_ms, self.n_neural
                hz_ia = (n_ia - self.n_ia_prev) / dt
                self.n_ia_prev = n_ia
                self.hud1 = f'{fps:.1f} FPS | geo+fusion {ms:.0f} ms | fallos plano {self.fallos}'
                self.hud2 = f'IA {hz_ia:.1f} Hz | YOLO {y_ms:.0f} ms | SegFormer {s_ms:.0f} ms'
                self.get_logger().info(f'{self.hud1} || {self.hud2}')
                self.t_log, self.n_frames, self.t_acc = ahora, 0, 0.0

            # 5) Visualización (demo)
            if quiere_visu:
                self.dibujar(msg, frame, final, geo, polys, names, dist_pub, desn_pub, H, W)

        except Exception as e:
            self.get_logger().error(f'Error procesando percepción: {e}')

    # =====================================================================
    def dibujar(self, msg, frame, final, geo, polys, names, dist_pub, desn_pub, H, W):
        clase = np.zeros((H, W), dtype=np.uint8)
        clase[final > 0] = 1
        if geo is not None:
            o = cv2.resize(geo[0], (W, H), interpolation=cv2.INTER_NEAREST) > 0
            dz = cv2.resize(geo[1], (W, H), interpolation=cv2.INTER_NEAREST) > 0
            clase[o] = 2
            clase[dz] = 3

        mezcla = cv2.addWeighted(frame, 0.55, PALETA[clase], 0.45, 0)
        vis = frame.copy()
        np.copyto(vis, mezcla, where=(clase > 0)[:, :, None])

        esc = np.array([W, H], dtype=np.float32)
        for poly, c in polys:
            pts = (poly * esc).astype(np.int32)
            cv2.polylines(vis, [pts], True, (255, 200, 0), 2)
            org = (int(pts[0][0]), int(pts[0][1]))
            cv2.putText(vis, str(names.get(c, c)), org, FONT, 0.5, (255, 200, 0), 1)

        if np.isnan(dist_pub):
            estado = 'SIN DATOS 3D'
        elif dist_pub > 0:
            estado = f'Obstaculo: {dist_pub:.2f} m'
        else:
            estado = 'Camino libre'
        if desn_pub:
            estado += ' | DESNIVEL'
        cv2.putText(vis, self.hud1, (10, 25), FONT, 0.55, (0, 255, 255), 2)
        cv2.putText(vis, self.hud2, (10, 50), FONT, 0.55, (0, 255, 255), 2)
        cv2.putText(vis, estado, (10, 80), FONT, 0.7, (255, 255, 255), 2)

        if self.pub_visu.get_subscription_count() > 0:
            mv = self.bridge.cv2_to_imgmsg(vis, encoding='bgr8')
            mv.header = msg.header
            self.pub_visu.publish(mv)

        if self.debug_view:
            m3 = cv2.cvtColor(final, cv2.COLOR_GRAY2BGR)
            cv2.putText(m3, 'MASCARA NAVEGABLE', (10, 30), FONT, 0.6, (255, 255, 255), 2)
            cv2.imshow(WIN, np.hstack((vis, m3)))
            cv2.waitKey(1)

    def cerrar(self):
        self._running = False
        self._ev.set()
        self._thread.join(timeout=2.0)


def main(args=None):
    rclpy.init(args=args)
    nodo = PercepcionCompleta()
    try:
        rclpy.spin(nodo)
    except KeyboardInterrupt:
        pass
    nodo.cerrar()
    cv2.destroyAllWindows()
    nodo.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()