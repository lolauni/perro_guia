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
from sensor_msgs.msg import Image, CameraInfo, LaserScan, PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Float32, Bool
from cv_bridge import CvBridge
from ultralytics import YOLO
from transformers import SegformerForSemanticSegmentation

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------
SEGFORMER_ADE20K = "nvidia/segformer-b0-finetuned-ade-512-512"
# Entrenado con segments/sidewalk-semantic: separa acera, calzada, paso de cebra y bordillo
SEGFORMER_SIDEWALK = "tobiasc/segformer-b0-finetuned-segments-sidewalk"

# Clases COCO que cuentan como obstáculo:
# persona, bici, coche, moto, bus, camión, semáforo, boca de incendios, stop,
# banco, gato, perro, maleta, silla, maceta, mesa
OBSTACLE_CLASSES = [0, 1, 2, 3, 5, 7, 9, 10, 11, 13, 15, 16, 28, 56, 58, 60]

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

YOLO_CANDIDATOS = [
    'yolov8n-seg_ncnn_model',
    'yolov8n-seg_openvino_model',
    'yolov8n-seg.pt',
]

# Filtro guiado: ajusta los bordes de la máscara de SegFormer a los de la imagen
GF_RADIO = 6      # px (más alto = más suave)
GF_EPS = 1e-3     # menor = bordes más fieles a la imagen

WIN = 'Supergiz - Percepcion'
FONT = cv2.FONT_HERSHEY_SIMPLEX
# 0 nada | 1 navegable (verde) | 2 obstáculo 3D (rojo) | 3 desnivel (magenta)  [BGR]
PALETA = np.array([[0, 0, 0], [0, 255, 0], [0, 0, 255], [255, 0, 255]], dtype=np.uint8)
# Colores para la vista de diagnóstico de clases de SegFormer
PAL_CLS = np.random.RandomState(7).randint(40, 256, (256, 3)).astype(np.uint8)

# Una sola imagen en cola: menos latencia si el nodo no da abasto
QOS_IMG = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                     history=HistoryPolicy.KEEP_LAST, depth=1)


def nombre_base(etiqueta):
    return etiqueta.split(',')[0].strip().lower()


def texto(img, s, org, escala, color):
    """Texto con contorno negro: legible sobre cielo, asfalto o sombra."""
    cv2.putText(img, s, org, FONT, escala, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, s, org, FONT, escala, color, 1, cv2.LINE_AA)


def guided_filter(I, p, r, eps):
    """Filtro guiado (He et al.): ajusta p a los bordes de I. I y p: float32 en [0, 1]."""
    k = (2 * r + 1, 2 * r + 1)
    mean_I = cv2.boxFilter(I, -1, k)
    mean_p = cv2.boxFilter(p, -1, k)
    cov_Ip = cv2.boxFilter(I * p, -1, k) - mean_I * mean_p
    var_I = cv2.boxFilter(I * I, -1, k) - mean_I * mean_I
    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I
    return cv2.boxFilter(a, -1, k) * I + cv2.boxFilter(b, -1, k)


class PercepcionCompleta(Node):
    """
    Tres velocidades en un solo nodo:
      - Hilo principal (a la velocidad de la cámara): geometría con profundidad
        (plano del suelo, obstáculos, desniveles), fusión de máscaras y ventana.
      - Hilo YOLO: detecta objetos lo más rápido que puede, sin esperar a SegFormer.
      - Hilo SegFormer (opcional): actualiza la semántica cada `seg_periodo` segundos.

    Publica:
      /percepcion/mascara_navegable    (mono8)   suelo (ver `fuente_suelo`), menos obstáculos y desniveles
      /percepcion/imagen_visu          (bgr8)    imagen de demo (solo si alguien se suscribe)
      /percepcion/distancia_obstaculo  (Float32) metros al obstáculo más cercano en el pasillo central;
                                                 -1.0 = no hay; NaN = sin datos 3D fiables
      /percepcion/desnivel             (Bool)    escalón / bordillo / hueco delante
      /percepcion/nube_obstaculos      (PointCloud2) obstáculos + desniveles, para el costmap de Nav2
      /percepcion/nube_suelo           (PointCloud2) suelo navegable, para limpiar el costmap
      /percepcion/scan                 (LaserScan) obstáculos + desniveles como un láser 2D (frame `scan_frame`)
      /percepcion/baldosa_tactil       (Bool)    hay baldosa de botones a la vista (paso de peatones)
      /percepcion/distancia_tactil     (Float32) metros a la baldosa de botones; -1 = no hay; NaN = sin 3D
    """

    def __init__(self):
        super().__init__('percepcion_completa')
        self.bridge = CvBridge()

        # ---------------- Parámetros (--ros-args -p nombre:=valor) ----------------
        d = self.declare_parameter
        d('base_path', '/home/lola/Documentos/ROS2/supergiz_ws')
        d('topico_color', '/camera/camera/color/image_raw')
        d('topico_depth', '/camera/camera/aligned_depth_to_color/image_raw')
        d('topico_info', '/camera/camera/color/camera_info')   # = intrínsecos de la profundidad alineada
        # De dónde sale el suelo navegable:
        #   'geometria' -> plano del robot conectado con sus patas (corta en bordillos)
        #   'semantica' -> clases de SegFormer
        #   'ambas'     -> intersección de las dos
        d('fuente_suelo', 'geometria')
        d('segformer', SEGFORMER_SIDEWALK) # id del hub o ruta local (ADE20K, Cityscapes...)
        # Clases transitables por nombre; solo se usan las que existan en el modelo cargado
        d('walkable_names', ['acera', 'paso_cebra',                        # modelo propio (entrenamiento/)
                             'flat-sidewalk', 'flat-crosswalk',            # modelo de aceras
                             'floor', 'sidewalk', 'earth', 'terrain'])     # ADE20K / Cityscapes
        d('road_names', ['calzada', 'flat-road', 'road'])   # calzada: sobre ella se busca el paso de cebra
        d('seg_w', 448)                    # múltiplos de 32; 448x256 ~ 16:9 como la D435 a 424x240
        d('seg_h', 256)
        d('seg_periodo', 0.5)              # s entre pasadas de SegFormer (arranque a arranque)
        d('normalizar_luz', False)         # CLAHE sobre la luminancia (reduce el contraste de sombras)
        d('gf_activo', True)               # filtro guiado de bordes
        d('cuantizar', False)              # cuantización dinámica int8 de SegFormer (experimental)
        d('ver_clases', False)             # diagnóstico: muestra las clases que ve SegFormer
        d('threads_yolo', 2)               # hilos de torch para YOLO (si usas .pt)
        d('threads_seg', 2)                # hilos de torch para SegFormer
        d('yolo_imgsz', 320)
        d('yolo_conf', 0.35)
        d('margen_yolo', 6)                # px de margen alrededor de cada objeto
        # --- baldosa podotáctil de botones (GuideTWSI, YOLOv11-seg) ---
        d('tactil_modelo', 'modelos/yolo11n_tactile.pt')   # relativo a base_path; '' = desactivado
        d('tactil_conf', 0.25)
        d('tactil_imgsz', 480)
        d('tactil_periodo', 0.25)          # s entre pasadas
        d('tactil_vida', 1.0)              # s que se mantiene una detección
        d('umbral_blanco', 185)            # brillo mínimo de las franjas del paso de cebra
        d('min_area_frac', 0.003)
        d('debug_view', True)              # ventana de demo
        # --- geometría 3D ---
        d('stride', 0)                     # 0 = automático (2 si la imagen es <=480 px de ancho, 4 si no)
        d('dist_min', 0.25)
        d('dist_max', 4.0)
        d('ransac_iters', 64)
        d('ransac_tol', 0.05)
        d('fila_suelo', 0.6)               # el plano se ajusta solo por debajo de esta fila (~1 m delante)
        d('suelo_tol', 0.04)               # m (+1 cm por metro) para contar como el mismo suelo
        d('altura_nube_desnivel', 0.05)    # m sobre el suelo a los que se publican los desniveles
        # --- escaneo láser virtual (obstáculos + desniveles proyectados al suelo) ---
        d('scan_frame', 'camera_link')     # x delante, y izquierda, z arriba (frame de realsense2_camera)
        d('scan_angulo', 0.70)             # rad a cada lado (~FOV horizontal de la D435 en color)
        d('scan_paso', 0.0087)             # rad por haz (0.5°)
        d('min_up', 0.5)
        d('altura_min', 0.1)
        d('altura_max', 1.8)
        d('obst_min', 0.08)
        d('obst_alto_max', 2.2)
        d('desnivel_min', 0.10)
        d('min_pix', 12)
        d('min_pix_desnivel', 80)
        d('desn_frames', 3)

        g = lambda n: self.get_parameter(n).value
        base = g('base_path')
        self.seg_w, self.seg_h = g('seg_w'), g('seg_h')
        self.seg_periodo = g('seg_periodo')
        self.normalizar_luz = g('normalizar_luz')
        self.gf_activo = g('gf_activo')
        self.ver_clases = g('ver_clases')
        self.threads_yolo, self.threads_seg = g('threads_yolo'), g('threads_seg')
        self.imgsz, self.conf = g('yolo_imgsz'), g('yolo_conf')
        self.margen_yolo = int(g('margen_yolo'))
        self.tactil_conf, self.tactil_imgsz = g('tactil_conf'), g('tactil_imgsz')
        self.tactil_periodo, self.tactil_vida = g('tactil_periodo'), g('tactil_vida')
        self.umbral_blanco = g('umbral_blanco')
        self.min_area_frac = g('min_area_frac')
        self.debug_view = g('debug_view')
        self.stride = g('stride')
        self.dist_min, self.dist_max = g('dist_min'), g('dist_max')
        self.ransac_iters, self.ransac_tol = g('ransac_iters'), g('ransac_tol')
        self.fila_suelo, self.suelo_tol = g('fila_suelo'), g('suelo_tol')
        self.altura_nube_desnivel = g('altura_nube_desnivel')
        self.scan_frame = g('scan_frame')
        self.scan_angulo, self.scan_paso = g('scan_angulo'), g('scan_paso')
        self.fuente_suelo = g('fuente_suelo')
        if self.fuente_suelo not in ('geometria', 'semantica', 'ambas'):
            raise ValueError(f"fuente_suelo debe ser geometria, semantica o ambas (no '{self.fuente_suelo}')")
        # SegFormer solo se carga si hace falta: en la Raspberry ahorra mucha CPU
        self.usa_seg = self.fuente_suelo != 'geometria' or self.ver_clases
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

        self.yolo_tactil = None
        if g('tactil_modelo'):
            ruta = os.path.join(base, g('tactil_modelo'))
            if os.path.exists(ruta):
                self.get_logger().info(f'Cargando detector de baldosa táctil desde: {ruta}')
                self.yolo_tactil = YOLO(ruta, task='segment')
            else:
                self.get_logger().warn(f'No está {ruta}: sigo sin detector de baldosa táctil.')
        nombres = self.yolo.names       # con NCNN cada acceso a .names vuelve a cargar el modelo
        self.get_logger().info('Obstáculos: ' + ', '.join(
            f'{i}={nombres[i]}' for i in OBSTACLE_CLASSES))

        self.segformer = None
        self.walkable_ids = set()
        self.road_id = None
        if self.usa_seg:
            seg_src = g('segformer')
            self.get_logger().info(f'Cargando SegFormer desde: {seg_src}')
            self.segformer = SegformerForSemanticSegmentation.from_pretrained(seg_src).eval()
            if g('cuantizar'):
                try:
                    self.segformer = torch.ao.quantization.quantize_dynamic(
                        self.segformer, {torch.nn.Linear}, dtype=torch.qint8)
                    self.get_logger().info('SegFormer cuantizado a int8 (dinámico).')
                except Exception as e:
                    self.get_logger().warn(f'No se pudo cuantizar SegFormer: {e}')

            self.id2label = {int(i): l for i, l in self.segformer.config.id2label.items()}
            n_cls = len(self.id2label)
            nombres_ok = {str(n).strip().lower() for n in g('walkable_names')}
            walkable_ids = [i for i, l in self.id2label.items() if nombre_base(l) in nombres_ok]
            self.walkable_ids = set(walkable_ids)
            road_names = {str(n).strip().lower() for n in g('road_names')}
            self.road_id = next((i for i, l in self.id2label.items()
                                 if nombre_base(l) in road_names), None)
            if not walkable_ids:
                raise RuntimeError(
                    f'Ninguna clase de SegFormer coincide con walkable_names={sorted(nombres_ok)}. '
                    f'Clases del modelo: {[nombre_base(l) for l in self.id2label.values()]}')
            self.get_logger().info(
                'Transitables: ' + ', '.join(f'{i}={self.id2label[i]}' for i in walkable_ids)
                + f' | calzada: {self.road_id}')

            self.walk_idx = torch.tensor(walkable_ids, dtype=torch.long)
            self.nonwalk_idx = torch.tensor(
                [i for i in range(n_cls) if i not in self.walkable_ids], dtype=torch.long)
            if self.road_id is not None:
                self.nonroad_idx = torch.tensor(
                    [i for i in range(n_cls) if i != self.road_id], dtype=torch.long)

        self.kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self.kernel_geo = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        self.clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))

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

        # ---------------- Estado hilos de IA ----------------
        self._lock = threading.Lock()
        self._frame_latest = None
        self._frame_id = 0
        self.sem_low = None          # máscara SegFormer (0/255)
        self.seg_cls = None          # mapa de clases de SegFormer (solo para diagnóstico)
        self.polys = []              # [(polígono normalizado, clase)]
        self.names = {}
        self.yolo_ms = 0.0
        self.seg_ms = 0.0
        self.n_yolo = 0
        self.n_seg = 0
        self.n_yolo_prev = 0
        self.n_seg_prev = 0
        self._running = True
        self._th_yolo = threading.Thread(target=self._worker_yolo, daemon=True)
        self.tactil_polys = []       # polígonos normalizados de baldosa de botones
        self.tactil_t = 0.0          # time.monotonic() de la última detección
        self.tactil_ms = 0.0
        self.z_geo = None            # profundidad (m) a la resolución de la geometría; 0 = no válida
        self._th_seg = threading.Thread(target=self._worker_seg, daemon=True)
        self._th_tactil = threading.Thread(target=self._worker_tactil, daemon=True)
        self._th_yolo.start()
        if self.usa_seg:
            self._th_seg.start()
        if self.yolo_tactil is not None:
            self._th_tactil.start()

        # ---------------- ROS 2 ----------------
        self.create_subscription(CameraInfo, g('topico_info'), self.info_cb, QOS_IMG)
        self.create_subscription(Image, g('topico_depth'), self.depth_cb, QOS_IMG)
        self.create_subscription(Image, g('topico_color'), self.color_cb, QOS_IMG)
        self.pub_mascara = self.create_publisher(Image, '/percepcion/mascara_navegable', 1)
        self.pub_visu = self.create_publisher(Image, '/percepcion/imagen_visu', 1)
        self.pub_dist = self.create_publisher(Float32, '/percepcion/distancia_obstaculo', 1)
        self.pub_desn = self.create_publisher(Bool, '/percepcion/desnivel', 1)
        self.pub_nube_obst = self.create_publisher(PointCloud2, '/percepcion/nube_obstaculos', 1)
        self.pub_nube_suelo = self.create_publisher(PointCloud2, '/percepcion/nube_suelo', 1)
        self.pub_scan = self.create_publisher(LaserScan, '/percepcion/scan', 1)
        self.pub_tactil = self.create_publisher(Bool, '/percepcion/baldosa_tactil', 1)
        self.pub_dist_tactil = self.create_publisher(Float32, '/percepcion/distancia_tactil', 1)

        # ---------------- Métricas y ventana ----------------
        self.n_in = 0
        self.n_frames = 0
        self.t_acc = 0.0
        self.t_log = time.perf_counter()
        self.hud1 = ''
        self.hud2 = ''
        self.win_ajustada = False

        if self.debug_view:
            try:
                cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
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
    # SegFormer
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
        """SegFormer (+ bordes afinados) -> máscara transitable (H x W, 0/255)."""
        H, W = frame.shape[:2]
        sw, sh = self.seg_w, self.seg_h
        small = cv2.resize(frame, (sw, sh), interpolation=cv2.INTER_AREA)

        if self.normalizar_luz:
            lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB)
            lab[:, :, 0] = self.clahe.apply(lab[:, :, 0])
            small = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

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
            if self.ver_clases:
                self.seg_cls = lg.argmax(dim=0).numpy().astype(np.uint8)

        # Puntuación -> resolución completa (-> filtro guiado con la imagen en gris)
        score = cv2.resize(walk.numpy(), (W, H), interpolation=cv2.INTER_LINEAR)
        gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        if self.gf_activo:
            p = np.clip(score / 8.0 + 0.5, 0.0, 1.0).astype(np.float32)
            p = guided_filter(gris, p, GF_RADIO, GF_EPS)
            binaria = p > 0.5
        else:
            binaria = score > 0
        mask = self.limpiar(binaria.astype(np.uint8) * 255)

        # Paso de cebra, también a resolución completa
        if road is not None:
            road_f = cv2.resize(road.numpy(), (W, H), interpolation=cv2.INTER_LINEAR) > 0
            blanco = ((gris > self.umbral_blanco / 255.0) & road_f).astype(np.uint8) * 255
            kw = max(3, int(round(13 * W / sw)))
            kh = max(3, int(round(9 * H / sh)))
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kw, kh))
            cebra = cv2.morphologyEx(blanco, cv2.MORPH_CLOSE, kernel)
            mask[(cebra > 0) & road_f] = 255
        return mask

    # =====================================================================
    # Hilos de IA (independientes: YOLO nunca espera a SegFormer)
    # =====================================================================
    def _worker_yolo(self):
        torch.set_num_threads(self.threads_yolo)     # se fija DENTRO del hilo
        ultimo_id = -1
        while self._running:
            with self._lock:
                frame, fid = self._frame_latest, self._frame_id
            if frame is None or fid == ultimo_id:
                time.sleep(0.003)
                continue
            ultimo_id = fid
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
                ms = (time.perf_counter() - t) * 1000.0
                with self._lock:
                    self.polys = polys
                    self.names = res.names
                    self.yolo_ms = ms
                    self.n_yolo += 1
            except Exception as e:
                self.get_logger().error(f'Error en el hilo de YOLO: {e}')

    def _worker_tactil(self):
        """Baldosa de botones (marca pasos de peatones y bordes de andén) cada `tactil_periodo` s."""
        ultimo_id = -1
        t_ultimo = 0.0
        while self._running:
            with self._lock:
                frame, fid = self._frame_latest, self._frame_id
            toca = (time.perf_counter() - t_ultimo) >= self.tactil_periodo
            if frame is None or fid == ultimo_id or not toca:
                time.sleep(0.01)
                continue
            ultimo_id = fid
            try:
                t0 = time.perf_counter()
                t_ultimo = t0
                res = self.yolo_tactil(frame, imgsz=self.tactil_imgsz, conf=self.tactil_conf,
                                       verbose=False)[0]
                polys = []
                if res.masks is not None:
                    polys = [p.astype(np.float32) for p in res.masks.xyn if len(p) >= 3]
                ms = (time.perf_counter() - t0) * 1000.0
                with self._lock:
                    self.tactil_ms = ms
                    if polys:
                        self.tactil_polys = polys
                        self.tactil_t = time.monotonic()
            except Exception as e:
                self.get_logger().error(f'Error en el hilo de baldosa táctil: {e}')

    def distancia_poligonos(self, polys):
        """Distancia (m) al punto más cercano de unos polígonos normalizados; NaN si no hay profundidad."""
        z = self.z_geo
        if z is None:
            return float('nan')
        hs, ws = z.shape
        m = np.zeros((hs, ws), dtype=np.uint8)
        esc = np.array([ws, hs], dtype=np.float32)
        for poly in polys:
            cv2.fillPoly(m, [(poly * esc).astype(np.int32)], 1)
        zz = z[(m > 0) & (z > 0)]
        return float(np.percentile(zz, 5)) if len(zz) >= self.min_pix else float('nan')

    def _worker_seg(self):
        torch.set_num_threads(self.threads_seg)      # se fija DENTRO del hilo
        ultimo_id = -1
        t_ultimo = 0.0
        while self._running:
            with self._lock:
                frame, fid = self._frame_latest, self._frame_id
            toca = (time.perf_counter() - t_ultimo) >= self.seg_periodo
            if frame is None or fid == ultimo_id or not toca:
                time.sleep(0.01)
                continue
            ultimo_id = fid
            try:
                t0 = time.perf_counter()
                sem = self.calcular_suelo(frame)
                ms = (time.perf_counter() - t0) * 1000.0
                t_ultimo = t0                        # periodo medido de arranque a arranque
                with self._lock:
                    self.sem_low = sem
                    self.seg_ms = ms
                    self.n_seg += 1
            except Exception as e:
                self.get_logger().error(f'Error en el hilo de SegFormer: {e}')
                t_ultimo = time.perf_counter()

    # =====================================================================
    # Geometría con profundidad
    # =====================================================================
    def ajustar_suelo(self, X, Y, Z, valid, hs):
        """RANSAC vectorizado sobre el suelo cercano (donde pisa el robot). Devuelve (normal, altura_camara)."""
        cand = valid.copy()
        cand[:int(hs * self.fila_suelo), :] = False
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
        s = self.stride if self.stride > 0 else (2 if W <= 480 else 4)
        if self.rx is None or self.shape_depth != (H, W):
            self.shape_depth = (H, W)
            u = np.arange(0, W, s, dtype=np.float32)
            v = np.arange(0, H, s, dtype=np.float32)
            self.rx = ((u - self.cx) / self.fx)[None, :]
            self.ry = ((v - self.cy) / self.fy)[:, None]

        z = depth[::s, ::s].astype(np.float32) * 0.001
        hs, ws = z.shape
        valid = (z > self.dist_min) & (z < self.dist_max)
        self.z_geo = np.where(valid, z, 0.0)
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

        # Suelo navegable: mismo plano que el robot y conectado con sus patas.
        # Un bordillo (escalón arriba o abajo) o un obstáculo corta la conexión.
        suelo = valid & (np.abs(h) < self.suelo_tol + 0.01 * z)
        suelo_u8 = cv2.morphologyEx(suelo.astype(np.uint8) * 255, cv2.MORPH_OPEN, self.kernel_geo)
        suelo_u8 = self.conectado_al_robot(suelo_u8)

        self.publicar_nubes(msg.header, X, Y, z, h, plano[0], obst_u8, desn_u8, suelo_u8)
        return obst_u8, desn_u8, dist, desn_raw, suelo_u8

    def publicar_nubes(self, header, X, Y, Z, h, n, obst_u8, desn_u8, suelo_u8):
        """Nubes para los costmaps de Nav2, en el frame óptico de la cámara (solo si alguien escucha).
        Los desniveles se suben al plano del suelo (+ altura_nube_desnivel) para que el costmap,
        que filtra lo que queda por debajo del suelo, los marque como prohibidos."""
        if self.pub_nube_obst.get_subscription_count() > 0:
            o = obst_u8 > 0
            d = desn_u8 > 0
            P_obst = np.stack([X[o], Y[o], Z[o]], axis=1)
            P_desn = np.stack([X[d], Y[d], Z[d]], axis=1)
            P_desn -= (h[d] - self.altura_nube_desnivel)[:, None] * n[None, :].astype(np.float32)
            self.pub_nube_obst.publish(
                point_cloud2.create_cloud_xyz32(header, np.vstack([P_obst, P_desn])))
        if self.pub_scan.get_subscription_count() > 0:
            self.publicar_scan(header, X, Y, Z, n, (obst_u8 > 0) | (desn_u8 > 0))
        if self.pub_nube_suelo.get_subscription_count() > 0:
            s = suelo_u8 > 0
            self.pub_nube_suelo.publish(
                point_cloud2.create_cloud_xyz32(header, np.stack([X[s], Y[s], Z[s]], axis=1)))

    def publicar_scan(self, header, X, Y, Z, n, bloqueo):
        """LaserScan virtual: por cada ángulo, distancia sobre el suelo a lo más cercano que bloquea
        (obstáculo o desnivel). Cualquier navegador que entienda /scan ve así bordillos y huecos
        sin cambiar nada. Las distancias se miden en el plano del suelo, así que la inclinación
        de la cámara no las falsea."""
        n = n.astype(np.float32)
        f = np.array([0.0, 0.0, 1.0], np.float32) - n[2] * n      # eje óptico proyectado al suelo
        f /= np.linalg.norm(f)
        izq = np.cross(n, f)                                       # y = z × x
        P = np.stack([X[bloqueo], Y[bloqueo], Z[bloqueo]], axis=1)
        a, b = P @ f, P @ izq
        ang = np.arctan2(b, a)
        r = np.hypot(a, b)

        nh = int(round(2 * self.scan_angulo / self.scan_paso)) + 1
        rangos = np.full(nh, np.inf, np.float32)                   # inf = nada a la vista (REP 117)
        i = np.round((ang + self.scan_angulo) / self.scan_paso).astype(np.int64)
        ok = (i >= 0) & (i < nh)
        np.minimum.at(rangos, i[ok], r[ok].astype(np.float32))

        msg = LaserScan()
        msg.header.stamp = header.stamp
        msg.header.frame_id = self.scan_frame
        msg.angle_min, msg.angle_max = -self.scan_angulo, self.scan_angulo
        msg.angle_increment = self.scan_paso
        msg.range_min, msg.range_max = float(self.dist_min), float(self.dist_max)
        msg.ranges = rangos.tolist()
        self.pub_scan.publish(msg)

    @staticmethod
    def conectado_al_robot(suelo_u8):
        """Deja solo las zonas de suelo que tocan la franja inferior central (delante de las patas)."""
        _, lab = cv2.connectedComponents(suelo_u8, connectivity=4)
        hs, ws = suelo_u8.shape
        semilla = lab[int(hs * 0.85):, ws // 4:3 * ws // 4]
        ids = np.unique(semilla[semilla > 0])
        return np.isin(lab, ids).astype(np.uint8) * 255

    @staticmethod
    def ampliar(m, W, H):
        """Escala una máscara pequeña con bordes redondeados en vez de bloques."""
        g = cv2.resize(m, (W, H), interpolation=cv2.INTER_LINEAR)
        return cv2.threshold(g, 127, 255, cv2.THRESH_BINARY)[1]

    # =====================================================================
    # Bucle principal: llega un frame de color
    # =====================================================================
    def color_cb(self, msg):
        try:
            t0 = time.perf_counter()
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            H, W = frame.shape[:2]
            self.n_in += 1

            # 1) Dejar el frame disponible para los hilos de IA (sin copiar)
            with self._lock:
                self._frame_latest = frame
                self._frame_id = self.n_in

            # 2) Geometría (rápida, en cada frame)
            geo = self.geometria()
            if geo is None:
                dist_pub, desn_pub = float('nan'), False
                self.hist_dist.clear()
                self.cnt_desn = 0
                self.get_logger().warn('Sin profundidad o sin suelo fiable: distancia = NaN.',
                                       throttle_duration_sec=3.0)
            else:
                _, _, dist_raw, desn_raw, _ = geo
                self.hist_dist.append(dist_raw if dist_raw > 0 else float('inf'))
                m = float(np.median(self.hist_dist))
                dist_pub = -1.0 if np.isinf(m) else m
                self.cnt_desn = self.cnt_desn + 1 if desn_raw else 0
                desn_pub = self.cnt_desn >= self.desn_frames
            self.pub_dist.publish(Float32(data=dist_pub))
            self.pub_desn.publish(Bool(data=bool(desn_pub)))

            # 2b) Baldosa de botones: ¿hay delante y a qué distancia? (-1 = no hay; NaN = sin profundidad)
            t_polys, dist_t = [], -1.0
            if self.yolo_tactil is not None:
                with self._lock:
                    if time.monotonic() - self.tactil_t <= self.tactil_vida:
                        t_polys = self.tactil_polys
                if t_polys:
                    dist_t = self.distancia_poligonos(t_polys) if geo is not None else float('nan')
                self.pub_tactil.publish(Bool(data=bool(t_polys)))
                self.pub_dist_tactil.publish(Float32(data=dist_t))

            # 3) Fusión: semántica AND NOT (obstáculo 3D | desnivel | objetos YOLO)
            quiere_visu = self.debug_view or self.pub_visu.get_subscription_count() > 0
            quiere_mask = self.pub_mascara.get_subscription_count() > 0
            final = None
            polys, names = [], {}
            if quiere_visu or quiere_mask:
                with self._lock:
                    sem, polys, names = self.sem_low, self.polys, self.names
                vacio = np.zeros((H, W), dtype=np.uint8)
                suelo_geo = vacio if geo is None else self.ampliar(geo[4], W, H)
                if sem is None:
                    suelo_sem = vacio                             # aún sin SegFormer
                elif sem.shape[:2] == (H, W):
                    suelo_sem = sem.copy()
                else:
                    suelo_sem = self.ampliar(sem, W, H)
                if self.fuente_suelo == 'geometria':
                    final = suelo_geo
                elif self.fuente_suelo == 'semantica':
                    final = suelo_sem
                else:
                    final = cv2.bitwise_and(suelo_geo, suelo_sem)
                if geo is not None:
                    bloq = self.ampliar(cv2.bitwise_or(geo[0], geo[1]), W, H)
                    final[bloq > 0] = 0
                esc = np.array([W, H], dtype=np.float32)
                for poly, _ in polys:
                    pts = (poly * esc).astype(np.int32)
                    cv2.fillPoly(final, [pts], 0)
                    if self.margen_yolo > 0:                     # margen: compensa el retraso de YOLO
                        cv2.polylines(final, [pts], True, 0, self.margen_yolo)

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
                    y_ms, s_ms = self.yolo_ms, self.seg_ms
                    n_y, n_s = self.n_yolo, self.n_seg
                hz_y = (n_y - self.n_yolo_prev) / dt
                hz_s = (n_s - self.n_seg_prev) / dt
                self.n_yolo_prev, self.n_seg_prev = n_y, n_s
                self.hud1 = f'{fps:.1f} FPS | geo+fusion {ms:.0f} ms | fallos plano {self.fallos}'
                self.hud2 = (f'YOLO {hz_y:.1f} Hz {y_ms:.0f} ms | '
                             f'Seg {hz_s:.1f} Hz {s_ms:.0f} ms')
                if self.yolo_tactil is not None:
                    self.hud2 += f' | Tactil {self.tactil_ms:.0f} ms'
                self.get_logger().info(f'{self.hud1} || {self.hud2}')
                self.t_log, self.n_frames, self.t_acc = ahora, 0, 0.0

            # 5) Visualización (demo)
            if quiere_visu:
                self.dibujar(msg, frame, final, geo, polys, names, dist_pub, desn_pub, H, W,
                             t_polys, dist_t)

        except Exception as e:
            self.get_logger().error(f'Error procesando percepción: {e}')

    # =====================================================================
    def panel_clases(self, frame, H, W, k):
        """Diagnóstico: clases que ve SegFormer. Las marcadas con * cuentan como navegables."""
        cls = self.seg_cls
        if cls is None:
            return frame.copy()
        cmap = PAL_CLS[cv2.resize(cls, (W, H), interpolation=cv2.INTER_NEAREST)]
        m3 = cv2.addWeighted(frame, 0.45, cmap, 0.55, 0)
        ids, cnt = np.unique(cls, return_counts=True)
        y = int(24 * k)
        for i in ids[np.argsort(-cnt)][:6]:
            col = tuple(int(c) for c in PAL_CLS[i])
            cv2.rectangle(m3, (6, y - int(10 * k)), (6 + int(14 * k), y), col, -1)
            marca = '* ' if int(i) in self.walkable_ids else ''
            texto(m3, marca + nombre_base(self.id2label.get(int(i), str(i))),
                  (6 + int(20 * k), y), 0.5 * k, (255, 255, 255))
            y += int(20 * k)
        return m3

    def dibujar(self, msg, frame, final, geo, polys, names, dist_pub, desn_pub, H, W,
                t_polys=(), dist_t=-1.0):
        k = W / 640.0                                  # escala del texto según la resolución
        grosor = max(1, int(round(2 * k)))

        clase = np.zeros((H, W), dtype=np.uint8)
        clase[final > 0] = 1
        if geo is not None:
            o = self.ampliar(geo[0], W, H) > 0
            dz = self.ampliar(geo[1], W, H) > 0
            clase[o] = 2
            clase[dz] = 3

        mezcla = cv2.addWeighted(frame, 0.55, PALETA[clase], 0.45, 0)
        vis = frame.copy()
        np.copyto(vis, mezcla, where=(clase > 0)[:, :, None])

        esc = np.array([W, H], dtype=np.float32)
        for poly, c in polys:
            pts = (poly * esc).astype(np.int32)
            cv2.polylines(vis, [pts], True, (255, 200, 0), grosor)
            org = (int(pts[0][0]), int(pts[0][1]))
            texto(vis, str(names.get(c, c)), org, 0.5 * k, (255, 200, 0))
        for poly in t_polys:                           # baldosa de botones en amarillo
            pts = (poly * esc).astype(np.int32)
            cv2.polylines(vis, [pts], True, (0, 230, 255), grosor)

        if np.isnan(dist_pub):
            estado = 'SIN DATOS 3D'
        elif dist_pub > 0:
            estado = f'Obstaculo: {dist_pub:.2f} m'
        else:
            estado = 'Camino libre'
        if desn_pub:
            estado += ' | DESNIVEL'
        if len(t_polys):
            estado += ' | BOTONES' + ('' if np.isnan(dist_t) else f' {dist_t:.2f} m')
        texto(vis, self.hud1, (6, int(24 * k)), 0.55 * k, (0, 255, 255))
        texto(vis, self.hud2, (6, int(48 * k)), 0.55 * k, (0, 255, 255))
        texto(vis, estado, (6, int(78 * k)), 0.7 * k, (255, 255, 255))

        if self.pub_visu.get_subscription_count() > 0:
            mv = self.bridge.cv2_to_imgmsg(vis, encoding='bgr8')
            mv.header = msg.header
            self.pub_visu.publish(mv)

        if self.debug_view:
            m3 = cv2.cvtColor(final, cv2.COLOR_GRAY2BGR)
            texto(m3, 'MASCARA NAVEGABLE', (6, int(30 * k)), 0.6 * k, (255, 255, 255))
            paneles = [vis, m3]
            if self.ver_clases:
                pc = self.panel_clases(frame, H, W, k)
                texto(pc, 'CLASES SEGFORMER (* = navegable)', (6, H - int(8 * k)),
                      0.5 * k, (255, 255, 255))
                paneles = [vis, pc, m3]
            if not self.win_ajustada:
                esc_v = min(1.5, 1650.0 / (len(paneles) * W))
                cv2.resizeWindow(WIN, int(len(paneles) * W * esc_v), int(H * esc_v))
                self.win_ajustada = True
            cv2.imshow(WIN, np.hstack(paneles))
            cv2.waitKey(1)

    def cerrar(self):
        self._running = False
        self._th_yolo.join(timeout=2.0)
        for th in (self._th_seg, self._th_tactil):
            if th.is_alive():
                th.join(timeout=2.0)


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