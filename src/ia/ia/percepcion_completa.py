import os
import time

import cv2
import numpy as np
import torch
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from ultralytics import YOLO
from transformers import SegformerForSemanticSegmentation

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------
SEGFORMER_ADE20K = "nvidia/segformer-b0-finetuned-ade-512-512"

# Clases transitables por NOMBRE (así no dependemos de números de ID)
WALKABLE_NAMES = {'floor', 'sidewalk', 'earth', 'terrain'}
ROAD_NAME = 'road'   # sobre ella se busca el paso de cebra

# Clases COCO que cuentan como obstáculo (persona, bici, coche, moto, bus, camión,
# semáforo, boca de incendios, stop, banco, pájaro, gato, perro, silla). Edita a gusto.
OBSTACLE_CLASSES = [0, 1, 2, 3, 5, 7, 9, 10, 11, 13, 14, 15, 16, 56]

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Orden de preferencia para cargar YOLO-seg (el primero que exista)
YOLO_CANDIDATOS = [
    'yolov8n-seg_ncnn_model',       # mejor opción en Raspberry Pi
    'yolov8n-seg_openvino_model',   # PC Intel
    'yolov8n-seg.pt',               # PyTorch, funciona siempre
]

WIN = 'Supergiz - Percepcion'


def nombre_base(etiqueta):
    return etiqueta.split(',')[0].strip().lower()


class PercepcionCompleta(Node):
    def __init__(self):
        super().__init__('percepcion_completa')
        self.bridge = CvBridge()

        # Parámetros (se cambian con --ros-args -p nombre:=valor)
        self.declare_parameter('base_path', '/home/lola/Documentos/ROS2/supergiz_ws')
        self.declare_parameter('topico_camara', '/camera/camera/color/image_raw')
        self.declare_parameter('segformer', SEGFORMER_ADE20K)  # id del hub o ruta local
        self.declare_parameter('seg_w', 320)          # múltiplos de 32
        self.declare_parameter('seg_h', 256)
        self.declare_parameter('seg_every_n', 4)      # SegFormer cada N frames
        self.declare_parameter('yolo_imgsz', 320)
        self.declare_parameter('yolo_conf', 0.35)
        self.declare_parameter('torch_threads', 4)
        self.declare_parameter('debug_view', True)    # ventana cámara + máscara
        self.declare_parameter('umbral_blanco', 185)  # brillo mínimo de las franjas
        self.declare_parameter('min_area_frac', 0.003)

        base = self.get_parameter('base_path').value
        self.seg_w = self.get_parameter('seg_w').value
        self.seg_h = self.get_parameter('seg_h').value
        self.seg_every_n = self.get_parameter('seg_every_n').value
        self.imgsz = self.get_parameter('yolo_imgsz').value
        self.conf = self.get_parameter('yolo_conf').value
        self.debug_view = self.get_parameter('debug_view').value
        self.umbral_blanco = self.get_parameter('umbral_blanco').value
        self.min_area_frac = self.get_parameter('min_area_frac').value

        torch.set_num_threads(self.get_parameter('torch_threads').value)

        # 1. YOLOv8-seg
        self.yolo = None
        for nombre in YOLO_CANDIDATOS:
            ruta = os.path.join(base, nombre)
            if os.path.exists(ruta):
                self.get_logger().info(f'Cargando YOLOv8-seg desde: {nombre}')
                self.yolo = YOLO(ruta, task='segment')
                break
        if self.yolo is None:
            raise FileNotFoundError(f'No se encontró yolov8n-seg.pt en {base}')

        # 2. SegFormer
        seg_src = self.get_parameter('segformer').value
        self.get_logger().info(f'Cargando SegFormer desde: {seg_src}')
        self.segformer = SegformerForSemanticSegmentation.from_pretrained(seg_src).eval()

        # IDs resueltos por nombre de clase
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

        # 3. ROS 2
        self.sub = self.create_subscription(
            Image, self.get_parameter('topico_camara').value,
            self.listener_callback, qos_profile_sensor_data)
        self.pub_visu = self.create_publisher(Image, '/percepcion/imagen_visu', 1)
        self.pub_mascara = self.create_publisher(Image, '/percepcion/mascara_navegable', 1)

        # 4. Estado y métricas
        self.frame_count = 0
        self.floor_low = None          # máscara de suelo a baja resolución
        self.t_log = time.perf_counter()
        self.n_frames = 0
        self.t_yolo = 0.0
        self.t_seg = 0.0
        self.n_seg = 0
        self.fps_txt = ''

        if self.debug_view:
            try:
                cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(WIN, 1280, 480)
            except cv2.error:
                self.get_logger().warn('No se pudo abrir la ventana; sigo sin ella.')
                self.debug_view = False

        self.get_logger().info('>>> Percepción lista <<<')

    # ------------------------------------------------------------------
    def limpiar(self, mask):
        """Quita ruido (a baja resolución, es barato)."""
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
            lg = self.segformer(pixel_values=x).logits[0]     # (C, H/4, W/4)
            walk = (lg[self.walk_idx].max(dim=0).values
                    - lg[self.nonwalk_idx].max(dim=0).values)
            road = None
            if self.road_id is not None:
                road = lg[self.road_id] - lg[self.nonroad_idx].max(dim=0).values

        walk = cv2.resize(walk.numpy(), (sw, sh), interpolation=cv2.INTER_LINEAR)
        mask = self.limpiar((walk > 0).astype(np.uint8) * 255)

        # Paso de cebra: blanco sobre la calzada, cierre para unir las franjas
        if road is not None:
            road = cv2.resize(road.numpy(), (sw, sh), interpolation=cv2.INTER_LINEAR) > 0
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            blanco = ((gray > self.umbral_blanco) & road).astype(np.uint8) * 255
            cebra = cv2.morphologyEx(blanco, cv2.MORPH_CLOSE, self.kernel_cebra)
            mask[(cebra > 0) & road] = 255
        return mask

    # ------------------------------------------------------------------
    def listener_callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            h, w = frame.shape[:2]

            # --- Suelo (SegFormer) cada N frames ---
            if self.floor_low is None or self.frame_count % self.seg_every_n == 0:
                t = time.perf_counter()
                self.floor_low = self.calcular_suelo(frame)
                self.t_seg += time.perf_counter() - t
                self.n_seg += 1
            self.frame_count += 1

            # Escalado con bordes suaves: bilineal + umbral
            mask = cv2.resize(self.floor_low, (w, h), interpolation=cv2.INTER_LINEAR)
            mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)[1]

            # --- Obstáculos (YOLOv8-seg): se resta el contorno real del objeto ---
            t = time.perf_counter()
            res = self.yolo(frame, imgsz=self.imgsz, conf=self.conf,
                            classes=OBSTACLE_CLASSES, verbose=False)[0]
            self.t_yolo += time.perf_counter() - t

            if res.masks is not None:
                esc = np.array([w, h], dtype=np.float32)
                for poly in res.masks.xyn:
                    if len(poly) >= 3:
                        cv2.fillPoly(mask, [(poly * esc).astype(np.int32)], 0)

            # --- Publicar máscara (solo si hay suscriptores) ---
            if self.pub_mascara.get_subscription_count() > 0:
                m = self.bridge.cv2_to_imgmsg(mask, encoding='mono8')
                m.header = msg.header
                self.pub_mascara.publish(m)

            # --- Métricas cada 2 s ---
            self.n_frames += 1
            ahora = time.perf_counter()
            if ahora - self.t_log >= 2.0:
                fps = self.n_frames / (ahora - self.t_log)
                y_ms = 1000.0 * self.t_yolo / self.n_frames
                s_ms = 1000.0 * self.t_seg / max(self.n_seg, 1)
                self.fps_txt = f'{fps:.1f} FPS | YOLO {y_ms:.0f} ms | SegFormer {s_ms:.0f} ms'
                self.get_logger().info(self.fps_txt)
                self.t_log, self.n_frames = ahora, 0
                self.t_yolo, self.t_seg, self.n_seg = 0.0, 0.0, 0

            # --- Visualización ---
            if self.debug_view or self.pub_visu.get_subscription_count() > 0:
                vis = res.plot(boxes=False)
                sel = mask > 0
                vis[sel] = (0.6 * vis[sel] + 0.4 * np.array([0, 255, 0])).astype(np.uint8)
                cv2.putText(vis, self.fps_txt, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                if self.pub_visu.get_subscription_count() > 0:
                    mv = self.bridge.cv2_to_imgmsg(vis, encoding='bgr8')
                    mv.header = msg.header
                    self.pub_visu.publish(mv)

                if self.debug_view:
                    mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
                    cv2.putText(mask_3ch, 'MASCARA NAVEGABLE', (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    cv2.imshow(WIN, np.hstack((vis, mask_3ch)))
                    cv2.waitKey(1)

        except Exception as e:
            self.get_logger().error(f'Error procesando percepción: {e}')


def main(args=None):
    rclpy.init(args=args)
    nodo = PercepcionCompleta()
    try:
        rclpy.spin(nodo)
    except KeyboardInterrupt:
        pass
    cv2.destroyAllWindows()
    nodo.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()