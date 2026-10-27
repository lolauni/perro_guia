import glob
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
HUB_ID_ADE20K = "nvidia/segformer-b0-finetuned-ade-512-512"

# Nombres de clase (primer término de la etiqueta) que cuentan como transitables.
# Valen tanto para ADE20K (floor, sidewalk, earth) como Cityscapes (sidewalk, terrain)
WALKABLE_NAMES = {'floor', 'sidewalk', 'earth', 'terrain'}
ROAD_NAME = 'road'   # sobre ella se busca el paso de cebra

# Clases COCO que cuentan como obstáculo (persona, bici, coche, moto, bus, camión,
# semáforo, boca de incendios, stop, banco, pájaro, gato, perro, silla). Edita a gusto.
OBSTACLE_CLASSES = [0, 1, 2, 3, 5, 7, 9, 10, 11, 13, 14, 15, 16, 56]

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Orden de preferencia para cargar YOLO-seg (el primero que exista)
YOLO_CANDIDATOS = [
    'yolov8n-seg_ncnn_model',       # mejor opción en Raspberry Pi (ARM)
    'yolov8n-seg_openvino_model',   # útil en PC Intel
    'yolov8n-seg.pt',               # PyTorch (más lento)
]


def nombre_base(etiqueta):
    return etiqueta.split(',')[0].strip().lower()


class PercepcionCompleta(Node):
    def __init__(self):
        super().__init__('percepcion_completa')
        self.bridge = CvBridge()

        # Parámetros (se cambian con --ros-args -p nombre:=valor)
        self.declare_parameter('base_path', '/home/lola/Documentos/ROS2/supergiz_ws')
        self.declare_parameter('topico_camara', '/camera/camera/color/image_raw')
        self.declare_parameter('segformer_path', '')   # vacío = autodetectar
        self.declare_parameter('seg_w', 320)           # múltiplos de 32
        self.declare_parameter('seg_h', 256)
        self.declare_parameter('yolo_imgsz', 320)
        self.declare_parameter('yolo_conf', 0.35)
        self.declare_parameter('seg_every_n', 3)       # SegFormer cada N frames
        self.declare_parameter('debug_view', False)    # ventana imshow (solo PC)

        base = self.get_parameter('base_path').value
        self.seg_w = self.get_parameter('seg_w').value
        self.seg_h = self.get_parameter('seg_h').value
        self.imgsz = self.get_parameter('yolo_imgsz').value
        self.conf = self.get_parameter('yolo_conf').value
        self.seg_every_n = self.get_parameter('seg_every_n').value
        self.debug_view = self.get_parameter('debug_view').value

        torch.set_num_threads(4)
        torch.set_grad_enabled(False)

        # 1. YOLOv8-seg
        self.yolo = None
        for nombre in YOLO_CANDIDATOS:
            ruta = os.path.join(base, nombre)
            if os.path.exists(ruta):
                self.get_logger().info(f'Cargando YOLOv8-seg desde: {nombre}')
                self.yolo = YOLO(ruta, task='segment')
                break
        if self.yolo is None:
            raise FileNotFoundError(
                f'No hay modelo YOLO-seg en {base}. Ejecuta: '
                'yolo export model=yolov8n-seg.pt format=ncnn imgsz=320')

        # 2. SegFormer (carpeta local si existe; si no, ADE20K desde internet)
        seg_src = self.get_parameter('segformer_path').value
        if not seg_src:
            locales = sorted(glob.glob(os.path.join(base, 'segformer-b0-finetuned-*')))
            seg_src = locales[0] if locales else HUB_ID_ADE20K
        self.get_logger().info(f'Cargando SegFormer desde: {seg_src}')
        self.segformer = SegformerForSemanticSegmentation.from_pretrained(seg_src).eval()

        # IDs resueltos por nombre de clase
        id2label = {int(i): l for i, l in self.segformer.config.id2label.items()}
        walkable_ids = [i for i, l in id2label.items() if nombre_base(l) in WALKABLE_NAMES]
        self.road_id = next((i for i, l in id2label.items() if nombre_base(l) == ROAD_NAME), None)
        if not walkable_ids:
            raise RuntimeError('El modelo SegFormer no tiene clases transitables conocidas.')
        self.get_logger().info(
            'Transitables: ' + ', '.join(f'{i}={id2label[i]}' for i in walkable_ids)
            + f' | calzada: {self.road_id}')

        self.walkable_lut = np.zeros(256, dtype=bool)
        self.walkable_lut[walkable_ids] = True

        # Kernel de cierre para el paso de cebra (escalado a la resolución de trabajo)
        kw = max(3, int(25 * self.seg_w / 640))
        kh = max(3, int(15 * self.seg_w / 640))
        self.kernel_cebra = cv2.getStructuringElement(cv2.MORPH_RECT, (kw, kh))

        # 3. ROS 2
        self.sub = self.create_subscription(
            Image, self.get_parameter('topico_camara').value,
            self.listener_callback, qos_profile_sensor_data)
        self.pub_visu = self.create_publisher(Image, '/percepcion/imagen_visu', 1)
        self.pub_mascara = self.create_publisher(Image, '/percepcion/mascara_navegable', 1)

        # 4. Caché y métricas
        self.frame_count = 0
        self.floor_mask = None
        self.t_acc, self.n_acc = 0.0, 0

        self.get_logger().info('>>> Percepción lista <<<')

    # ------------------------------------------------------------------
    def calcular_suelo(self, cv_image):
        """SegFormer a baja resolución -> máscara transitable (seg_h x seg_w, 0/255)."""
        small = cv2.resize(cv_image, (self.seg_w, self.seg_h), interpolation=cv2.INTER_AREA)

        # Preprocesado manual (el SegformerImageProcessor es lento)
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        rgb = (rgb - MEAN) / STD
        x = torch.from_numpy(rgb.transpose(2, 0, 1)).unsqueeze(0)

        logits = self.segformer(pixel_values=x).logits           # (1, C, H/4, W/4)
        seg = logits.argmax(dim=1)[0].numpy().astype(np.uint8)   # argmax ANTES de escalar
        seg = cv2.resize(seg, (self.seg_w, self.seg_h), interpolation=cv2.INTER_NEAREST)

        mask = self.walkable_lut[seg].astype(np.uint8) * 255

        # Paso de cebra: blanco sobre la calzada + cierre morfológico
        if self.road_id is not None:
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            blanco = ((gray > 185) & (seg == self.road_id)).astype(np.uint8) * 255
            cebra = cv2.morphologyEx(blanco, cv2.MORPH_CLOSE, self.kernel_cebra)
            mask[cebra > 0] = 255
        return mask

    # ------------------------------------------------------------------
    def listener_callback(self, msg):
        try:
            t0 = time.perf_counter()
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            h, w = cv_image.shape[:2]

            # --- Suelo: solo cada N frames ---
            if self.floor_mask is None or self.frame_count % self.seg_every_n == 0:
                self.floor_mask = self.calcular_suelo(cv_image)
            self.frame_count += 1
            mask = self.floor_mask.copy()

            # --- Obstáculos: YOLOv8-seg (restamos el polígono real, no la caja) ---
            res = self.yolo(cv_image, imgsz=self.imgsz, conf=self.conf,
                            classes=OBSTACLE_CLASSES, verbose=False)[0]
            if res.masks is not None:
                escala = np.array([self.seg_w, self.seg_h], dtype=np.float32)
                for poly in res.masks.xyn:
                    if len(poly) >= 3:
                        cv2.fillPoly(mask, [(poly * escala).astype(np.int32)], 0)

            # --- Publicar máscara (solo si hay suscriptores) ---
            if self.pub_mascara.get_subscription_count() > 0:
                mask_full = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
                m = self.bridge.cv2_to_imgmsg(mask_full, encoding='mono8')
                m.header = msg.header
                self.pub_mascara.publish(m)

            # --- FPS medio cada 30 frames ---
            self.t_acc += time.perf_counter() - t0
            self.n_acc += 1
            if self.n_acc == 30:
                self.get_logger().info(f'{self.n_acc / self.t_acc:.1f} FPS')
                self.t_acc, self.n_acc = 0.0, 0

            # --- Visualización: solo en debug o si alguien escucha ---
            if self.debug_view or self.pub_visu.get_subscription_count() > 0:
                vis = res.plot(boxes=False)      # máscaras YOLO sobre la imagen
                floor_full = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
                sel = floor_full > 0
                vis[sel] = (0.6 * vis[sel] + 0.4 * np.array([0, 255, 0])).astype(np.uint8)

                if self.pub_visu.get_subscription_count() > 0:
                    mv = self.bridge.cv2_to_imgmsg(vis, encoding='bgr8')
                    mv.header = msg.header
                    self.pub_visu.publish(mv)

                if self.debug_view:
                    mask_3ch = cv2.cvtColor(floor_full, cv2.COLOR_GRAY2BGR)
                    cv2.putText(mask_3ch, 'MASCARA NAVEGABLE', (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    cv2.imshow('Supergiz - Percepcion', np.hstack((vis, mask_3ch)))
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