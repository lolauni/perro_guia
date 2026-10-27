import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import torch
import time
import message_filters
from ultralytics import YOLO
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation

class Percepcion3D(Node):
    def __init__(self):
        super().__init__('percepcion_3d')
        self.bridge = CvBridge()

        # 1. Cargar modelos de IA
        self.get_logger().info('Cargando YOLOv8 y SegFormer ADE20K...')
        self.yolo_model = YOLO('/home/lola/ROS2_workspace/supergiz_ws/yolov8n-seg.pt')
        
        modelo_id = "nvidia/segformer-b0-finetuned-ade-512-512"
        self.processor = SegformerImageProcessor.from_pretrained(modelo_id)
        self.segformer_model = SegformerForSemanticSegmentation.from_pretrained(modelo_id)

        # 2. Suscriptores Sincronizados (Color + Profundidad RealSense D435)
        topico_color = '/camera/camera/color/image_raw'
        topico_depth = '/camera/camera/aligned_depth_to_color/image_raw'

        sub_color = message_filters.Subscriber(self, Image, topico_color)
        sub_depth = message_filters.Subscriber(self, Image, topico_depth)

        # Sincronización por marcas de tiempo aproximadas
        self.ts = message_filters.ApproximateTimeSynchronizer([sub_color, sub_depth], queue_size=10, slop=0.1)
        self.ts.registerCallback(self.sincro_callback)

        # 3. Publicadores de ROS 2
        self.pub_visu = self.create_publisher(Image, '/percepcion/imagen_visu', 10)
        self.pub_mascara = self.create_publisher(Image, '/percepcion/mascara_navegable', 10)

        # Mapa de colores para ADE20K (BGR)
        self.color_map = np.zeros((256, 3), dtype=np.uint8)
        self.color_map[6]  = [0, 0, 255]   # Rojo -> Carretera
        self.color_map[11] = [0, 255, 0]   # Verde -> Acera
        self.color_map[3]  = [0, 255, 0]   # Verde -> Baldosas
        self.color_map[12] = [0, 255, 0]   # Verde -> Tierra/Suelo firme

        self.get_logger().info('>>> Nodo Percepcion 3D Sincronizado Preparado <<<')

    def sincro_callback(self, msg_color, msg_depth):
        try:
            start_time = time.time()

            # Convertir imágenes de ROS a formato OpenCV
            cv_color = self.bridge.imgmsg_to_cv2(msg_color, desired_encoding='bgr8')
            depth_raw = self.bridge.imgmsg_to_cv2(msg_depth, desired_encoding='passthrough')

            # Convertir la matriz de profundidad (mm) a METROS
            depth_meters = depth_raw.astype(np.float32) / 1000.0
            h_orig, w_orig = cv_color.shape[:2]

            # =========================================================
            # PASO 1: SEGMENTACIÓN DEL SUELO (SegFormer)
            # =========================================================
            cv_small = cv2.resize(cv_color, (512, 512))
            rgb_small = cv2.cvtColor(cv_small, cv2.COLOR_BGR2RGB)
            inputs = self.processor(images=rgb_small, return_tensors="pt")

            with torch.no_grad():
                outputs = self.segformer_model(**inputs)

            logits = outputs.logits
            upsampled_logits = torch.nn.functional.interpolate(
                logits, size=(h_orig, w_orig), mode="bilinear", align_corners=False
            )
            seg_map = upsampled_logits.argmax(dim=1)[0].cpu().numpy()

            # Máscara binaria inicial
            mascara_binaria = np.zeros((h_orig, w_orig), dtype=np.uint8)
            zonas_transitables = (seg_map == 3) | (seg_map == 11) | (seg_map == 12)
            mascara_binaria[zonas_transitables] = 255

            # =========================================================
            # PASO 2: DETECCIÓN DE PASO DE CEBRA EN CARRETERA
            # =========================================================
            mask_carretera = (seg_map == 6).astype(np.uint8)
            gray = cv2.cvtColor(cv_color, cv2.COLOR_BGR2GRAY)
            _, thresh_blanco = cv2.threshold(gray, 185, 255, cv2.THRESH_BINARY)
            lineas_cebra = cv2.bitwise_and(thresh_blanco, thresh_blanco, mask=mask_carretera)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 15))
            paso_cebra_conectado = cv2.morphologyEx(lineas_cebra, cv2.MORPH_CLOSE, kernel)

            mascara_binaria[paso_cebra_conectado > 0] = 255

            # =========================================================
            # PASO 3: FILTRADO 3D Y MEDICIÓN DE DISTANCIAS
            # =========================================================
            # Ignorar aceras situadas a más de 4 metros de distancia
            mascara_binaria[depth_meters > 4.0] = 0

            # Calcular la distancia mediana real a la acera
            acera_indices = (mascara_binaria == 255) & (depth_meters > 0.1)
            distancia_media_acera = np.median(depth_meters[acera_indices]) if np.any(acera_indices) else 0.0

            # Overlay visual
            mask_suelo_color = self.color_map[seg_map]
            mask_suelo_color[paso_cebra_conectado > 0] = [255, 255, 0] # Cian para paso de cebra
            suelo_overlay = cv2.addWeighted(cv_color, 0.6, mask_suelo_color, 0.4, 0)

            # =========================================================
            # PASO 4: RESTA DE OBSTÁCULOS (YOLOv8)
            # =========================================================
            yolo_results = self.yolo_model(cv_color, verbose=False)
            final_frame = yolo_results[0].plot(img=suelo_overlay, boxes=True, masks=True)

            if yolo_results[0].masks is not None:
                for mask in yolo_results[0].masks.data:
                    mask_np = mask.cpu().numpy()
                    mask_resized = cv2.resize(mask_np, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)
                    mascara_binaria[mask_resized > 0.5] = 0

            # Telemetría en pantalla
            fps = 1.0 / (time.time() - start_time)
            cv2.putText(final_frame, f'FPS: {fps:.1f}', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.putText(final_frame, f'Distancia Acera: {distancia_media_acera:.2f} m', (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            # =========================================================
            # PASO 5: PUBLICACIÓN Y VISUALIZACIÓN DUAL
            # =========================================================
            self.pub_visu.publish(self.bridge.cv2_to_imgmsg(final_frame, encoding='bgr8'))
            self.pub_mascara.publish(self.bridge.cv2_to_imgmsg(mascara_binaria, encoding='mono8'))

            mascara_3ch = cv2.cvtColor(mascara_binaria, cv2.COLOR_GRAY2BGR)
            vista_combinada = np.hstack((final_frame, mascara_3ch))
            cv2.imshow("Supergiz - Percepcion 3D (Color vs Mascara)", vista_combinada)
            cv2.waitKey(1)

        except Exception as e:
            self.get_logger().error(f'Error en callback 3D: {e}')

def main(args=None):
    rclpy.init(args=args)
    nodo = Percepcion3D()
    try:
        rclpy.spin(nodo)
    except KeyboardInterrupt:
        pass
    cv2.destroyAllWindows()
    nodo.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()