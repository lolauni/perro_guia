"""
PERECEPCION COMPLETA USANDO YOLO Y CITYSCAPES

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation

class PercepcionCompleta(Node):
    def __init__(self):
        super().__init__('percepcion_completa')
        self.bridge = CvBridge()

        # 1. Cargar modelo YOLOv8 local (Obstáculos / Personas / Vehículos)
        self.get_logger().info('Cargando YOLOv8...')
        self.yolo_model = YOLO('/home/lola/ROS2_workspace/supergiz_ws/yolov8n-seg.pt')


        # 2. Cargar modelo SegFormer local (Superficie / Suelo Cityscapes)
        self.get_logger().info('Cargando SegFormer local...')
        modelo_id = "/home/lola/ROS2_workspace/supergiz_ws/segformer-b0-finetuned-cityscapes-512-1024"
        #modelo_id = "nvidia/segformer-b0-finetuned-ade-512-512"
        self.processor = SegformerImageProcessor.from_pretrained(modelo_id)
        self.segformer_model = SegformerForSemanticSegmentation.from_pretrained(modelo_id)

        # Suscriptor al tópico de cámara
        self.subscription = self.create_subscription(
            Image,
            '/imagen_camara',
            self.listener_callback,
            10
        )
    
        # Mapa de colores para Cityscapes (BGR)
        # 0: Road (Rojo - Zona no transitable para el perro)
        # 1: Sidewalk (Verde - Zona segura)
        self.color_map = np.zeros((256, 3), dtype=np.uint8)
        self.color_map[0] = [0, 0, 255]   # Rojo (Carretera)
        self.color_map[1] = [0, 255, 0]   # Verde (Acera)

        self.get_logger().info('>>> Sistema de Percepción Completa Preparado <<<')

    def listener_callback(self, msg):
        self.get_logger().info('Procesando imagen con pipeline dual (SegFormer + YOLOv8)...')
        try:
            # Convertir mensaje de ROS a imagen OpenCV
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')

            # ==========================================
            # PASO 1: SEGMENTACIÓN SEMÁNTICA DEL SUELO
            # ==========================================
            rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            inputs = self.processor(images=rgb_image, return_tensors="pt")
            
            with torch.no_grad():
                outputs = self.segformer_model(**inputs)

            logits = outputs.logits
            upsampled_logits = torch.nn.functional.interpolate(
                logits,
                size=cv_image.shape[:2],
                mode="bilinear",
                align_corners=False
            )
            seg_map = upsampled_logits.argmax(dim=1)[0].cpu().numpy()
            mask_suelo = self.color_map[seg_map]

            # Fusionar imagen original con la máscara del suelo
            alpha = 0.4
            suelo_overlay = cv2.addWeighted(cv_image, 1 - alpha, mask_suelo, alpha, 0)

            # ==========================================
            # PASO 2: DETECCIÓN DE OBJETOS CON YOLOv8
            # ==========================================
            yolo_results = self.yolo_model(cv_image)

            # Dibujar las máscaras/cajas de YOLO encima del suelo coloreado
            final_frame = yolo_results[0].plot(img=suelo_overlay, boxes=True, masks=True)

            # Guardar el resultado final combinado
            cv2.imwrite('resultado_completo.jpg', final_frame)
            self.get_logger().info('¡Inferencia completada! Guardado en resultado_completo.jpg')

        except Exception as e:
            self.get_logger().error(f'Error procesando percepcion: {e}')

def main(args=None):
    rclpy.init(args=args)
    nodo = PercepcionCompleta()
    rclpy.spin(nodo)
    nodo.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

"""

"""
PERECEPCION COMPLETA USANDO YOLO Y ADE20K

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation

class PercepcionCompleta(Node):
    def __init__(self):
        super().__init__('percepcion_completa')
        self.bridge = CvBridge()

        # 1. Cargar modelo YOLOv8 local (Obstáculos / Personas / Vehículos)
        self.get_logger().info('Cargando YOLOv8...')
        self.yolo_model = YOLO('/home/lola/ROS2_workspace/supergiz_ws/yolov8n-seg.pt')

        # 2. Cargar modelo SegFormer ADE20K (150 clases)
        self.get_logger().info('Cargando SegFormer ADE20K...')
        modelo_id = "nvidia/segformer-b0-finetuned-ade-512-512"
        self.processor = SegformerImageProcessor.from_pretrained(modelo_id)
        self.segformer_model = SegformerForSemanticSegmentation.from_pretrained(modelo_id)

        # Suscriptor al tópico de cámara 
        # (Usa '/imagen_camara' para pruebas con publicador_foto o 
        #  '/camera/camera/color/image_raw' para la RealSense D435)
        self.subscription = self.create_subscription(
            Image,
            '/imagen_camara',
            self.listener_callback,
            10
        )

        # Mapa de colores filtrado para ADE20K (Formato BGR para OpenCV)
        # Inicializado en cero: edificios, paredes, cielo y objetos quedan transparentes [0, 0, 0]
        self.color_map = np.zeros((256, 3), dtype=np.uint8)

        # --- ZONAS NO TRANSITABLES (Rojo) ---
        self.color_map[6]  = [0, 0, 255]   # Clase 6: Carretera / Calzada (Road)

        # --- ZONAS TRANSITABLES (Verde) ---
        self.color_map[11] = [0, 255, 0]   # Clase 11: Acera / Pavimento (Sidewalk)
        self.color_map[3]  = [0, 255, 0]   # Clase 3: Baldosas / Suelo peatonal (Floor)
        self.color_map[12] = [0, 255, 0]   # Clase 12: Tierra / Suelo firme (Earth/Ground)

        self.get_logger().info('>>> Sistema de Percepción Completa Preparado (ADE20K + YOLOv8) <<<')

    def listener_callback(self, msg):
        self.get_logger().info('Procesando imagen con pipeline dual...')
        try:
            # Convertir mensaje de ROS a imagen OpenCV
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            h_orig, w_orig = cv_image.shape[:2]

            # ==========================================
            # PASO 1: SEGMENTACIÓN SEMÁNTICA DEL SUELO
            # ==========================================
            # Reducir resolución para acelerar la inferencia de SegFormer
            cv_image_small = cv2.resize(cv_image, (512, 512))
            rgb_image = cv2.cvtColor(cv_image_small, cv2.COLOR_BGR2RGB)

            inputs = self.processor(images=rgb_image, return_tensors="pt")
            
            with torch.no_grad():
                outputs = self.segformer_model(**inputs)

            # Redimensionar la máscara resultante de vuelta a la resolución original de la imagen
            logits = outputs.logits
            upsampled_logits = torch.nn.functional.interpolate(
                logits,
                size=(h_orig, w_orig),
                mode="bilinear",
                align_corners=False
            )
            seg_map = upsampled_logits.argmax(dim=1)[0].cpu().numpy()

            # Pintar la máscara aplicando únicamente las clases filtradas
            mask_suelo = self.color_map[seg_map]

            # Fusionar imagen original con la máscara del suelo
            alpha = 0.4
            suelo_overlay = cv2.addWeighted(cv_image, 1 - alpha, mask_suelo, alpha, 0)

            # ==========================================
            # PASO 2: DETECCIÓN DE OBJETOS CON YOLOv8
            # ==========================================
            yolo_results = self.yolo_model(cv_image)

            # Dibujar las detecciones de YOLO encima del suelo ya procesado
            final_frame = yolo_results[0].plot(img=suelo_overlay, boxes=True, masks=True)

            # Guardar la imagen final combinada
            cv2.imwrite('resultado_completo.jpg', final_frame)
            self.get_logger().info('¡Inferencia completada! Guardado en resultado_completo.jpg')

        except Exception as e:
            self.get_logger().error(f'Error en el callback de percepcion_completa: {e}')

def main(args=None):
    rclpy.init(args=args)
    nodo = PercepcionCompleta()
    rclpy.spin(nodo)
    nodo.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

"""


#ESTA VERSION FUNCIONA BIEN, HACE UNA MASCARA BINARIA DEL SUELO TRANSITABLE Y AÑADE EL PASO DE CEBRA DETECTADO SOBRE LA CALZADA, ADEMÁS DE DETECTAR OBSTACULOS CON YOLOv8
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import torch
import time
from ultralytics import YOLO
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation

class PercepcionCompleta(Node):
    def __init__(self):
        super().__init__('percepcion_completa')
        self.bridge = CvBridge()

        # 1. Cargar YOLOv8 local
        self.get_logger().info('Cargando YOLOv8...')
        self.yolo_model = YOLO('/home/lola/ROS2_workspace/supergiz_ws/yolov8n-seg.pt')

        # 2. Cargar SegFormer ADE20K
        self.get_logger().info('Cargando SegFormer ADE20K...')
        modelo_id = "nvidia/segformer-b0-finetuned-ade-512-512"
        self.processor = SegformerImageProcessor.from_pretrained(modelo_id)
        self.segformer_model = SegformerForSemanticSegmentation.from_pretrained(modelo_id)

        # 3. Suscriptor al tópico de cámara
        #topico_camara = '/imagen_camara' #esto es para hacer pruebas con imagenes cargadas
        topico_camara = '/camera/camera/color/image_raw'
        self.subscription = self.create_subscription(
            Image,
            topico_camara,
            self.listener_callback,
            10
        )

        # 4. Publicadores de ROS 2
        self.pub_visu = self.create_publisher(Image, '/percepcion/imagen_visu', 10)
        self.pub_mascara = self.create_publisher(Image, '/percepcion/mascara_navegable', 10)

        # Mapa de colores para visualización BGR
        self.color_map = np.zeros((256, 3), dtype=np.uint8)
        self.color_map[6]  = [0, 0, 255]   # Rojo -> Carretera
        self.color_map[11] = [0, 255, 0]   # Verde -> Acera
        self.color_map[3]  = [0, 255, 0]   # Verde -> Baldosas
        self.color_map[12] = [0, 255, 0]   # Verde -> Tierra/Suelo firme

        self.get_logger().info('>>> Percepción lista (Suelo + Detección de Pasos de Cebra) <<<')

    def listener_callback(self, msg):
        try:
            start_time = time.time()

            # Convertir mensaje de ROS a imagen OpenCV
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            h_orig, w_orig = cv_image.shape[:2]

            # =========================================================
            # PASO 1: SEGMENTACIÓN SEMÁNTICA DEL SUELO (SegFormer)
            # =========================================================
            cv_image_small = cv2.resize(cv_image, (512, 512))
            rgb_image = cv2.cvtColor(cv_image_small, cv2.COLOR_BGR2RGB)
            inputs = self.processor(images=rgb_image, return_tensors="pt")

            with torch.no_grad():
                outputs = self.segformer_model(**inputs)

            logits = outputs.logits
            upsampled_logits = torch.nn.functional.interpolate(
                logits,
                size=(h_orig, w_orig),
                mode="bilinear",
                align_corners=False
            )
            seg_map = upsampled_logits.argmax(dim=1)[0].cpu().numpy()

            # Máscara binaria inicial (3: Floor, 11: Sidewalk, 12: Earth)
            mascara_binaria = np.zeros((h_orig, w_orig), dtype=np.uint8)
            zonas_transitables = (seg_map == 3) | (seg_map == 11) | (seg_map == 12)
            mascara_binaria[zonas_transitables] = 255

            # =========================================================
            # PASO 2: DETECCIÓN DE PASOS DE CEBRA SOBRE LA CALZADA
            # =========================================================
            mask_carretera = (seg_map == 6).astype(np.uint8)
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

            # Umbral de brillo para detectar las líneas blancas (> 185)
            _, thresh_blanco = cv2.threshold(gray, 185, 255, cv2.THRESH_BINARY)

            # Solo líneas brillantes situadas exactamente sobre la carretera
            lineas_cebra = cv2.bitwise_and(thresh_blanco, thresh_blanco, mask=mask_carretera)

            # Morfología para conectar las franjas adyacentes
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 15))
            paso_cebra_conectado = cv2.morphologyEx(lineas_cebra, cv2.MORPH_CLOSE, kernel)

            # Añadir el paso de cebra a la máscara binaria navegable
            mascara_binaria[paso_cebra_conectado > 0] = 255

            # Superposición gráfica de color
            mask_suelo_color = self.color_map[seg_map]
            # Pintar el paso de cebra detectado de color CIAN [255, 255, 0]
            mask_suelo_color[paso_cebra_conectado > 0] = [255, 255, 0]

            alpha = 0.4
            suelo_overlay = cv2.addWeighted(cv_image, 1 - alpha, mask_suelo_color, alpha, 0)

            # =========================================================
            # PASO 3: DETECCIÓN Y RESTA DE OBSTÁCULOS (YOLOv8)
            # =========================================================
            yolo_results = self.yolo_model(cv_image, verbose=False)
            final_frame = yolo_results[0].plot(img=suelo_overlay, boxes=True, masks=True)

            if yolo_results[0].masks is not None:
                for mask in yolo_results[0].masks.data:
                    mask_np = mask.cpu().numpy()
                    mask_resized = cv2.resize(mask_np, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)
                    mascara_binaria[mask_resized > 0.5] = 0

            # Calcular FPS
            current_time = time.time()
            fps = 1.0 / (current_time - start_time)
            cv2.putText(
                final_frame, 
                f'FPS: {fps:.1f}', 
                (20, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                1.0, 
                (0, 255, 255), 
                2
            )

            # =========================================================
            # PASO 4: PUBLICACIÓN EN TÓPICOS ROS 2
            # =========================================================
            msg_visu = self.bridge.cv2_to_imgmsg(final_frame, encoding='passthrough')
            msg_mascara = self.bridge.cv2_to_imgmsg(mascara_binaria, encoding='passthrough')

            self.pub_visu.publish(msg_visu)
            self.pub_mascara.publish(msg_mascara)

            # =========================================================
            # PASO 5: VISUALIZACIÓN DUAL
            # =========================================================
            mascara_3ch = cv2.cvtColor(mascara_binaria, cv2.COLOR_GRAY2BGR)
            cv2.putText(
                mascara_3ch, 
                'MASCARA NAVEGABLE (Blanco=Si, Negro=No)', 
                (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.6, 
                (255, 255, 255), 
                2
            )

            vista_combinada = np.hstack((final_frame, mascara_3ch))
            cv2.imshow("Supergiz - Percepcion Color vs Mascara Binaria", vista_combinada)
            cv2.waitKey(1)

        except Exception as e:
            self.get_logger().error(f'Error en callback de percepcion: {e}')

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
