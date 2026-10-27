"""
VERSIÓN 2
PERECEPCION COMPLETA USANDO YOLO Y ADE20K
"""

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
