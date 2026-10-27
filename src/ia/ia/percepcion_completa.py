"""
VERSION 1
PERECEPCION COMPLETA USANDO YOLO Y CITYSCAPES
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

