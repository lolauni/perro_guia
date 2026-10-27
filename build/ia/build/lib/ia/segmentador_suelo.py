import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import torch
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation

class SegmentadorSuelo(Node):
    def __init__(self):
        super().__init__('segmentador_suelo')
        
        # Inicializar CvBridge
        self.bridge = CvBridge()
        
        # Cargar el modelo SegFormer B0 pre-entrenado en Cityscapes
        self.get_logger().info('Cargando procesador y modelo SegFormer...')
        modelo_id = "/home/lola/ROS2_workspace/supergiz_ws/segformer-b0-finetuned-cityscapes-512-1024"
        self.processor = SegformerImageProcessor.from_pretrained(modelo_id)
        self.model = SegformerForSemanticSegmentation.from_pretrained(modelo_id)
        self.get_logger().info('Modelo cargado exitosamente.')

        # Suscriptor al tópico de la cámara
        self.subscription = self.create_subscription(
            Image,
            '/imagen_camara',
            self.listener_callback,
            10
        )

        # Mapa de colores para las clases de Cityscapes (Formato BGR para OpenCV)
        # 0: Road (Rojo - Zona prohibida)
        # 1: Sidewalk (Verde - Zona segura/navegable)
        self.color_map = np.zeros((256, 3), dtype=np.uint8)
        self.color_map[0] = [0, 0, 255]   # Carretera
        self.color_map[1] = [0, 255, 0]   # Acera
        # El resto de clases (edificios, coches, cielo) se quedarán en negro [0, 0, 0]

    def listener_callback(self, msg):
        self.get_logger().info('Recibiendo imagen para segmentación semántica...')
        
        try:
            # Convertir mensaje ROS a imagen OpenCV
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            
            # Convertir de BGR a RGB para el modelo
            rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            
            # Preparar imagen para el modelo
            inputs = self.processor(images=rgb_image, return_tensors="pt")
            
            # Inferencia sin cálculo de gradientes
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # Recuperar los logits y redimensionar al tamaño original de la imagen
            logits = outputs.logits
            upsampled_logits = torch.nn.functional.interpolate(
                logits,
                size=cv_image.shape[:2], # Alto y ancho original
                mode="bilinear",
                align_corners=False
            )
            
            # Obtener la clase con mayor probabilidad por cada píxel
            seg_map = upsampled_logits.argmax(dim=1)[0].cpu().numpy()
            
            # Colorear la máscara usando el mapa de colores
            mask = self.color_map[seg_map]
            
            # Fusionar imagen original con la máscara de segmentación
            alpha = 0.5
            overlay = cv2.addWeighted(cv_image, 1 - alpha, mask, alpha, 0)
            
            # Guardar el resultado para verificar
            cv2.imwrite('resultado_suelo.jpg', overlay)
            self.get_logger().info('Imagen guardada como resultado_suelo.jpg')
            
        except Exception as e:
            self.get_logger().error(f'Error en el callback: {e}')

def main(args=None):
    rclpy.init(args=args)
    nodo = SegmentadorSuelo()
    rclpy.spin(nodo)
    nodo.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()