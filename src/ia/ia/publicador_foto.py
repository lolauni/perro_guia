"""
PUBLICADOR FOTO, Función: Publica una foto en un tópico de ROS 2 para pruebas de percepción.
Este nodo se utiliza para enviar una imagen local a un tópico de ROS 2, simulando la entrada de una cámara en tiempo real. 

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import sys
import time

class PublicadorFoto(Node):
    def __init__(self, ruta_imagen):
        super().__init__('publicador_foto')
        self.publisher_ = self.create_publisher(Image, '/imagen_camara', 10)
        self.bridge = CvBridge()
        
        img = cv2.imread(ruta_imagen)
        if img is None:
            self.get_logger().error(f'No se pudo encontrar la imagen: {ruta_imagen}')
            return
        
        msg = self.bridge.cv2_to_imgmsg(img, encoding='passthrough')
        
        # Pausa breve para conectar el tópico de ROS 2 antes de publicar
        time.sleep(0.5)
        
        # Publicar 1 sola vez
        self.publisher_.publish(msg)
        self.get_logger().info(f'Imagen "{ruta_imagen}" enviada correctamente.')

def main(args=None):
    rclpy.init(args=args)
    
    ruta = 'bus.jpg'
    if len(sys.argv) > 1:
        ruta = sys.argv[1]
        
    nodo = PublicadorFoto(ruta)
    rclpy.spin_once(nodo, timeout_sec=0.5)
    nodo.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
"""

#Para simular una cámara de vídeo real que envíe un stream continuo de fotogramas,  se publican las imagenes en bucle con un timer.

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

class PublicadorFoto(Node):
    def __init__(self):
        super().__init__('publicador_foto')
        self.publisher_ = self.create_publisher(Image, '/imagen_camara', 10)
        self.bridge = CvBridge()
        
        # Cargar la imagen local de prueba
        ruta_imagen = '/home/lola/Documentos/ROS2/supergiz_ws/calle4.jpg'
        self.cv_image = cv2.imread(ruta_imagen)
        
        if self.cv_image is None:
            self.get_logger().error(f'No se pudo encontrar la imagen en: {ruta_imagen}')
        else:
            # Crear un temporizador para publicar la imagen a 10 Hz (cada 0.1 segundos)
            self.timer = self.create_timer(0.1, self.timer_callback)
            self.get_logger().info('>>> Simulador de cámara activo: transmitiendo vídeo en bucle a 10 Hz <<<')

    def timer_callback(self):
        if self.cv_image is not None:
            msg = self.bridge.cv2_to_imgmsg(self.cv_image, encoding='passthrough')
            self.publisher_.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    nodo = PublicadorFoto()
    try:
        rclpy.spin(nodo)
    except KeyboardInterrupt:
        pass
    nodo.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()