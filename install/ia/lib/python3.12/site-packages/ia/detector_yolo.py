import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from ultralytics import YOLO
import cv2

class DetectorYolo(Node):
    def __init__(self):
        super().__init__('detector_yolo')
        self.get_logger().info('Cargando modelo YOLOv8 Segmentación...')
        self.bridge = CvBridge()
        
        # Debe incluir '-seg.pt' obligatoriamente
        self.model = YOLO('yolov8n-seg.pt')
        
        self.subscription = self.create_subscription(
            Image,
            '/imagen_camara',
            self.listener_callback,
            10
        )
        self.get_logger().info('Esperando imágenes en /imagen_camara...')

    def listener_callback(self, msg):
        self.get_logger().info('¡Imagen recibida!')
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        
        results = self.model(cv_image, verbose=False)
        
        for result in results:
            # Ocultamos los rectángulos (boxes=False) para mostrar solo las sombras coladas
            img_resultado = result.plot(boxes=False, masks=True)
            cv2.imwrite('resultado.jpg', img_resultado)
            self.get_logger().info('Resultado guardado en resultado.jpg')

def main(args=None):
    rclpy.init(args=args)
    nodo = DetectorYolo()
    rclpy.spin(nodo)
    nodo.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()