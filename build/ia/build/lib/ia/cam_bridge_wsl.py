import cv2
import numpy as np
import zmq
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

class CameraBridgeNode(Node):
    def __init__(self):
        super().__init__('camera_bridge_node')
        # Publica directamente en /imagen_camara que es donde escucha tu percepcion_completa.py
        self.publisher_ = self.create_publisher(Image, '/imagen_camara', 10)
        self.bridge = CvBridge()

        # Conexión ZMQ al localhost de Windows
        context = zmq.Context()
        self.socket = context.socket(zmq.SUB)
        self.socket.connect("tcp://127.0.0.1:5555")
        self.socket.setsockopt_string(zmq.SUBSCRIBE, "")

        self.get_logger().info('>>> Puente ZMQ activo: Recibiendo vídeo de Windows y publicando en /imagen_camara <<<')
        
        # Temporizador de lectura continua a alta frecuencia (1ms)
        self.timer = self.create_timer(0.001, self.read_frame)

    def read_frame(self):
        try:
            # Recibir búfer de imagen sin bloquear la ejecución
            jpg_buffer = self.socket.recv(flags=zmq.NOBLOCK)
            np_arr = np.frombuffer(jpg_buffer, dtype=np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            if frame is not None:
                msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
                self.publisher_.publish(msg)
        except zmq.Again:
            pass

def main(args=None):
    rclpy.init(args=args)
    node = CameraBridgeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()