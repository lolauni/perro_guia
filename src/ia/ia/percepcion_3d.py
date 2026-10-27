import time

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import Float32, Bool
from cv_bridge import CvBridge

WIN = 'Supergiz - Percepcion 3D'


class Percepcion3D(Node):
    """
    Geometría a partir de la profundidad alineada de la D435:
      1) ajusta el plano del suelo (RANSAC) en cada frame,
      2) clasifica cada píxel como suelo / obstáculo / desnivel según su altura sobre ese plano,
      3) si existe la máscara semántica (/percepcion/mascara_navegable) la fusiona y publica
         /percepcion/mascara_final = semántica AND NOT (obstáculo OR desnivel).

    Publica además:
      /percepcion/distancia_obstaculo  (Float32): distancia al obstáculo más cercano en el
                                                  pasillo central; -1.0 si no hay ninguno
      /percepcion/desnivel             (Bool):    hay un escalón/bordillo/hueco delante
    """

    def __init__(self):
        super().__init__('percepcion_3d')
        self.bridge = CvBridge()

        # Parámetros (se cambian con --ros-args -p nombre:=valor)
        self.declare_parameter('topico_depth', '/camera/camera/aligned_depth_to_color/image_raw')
        self.declare_parameter('topico_info', '/camera/camera/aligned_depth_to_color/camera_info')
        self.declare_parameter('topico_color', '/camera/camera/color/image_raw')
        self.declare_parameter('topico_semantica', '/percepcion/mascara_navegable')
        self.declare_parameter('stride', 4)            # 640x480 -> 160x120
        self.declare_parameter('dist_min', 0.25)       # m
        self.declare_parameter('dist_max', 4.0)        # m
        self.declare_parameter('ransac_iters', 64)
        self.declare_parameter('ransac_tol', 0.05)     # m
        self.declare_parameter('min_up', 0.5)          # cos del ángulo máx. del plano vs vertical
        self.declare_parameter('obst_min', 0.08)       # m sobre el suelo para contar como obstáculo
        self.declare_parameter('obst_alto_max', 2.2)   # m (por encima se ignora)
        self.declare_parameter('desnivel_min', 0.10)   # m bajo el suelo para contar como desnivel
        self.declare_parameter('min_pix', 12)          # píxeles mínimos para disparar alerta
        self.declare_parameter('debug_view', True)     # ventana de demo
        self.declare_parameter('publicar_visu', False) # publicar imagen de visualización

        g = self.get_parameter
        self.stride = g('stride').value
        self.dist_min = g('dist_min').value
        self.dist_max = g('dist_max').value
        self.ransac_iters = g('ransac_iters').value
        self.ransac_tol = g('ransac_tol').value
        self.min_up = g('min_up').value
        self.obst_min = g('obst_min').value
        self.obst_alto_max = g('obst_alto_max').value
        self.desnivel_min = g('desnivel_min').value
        self.min_pix = g('min_pix').value
        self.debug_view = g('debug_view').value
        self.publicar_visu = g('publicar_visu').value

        self.rng = np.random.default_rng()
        self.kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

        # Estado
        self.fx = self.fy = self.cx = self.cy = None
        self.rx = self.ry = None
        self.shape_full = None
        self.plano = None            # (normal, d) suavizado
        self.fallos = 0
        self.fallos_seguidos = 0
        self.last_color = None
        self.sem = None
        self.sem_t = 0.0

        # ROS 2
        self.create_subscription(CameraInfo, g('topico_info').value,
                                 self.info_cb, qos_profile_sensor_data)
        self.create_subscription(Image, g('topico_depth').value,
                                 self.depth_cb, qos_profile_sensor_data)
        self.create_subscription(Image, g('topico_semantica').value, self.sem_cb, 1)
        if self.debug_view or self.publicar_visu:
            self.create_subscription(Image, g('topico_color').value,
                                     self.color_cb, qos_profile_sensor_data)

        self.pub_final = self.create_publisher(Image, '/percepcion/mascara_final', 1)
        self.pub_visu = self.create_publisher(Image, '/percepcion/imagen_visu_3d', 1)
        self.pub_dist = self.create_publisher(Float32, '/percepcion/distancia_obstaculo', 1)
        self.pub_desn = self.create_publisher(Bool, '/percepcion/desnivel', 1)

        if self.debug_view:
            try:
                cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(WIN, 1280, 480)
            except cv2.error:
                self.get_logger().warn('No se pudo abrir la ventana; sigo sin ella.')
                self.debug_view = False

        self.t_log = time.perf_counter()
        self.t_acc, self.n_acc = 0.0, 0
        self.fps_txt = ''
        self.get_logger().info('>>> Percepción 3D lista (esperando camera_info y profundidad) <<<')

    # ------------------------------------------------------------------
    def info_cb(self, msg):
        if self.fx is None:
            self.fx, self.fy = msg.k[0], msg.k[4]
            self.cx, self.cy = msg.k[2], msg.k[5]
            self.get_logger().info(
                f'Intrínsecos: fx={self.fx:.1f} fy={self.fy:.1f} cx={self.cx:.1f} cy={self.cy:.1f}')

    def color_cb(self, msg):
        self.last_color = msg          # solo se guarda; se convierte si hace falta dibujar

    def sem_cb(self, msg):
        try:
            self.sem = self.bridge.imgmsg_to_cv2(msg, 'mono8')
            self.sem_t = time.monotonic()
        except Exception as e:
            self.get_logger().warn(f'Máscara semántica no válida: {e}')

    # ------------------------------------------------------------------
    def ajustar_suelo(self, X, Y, Z, valid, hs):
        """RANSAC vectorizado sobre la mitad inferior de la imagen. Devuelve (normal, d)."""
        cand = valid.copy()
        cand[:int(hs * 0.4), :] = False
        cand &= (Z < 3.5)
        P = np.stack([X[cand], Y[cand], Z[cand]], axis=1)
        N = len(P)

        nuevo = None
        if N >= 150:
            if N > 3000:
                P = P[self.rng.choice(N, 3000, replace=False)]
                N = 3000
            K = self.ransac_iters
            idx = self.rng.integers(0, N, size=(K, 3))
            p0, p1, p2 = P[idx[:, 0]], P[idx[:, 1]], P[idx[:, 2]]
            nrm = np.cross(p1 - p0, p2 - p0)
            norma = np.linalg.norm(nrm, axis=1, keepdims=True)
            nrm = nrm / np.maximum(norma, 1e-9)
            nrm[nrm[:, 1] > 0] *= -1        # normal hacia "arriba" (-Y en el sistema óptico)
            ok = (norma[:, 0] > 1e-6) & (-nrm[:, 1] > self.min_up)
            dd = -np.einsum('kj,kj->k', nrm, p0)
            dist = np.abs(P @ nrm.T + dd)    # (N, K)
            inl = (dist < self.ransac_tol).sum(axis=0)
            inl[~ok] = 0
            mejor = int(inl.argmax())

            if inl[mejor] >= 0.2 * N:
                Q = P[dist[:, mejor] < self.ransac_tol].astype(np.float64)
                c = Q.mean(axis=0)
                _, _, vt = np.linalg.svd(Q - c, full_matrices=False)
                n = vt[-1]
                if n[1] > 0:
                    n = -n
                if -n[1] > self.min_up:
                    nuevo = (n, -float(n @ c))

        if nuevo is None:
            self.fallos += 1
            self.fallos_seguidos += 1
            if self.fallos_seguidos > 10:   # plano demasiado viejo: no fiarse
                self.plano = None
            return self.plano

        self.fallos_seguidos = 0
        if self.plano is None:
            self.plano = nuevo
        else:
            a = 0.3                         # suavizado temporal
            n = a * nuevo[0] + (1 - a) * self.plano[0]
            n /= np.linalg.norm(n)
            self.plano = (n, a * nuevo[1] + (1 - a) * self.plano[1])
        return self.plano

    # ------------------------------------------------------------------
    def depth_cb(self, msg):
        if self.fx is None:
            return
        try:
            t0 = time.perf_counter()
            depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')   # uint16, mm
            H, W = depth.shape[:2]
            s = self.stride

            if self.rx is None or self.shape_full != (H, W):
                self.shape_full = (H, W)
                u = np.arange(0, W, s, dtype=np.float32)
                v = np.arange(0, H, s, dtype=np.float32)
                self.rx = ((u - self.cx) / self.fx)[None, :]
                self.ry = ((v - self.cy) / self.fy)[:, None]

            # Submuestreo directo (sin promediar: evita mezclar bordes de objetos)
            z = depth[::s, ::s].astype(np.float32) * 0.001
            hs, ws = z.shape
            valid = (z > self.dist_min) & (z < self.dist_max)
            X = self.rx * z
            Y = self.ry * z

            plano = self.ajustar_suelo(X, Y, z, valid, hs)
            if plano is None:
                return
            nx, ny, nz = (float(v) for v in plano[0])
            h = nx * X + ny * Y + nz * z + float(plano[1])    # altura sobre el suelo (m)

            # Umbrales que crecen con la distancia (el ruido de la D435 también)
            suelo = valid & (np.abs(h) < 0.03 + 0.02 * z)
            obst = valid & (h > self.obst_min + 0.03 * z) & (h < self.obst_alto_max)
            desn = valid & (h < -(self.desnivel_min + 0.03 * z))

            suelo_u8 = suelo.astype(np.uint8) * 255
            obst_u8 = cv2.morphologyEx(obst.astype(np.uint8) * 255, cv2.MORPH_OPEN, self.kernel)
            desn_u8 = cv2.morphologyEx(desn.astype(np.uint8) * 255, cv2.MORPH_OPEN, self.kernel)

            # Pasillo central
            c0, c1 = ws // 3, 2 * ws // 3
            corr = obst_u8[:, c0:c1] > 0
            dist = -1.0
            if int(corr.sum()) >= self.min_pix:
                dist = float(np.percentile(z[:, c0:c1][corr], 5))
            hay_desnivel = int((desn_u8[hs // 2:, c0:c1] > 0).sum()) >= self.min_pix

            self.pub_dist.publish(Float32(data=dist))
            self.pub_desn.publish(Bool(data=hay_desnivel))

            # Métricas (sin contar el dibujo)
            self.t_acc += time.perf_counter() - t0
            self.n_acc += 1
            ahora = time.perf_counter()
            if ahora - self.t_log >= 2.0:
                fps = self.n_acc / (ahora - self.t_log)
                ms = 1000.0 * self.t_acc / self.n_acc
                self.fps_txt = f'{fps:.1f} FPS | 3D {ms:.1f} ms | fallos plano {self.fallos}'
                self.get_logger().info(self.fps_txt)
                self.t_log, self.t_acc, self.n_acc = ahora, 0.0, 0

            # Máscara final (fusión con la semántica si está fresca)
            quiere_vis = self.debug_view or self.publicar_visu
            if self.pub_final.get_subscription_count() > 0 or quiere_vis:
                bloq_small = cv2.bitwise_or(obst_u8, desn_u8)
                if self.sem is not None and time.monotonic() - self.sem_t < 1.0:
                    final = self.sem.copy()
                    bloq = cv2.resize(bloq_small, (final.shape[1], final.shape[0]),
                                      interpolation=cv2.INTER_NEAREST)
                else:
                    final = cv2.resize(suelo_u8, (W, H), interpolation=cv2.INTER_NEAREST)
                    bloq = cv2.resize(bloq_small, (W, H), interpolation=cv2.INTER_NEAREST)
                final[bloq > 0] = 0

                if self.pub_final.get_subscription_count() > 0:
                    m = self.bridge.cv2_to_imgmsg(final, encoding='mono8')
                    m.header = msg.header
                    self.pub_final.publish(m)

                if quiere_vis:
                    self.dibujar(msg, final, suelo_u8, obst_u8, desn_u8, dist, hay_desnivel, H, W)

        except Exception as e:
            self.get_logger().error(f'Error en percepción 3D: {e}')

    # ------------------------------------------------------------------
    def dibujar(self, msg, final, suelo_u8, obst_u8, desn_u8, dist, hay_desnivel, H, W):
        if self.last_color is not None:
            vis = self.bridge.imgmsg_to_cv2(self.last_color, 'bgr8')
            if vis.shape[:2] != (H, W):
                vis = cv2.resize(vis, (W, H))
            vis = vis.copy()
        else:
            vis = np.zeros((H, W, 3), dtype=np.uint8)

        def pintar(m_small, color):
            m = cv2.resize(m_small, (W, H), interpolation=cv2.INTER_NEAREST) > 0
            vis[m] = (0.5 * vis[m] + 0.5 * np.array(color)).astype(np.uint8)

        pintar(suelo_u8, (0, 255, 0))      # verde: suelo
        pintar(obst_u8, (0, 0, 255))       # rojo: obstáculo
        pintar(desn_u8, (255, 0, 255))     # magenta: desnivel

        txt = f'Obstaculo: {dist:.2f} m' if dist > 0 else 'Camino libre'
        if hay_desnivel:
            txt += ' | DESNIVEL'
        cv2.putText(vis, self.fps_txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(vis, txt, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        if self.publicar_visu and self.pub_visu.get_subscription_count() > 0:
            mv = self.bridge.cv2_to_imgmsg(vis, encoding='bgr8')
            mv.header = msg.header
            self.pub_visu.publish(mv)

        if self.debug_view:
            if final.shape[:2] != (H, W):
                final = cv2.resize(final, (W, H), interpolation=cv2.INTER_NEAREST)
            fin3 = cv2.cvtColor(final, cv2.COLOR_GRAY2BGR)
            cv2.putText(fin3, 'MASCARA FINAL', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.imshow(WIN, np.hstack((vis, fin3)))
            cv2.waitKey(1)


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