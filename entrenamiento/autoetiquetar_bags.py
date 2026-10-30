"""Etiqueta frames de los rosbags con la geometría de la D435 (sin anotar a mano).

Misma idea que percepcion_fusion en modo 'geometria':
  - suelo al nivel del robot y conectado con sus patas  -> acera (1)
  - obstáculos por encima del suelo                      -> otro (0)
  - todo lo demás (desniveles, sin profundidad, lejos)   -> ignorar (255)
Así la red aprende el aspecto del suelo DESDE LA ALTURA REAL DE LA CÁMARA (15 cm), que es
lo que no tiene ningún dataset público. Los desniveles se ignoran porque detrás de un
bordillo puede haber calzada o no, y una etiqueta mala hace más daño que ninguna.

Uso (con el entorno de ROS cargado, p. ej. `pixi run python ...`):
  python autoetiquetar_bags.py <salida> <bag1> [<bag2> ...] [--cada 5]
"""
import argparse
import os
import sys

import cv2
import numpy as np
import rosbag2_py
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import CameraInfo, Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from clases import IGNORAR  # noqa: E402

T_COLOR = '/camera/camera/color/image_raw'
T_DEPTH = '/camera/camera/aligned_depth_to_color/image_raw'
T_INFO = '/camera/camera/aligned_depth_to_color/camera_info'

rng = np.random.default_rng(0)
K3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))


def plano_suelo(X, Y, Z, valid, H):
    """RANSAC sobre el suelo cercano (filas inferiores). Devuelve (n, d) con n hacia arriba."""
    cand = valid.copy()
    cand[:int(H * 0.6)] = False
    cand &= Z < 3.5
    P = np.stack([X[cand], Y[cand], Z[cand]], 1)
    if len(P) < 150:
        return None
    if len(P) > 3000:
        P = P[rng.choice(len(P), 3000, replace=False)]
    idx = rng.integers(0, len(P), (128, 3))
    p0, p1, p2 = P[idx[:, 0]], P[idx[:, 1]], P[idx[:, 2]]
    n = np.cross(p1 - p0, p2 - p0)
    nn = np.linalg.norm(n, axis=1, keepdims=True)
    n = n / np.maximum(nn, 1e-9)
    n[n[:, 1] > 0] *= -1
    d = -np.einsum('kj,kj->k', n, p0)
    ok = (nn[:, 0] > 1e-6) & (-n[:, 1] > 0.5) & (d > 0.05) & (d < 1.0)
    dist = np.abs(P @ n.T + d)
    inl = (dist < 0.03).sum(0)
    inl[~ok] = 0
    b = int(inl.argmax())
    if inl[b] < 0.3 * len(P):
        return None
    Q = P[dist[:, b] < 0.03]
    c = Q.mean(0)
    nv = np.linalg.svd(Q - c, full_matrices=False)[2][-1]
    if nv[1] > 0:
        nv = -nv
    return nv, -float(nv @ c)


def etiquetar(depth_mm, K):
    fx, fy, cx, cy = K
    z = depth_mm.astype(np.float32) * 0.001
    H, W = z.shape
    v, u = np.mgrid[0:H, 0:W].astype(np.float32)
    X, Y = (u - cx) / fx * z, (v - cy) / fy * z
    valid = (z > 0.25) & (z < 4.0)
    pl = plano_suelo(X, Y, z, valid, H)
    if pl is None:
        return None
    n, d = pl
    h = n[0] * X + n[1] * Y + n[2] * z + d

    suelo = (valid & (np.abs(h) < 0.04 + 0.01 * z)).astype(np.uint8)
    suelo = cv2.morphologyEx(suelo, cv2.MORPH_OPEN, K3)
    _, lab = cv2.connectedComponents(suelo, connectivity=4)
    sem = lab[int(H * 0.85):, W // 4:3 * W // 4]
    suelo = np.isin(lab, np.unique(sem[sem > 0]))
    obst = valid & (h > 0.08 + 0.03 * z)
    obst = cv2.morphologyEx(obst.astype(np.uint8), cv2.MORPH_OPEN, K3) > 0

    m = np.full((H, W), IGNORAR, np.uint8)
    m[obst] = 0
    # Erosión: los bordes de cada zona son los más dudosos (profundidad mezclada)
    m[cv2.erode(suelo.astype(np.uint8), K3, iterations=2) > 0] = 1
    if suelo.mean() < 0.05:                         # casi sin suelo: frame poco fiable
        return None
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('salida')
    ap.add_argument('bags', nargs='+')
    ap.add_argument('--cada', type=int, default=5, help='un frame de cada N')
    a = ap.parse_args()
    os.makedirs(os.path.join(a.salida, 'img'), exist_ok=True)
    os.makedirs(os.path.join(a.salida, 'mask'), exist_ok=True)

    for bag in a.bags:
        nombre = os.path.basename(os.path.normpath(bag))
        r = rosbag2_py.SequentialReader()
        r.open(rosbag2_py.StorageOptions(uri=bag, storage_id='mcap'),
               rosbag2_py.ConverterOptions('', ''))
        K, depth, t_depth, n, guardados = None, None, 0, 0, 0
        while r.has_next():
            t, data, stamp = r.read_next()
            if t == T_INFO and K is None:
                m = deserialize_message(data, CameraInfo)
                K = (m.k[0], m.k[4], m.k[2], m.k[5])
            elif t == T_DEPTH:
                m = deserialize_message(data, Image)
                depth = np.frombuffer(m.data, np.uint16).reshape(m.height, m.width)
                t_depth = stamp
            elif t == T_COLOR:
                n += 1
                if n % a.cada or K is None or depth is None or abs(stamp - t_depth) > 50e6:
                    continue
                m = deserialize_message(data, Image)
                img = np.frombuffer(m.data, np.uint8).reshape(m.height, m.width, -1)
                if m.encoding == 'rgb8':
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                if img.shape[:2] != depth.shape:
                    continue
                etiq = etiquetar(depth, K)
                if etiq is None:
                    continue
                base = f'{nombre}_{n:05d}'
                cv2.imwrite(os.path.join(a.salida, 'img', base + '.jpg'), img,
                            [cv2.IMWRITE_JPEG_QUALITY, 92])
                cv2.imwrite(os.path.join(a.salida, 'mask', base + '.png'), etiq)
                guardados += 1
        print(f'{nombre}: {n} frames de color, {guardados} etiquetados')


if __name__ == '__main__':
    main()
