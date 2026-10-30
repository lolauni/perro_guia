"""Convierte el export COCO Segmentation de Barcelona Streets (Roboflow) en imágenes + máscaras semánticas.

- Se queda con UNA copia de cada imagen original (Roboflow trae 3 versiones aumentadas de cada una;
  aumentamos nosotros al entrenar).
- Separa train / val por VÍDEO, no por imagen: frames del mismo vídeo son casi iguales y, si caen
  en los dos lados, la validación sale engañosamente buena.

Uso:
  python preparar_barcelona.py <carpeta_o_zip_roboflow> <salida> [--lado 320] [--val 0.15]
"""
import argparse
import hashlib
import json
import os
import re
import sys
import zipfile
from collections import defaultdict

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from clases import BARCELONA, PRIORIDAD  # noqa: E402


def original(nombre):
    """'IMG_6433_mp4-0014_jpg.rf.<hash>.jpg' -> 'IMG_6433_mp4-0014_jpg'"""
    return os.path.basename(nombre).split('.rf.')[0]


def video(orig):
    """Quita el número de frame: 'IMG_6433_mp4-0014_jpg' -> 'IMG_6433_mp4'."""
    return re.sub(r'[-_](f-)?\d+_(jpe?g|png)$', '', orig, flags=re.I)


def es_val(grupo, frac):
    h = int(hashlib.md5(grupo.encode()).hexdigest(), 16)
    return (h % 1000) / 1000.0 < frac


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('entrada')
    ap.add_argument('salida')
    ap.add_argument('--lado', type=int, default=320, help='lado corto al guardar (px)')
    ap.add_argument('--val', type=float, default=0.15)
    a = ap.parse_args()

    raiz = a.entrada
    if raiz.endswith('.zip'):
        destino = os.path.splitext(raiz)[0]
        if not os.path.isdir(destino):
            with zipfile.ZipFile(raiz) as z:
                z.extractall(destino)
        raiz = destino

    jsons = []
    for d, _, fs in os.walk(raiz):
        jsons += [os.path.join(d, f) for f in fs if f.endswith('.coco.json')]
    if not jsons:
        sys.exit(f'No hay *_annotations.coco.json en {raiz}')

    vistos = set()
    por_grupo = defaultdict(int)
    stats = np.zeros(256, np.int64)
    for split in ('train', 'val'):
        for sub in ('img', 'mask'):
            os.makedirs(os.path.join(a.salida, split, sub), exist_ok=True)

    for jf in sorted(jsons):
        coco = json.load(open(jf))
        cats = {c['id']: BARCELONA.get(c['name'].strip().lower()) for c in coco['categories']}
        print(jf, '->', {c['name']: cats[c['id']] for c in coco['categories']})
        anns = defaultdict(list)
        for an in coco['annotations']:
            anns[an['image_id']].append(an)

        for im in coco['images']:
            orig = original(im['file_name'])
            if orig in vistos:
                continue
            vistos.add(orig)
            img = cv2.imread(os.path.join(os.path.dirname(jf), im['file_name']))
            if img is None:
                continue
            H, W = img.shape[:2]
            mask = np.zeros((H, W), np.uint8)               # 0 = otro (sin anotar)
            capas = defaultdict(list)
            for an in anns[im['id']]:
                c = cats.get(an['category_id'])
                if c is None or not isinstance(an.get('segmentation'), list):
                    continue
                for seg in an['segmentation']:
                    if len(seg) >= 6:
                        capas[c].append(np.array(seg, np.float32).reshape(-1, 2).astype(np.int32))
            for c in PRIORIDAD:
                if capas[c]:
                    cv2.fillPoly(mask, capas[c], int(c))

            esc = a.lado / min(H, W)
            if esc < 1.0:
                img = cv2.resize(img, (round(W * esc), round(H * esc)), interpolation=cv2.INTER_AREA)
                mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

            g = video(orig)
            split = 'val' if es_val(g, a.val) else 'train'
            por_grupo[(split, g)] += 1
            cv2.imwrite(os.path.join(a.salida, split, 'img', orig + '.jpg'), img,
                        [cv2.IMWRITE_JPEG_QUALITY, 92])
            cv2.imwrite(os.path.join(a.salida, split, 'mask', orig + '.png'), mask)
            stats += np.bincount(mask.ravel(), minlength=256)

    for split in ('train', 'val'):
        gs = {g: n for (s, g), n in por_grupo.items() if s == split}
        print(f'{split}: {sum(gs.values())} imágenes de {len(gs)} vídeos')
    tot = stats.sum()
    print('Píxeles por clase:', {i: f'{100 * stats[i] / tot:.1f}%' for i in range(7)})


if __name__ == '__main__':
    main()
