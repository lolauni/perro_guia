"""Fine-tuning de SegFormer-b0 para el suelo de Supergiz (CPU o GPU).

Parte del modelo de aceras (segments/sidewalk-semantic) y reaprovecha su cabeza: cada clase
nuestra arranca con los pesos de su clase equivalente, así converge mucho antes.

Datos: carpetas con img/*.jpg + mask/*.png (valores = índices de clases.py, 255 = ignorar).
  --train  carpeta[:peso] ...   (el peso reparte cuántas muestras salen de cada dataset)
  --val    carpeta ...

Aumentos pensados para la cámara del robot (15 cm del suelo):
  - escalado anisótropo (se aplasta la imagen en vertical): imita ver el suelo muy de lado
  - recortes 448x256, volteo horizontal, brillo/contraste/saturación, desenfoque

Uso:
  python entrenar_segformer.py --train datasets/barcelona/train:1 datasets/bags_auto/train:1 \
      --val datasets/barcelona/val datasets/bags_auto/val --salida modelos/segformer_supergiz --horas 3
"""
import argparse
import glob
import os
import sys
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from transformers import SegformerForSemanticSegmentation

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from clases import CLASES, IGNORAR  # noqa: E402

BASE = 'tobiasc/segformer-b0-finetuned-segments-sidewalk'
# clase nuestra -> clase del modelo de aceras cuya fila del clasificador se copia
INIT_CABEZA = {'otro': 18, 'acera': 2, 'calzada': 1, 'paso_cebra': 3, 'bordillo': 7,
               'escalera': 24, 'agujero': 31}
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
W_IN, H_IN = 448, 256


class Carpeta(Dataset):
    def __init__(self, raiz, aumentar):
        self.imgs = sorted(glob.glob(os.path.join(raiz, 'img', '*.jpg')))
        self.raiz, self.aumentar = raiz, aumentar
        if not self.imgs:
            raise FileNotFoundError(f'No hay imágenes en {raiz}/img')

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, i):
        f = self.imgs[i]
        img = cv2.imread(f)
        m = cv2.imread(os.path.join(self.raiz, 'mask', os.path.basename(f)[:-4] + '.png'),
                       cv2.IMREAD_GRAYSCALE)
        img, m = (aumentar if self.aumentar else ajustar)(img, m)
        x = (cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0 - MEAN) / STD
        return torch.from_numpy(x.transpose(2, 0, 1)), torch.from_numpy(m.astype(np.int64))


def ajustar(img, m):
    """Validación: escala para cubrir 448x256 y recorte centrado-abajo (el suelo está abajo)."""
    H, W = img.shape[:2]
    e = max(W_IN / W, H_IN / H)
    img = cv2.resize(img, (max(W_IN, round(W * e)), max(H_IN, round(H * e))), interpolation=cv2.INTER_AREA)
    m = cv2.resize(m, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
    x0 = (img.shape[1] - W_IN) // 2
    y0 = img.shape[0] - H_IN
    return img[y0:y0 + H_IN, x0:x0 + W_IN], m[y0:y0 + H_IN, x0:x0 + W_IN]


def aumentar(img, m):
    H, W = img.shape[:2]
    e = max(W_IN / W, H_IN / H) * np.random.uniform(1.0, 1.6)
    ey = e * np.random.uniform(0.55, 1.0) if np.random.rand() < 0.6 else e   # aplastar en vertical
    nw, nh = max(W_IN, round(W * e)), max(H_IN, round(H * ey))
    img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    m = cv2.resize(m, (nw, nh), interpolation=cv2.INTER_NEAREST)
    x0 = np.random.randint(0, nw - W_IN + 1)
    y0 = np.random.randint(0, nh - H_IN + 1)
    img, m = img[y0:y0 + H_IN, x0:x0 + W_IN], m[y0:y0 + H_IN, x0:x0 + W_IN]
    if np.random.rand() < 0.5:
        img, m = img[:, ::-1], m[:, ::-1]
    img = img.astype(np.float32)
    img = img * np.random.uniform(0.7, 1.3) + np.random.uniform(-30, 30)       # contraste / brillo
    gris = img.mean(axis=2, keepdims=True)
    img = gris + (img - gris) * np.random.uniform(0.6, 1.3)                     # saturación
    img = np.clip(img, 0, 255).astype(np.uint8)
    if np.random.rand() < 0.2:
        img = cv2.GaussianBlur(img, (3, 3), 0)
    return np.ascontiguousarray(img), np.ascontiguousarray(m)


def crear_modelo():
    base = SegformerForSemanticSegmentation.from_pretrained(BASE)
    modelo = SegformerForSemanticSegmentation.from_pretrained(
        BASE, num_labels=len(CLASES),
        id2label=dict(enumerate(CLASES)), label2id={c: i for i, c in enumerate(CLASES)},
        ignore_mismatched_sizes=True)
    with torch.no_grad():
        cb, cm = base.decode_head.classifier, modelo.decode_head.classifier
        for i, c in enumerate(CLASES):
            cm.weight[i] = cb.weight[INIT_CABEZA[c]]
            cm.bias[i] = cb.bias[INIT_CABEZA[c]]
    return modelo


def perdida(logits, y, pesos):
    logits = F.interpolate(logits, size=y.shape[-2:], mode='bilinear', align_corners=False)
    return F.cross_entropy(logits, y, weight=pesos, ignore_index=IGNORAR)


@torch.inference_mode()
def evaluar(modelo, carga):
    modelo.eval()
    n = len(CLASES)
    conf = torch.zeros(n, n, dtype=torch.int64)
    for x, y in carga:
        lg = F.interpolate(modelo(pixel_values=x).logits, size=y.shape[-2:], mode='bilinear')
        p = lg.argmax(1)
        ok = y != IGNORAR
        conf += torch.bincount(y[ok] * n + p[ok], minlength=n * n).reshape(n, n)
    modelo.train()
    inter = conf.diag().float()
    union = conf.sum(0).float() + conf.sum(1).float() - inter
    presentes = conf.sum(1) > 0
    iou = torch.where(union > 0, inter / union.clamp(min=1), torch.zeros_like(inter))
    return float(iou[presentes].mean()), iou, conf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train', nargs='+', required=True)
    ap.add_argument('--val', nargs='+', default=[])
    ap.add_argument('--salida', required=True)
    ap.add_argument('--horas', type=float, default=3.0)
    ap.add_argument('--lote', type=int, default=8)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--muestras_epoca', type=int, default=2000)
    ap.add_argument('--hilos', type=int, default=6)
    ap.add_argument('--reanudar', action='store_true',
                    help='continuar desde el modelo ya guardado en --salida (p. ej. tras apagar el PC)')
    a = ap.parse_args()
    torch.set_num_threads(a.hilos)

    dsets, pesos_muestra = [], []
    for spec in a.train:
        ruta, _, peso = spec.partition(':')
        d = Carpeta(ruta, aumentar=True)
        dsets.append(d)
        pesos_muestra += [float(peso or 1) / len(d)] * len(d)
        print(f'train {ruta}: {len(d)} imágenes (peso {peso or 1})')
    train = torch.utils.data.ConcatDataset(dsets)
    sampler = WeightedRandomSampler(pesos_muestra, a.muestras_epoca, replacement=True)
    carga = DataLoader(train, batch_size=a.lote, sampler=sampler, num_workers=2, drop_last=True)
    vals = [(r, DataLoader(Carpeta(r, aumentar=False), batch_size=8, num_workers=2)) for r in a.val]

    # Pesos por clase: raíz de la frecuencia inversa (las clases finas no se pierden)
    cuenta = np.ones(len(CLASES))
    for d in dsets:
        for f in d.imgs[::max(1, len(d.imgs) // 200)]:
            m = cv2.imread(os.path.join(d.raiz, 'mask', os.path.basename(f)[:-4] + '.png'), 0)
            cuenta += np.bincount(m[m != IGNORAR].ravel(), minlength=256)[:len(CLASES)]
    frec = cuenta / cuenta.sum()
    pesos = torch.tensor(np.clip(np.sqrt(np.median(frec) / frec), 0.3, 5.0), dtype=torch.float32)
    print('pesos de clase:', {c: round(float(p), 2) for c, p in zip(CLASES, pesos)})

    reanudar = a.reanudar and os.path.exists(os.path.join(a.salida, 'config.json'))
    if reanudar:
        print(f'Reanudando desde {a.salida}')
        modelo = SegformerForSemanticSegmentation.from_pretrained(a.salida)
    else:
        modelo = crear_modelo()
    modelo.train()
    opt = torch.optim.AdamW([
        {'params': modelo.segformer.parameters(), 'lr': a.lr},
        {'params': modelo.decode_head.parameters(), 'lr': a.lr * 10},
    ], weight_decay=0.01)

    os.makedirs(a.salida, exist_ok=True)
    # monotonic no avanza con el PC suspendido: las horas son de entrenamiento real
    t_fin = time.monotonic() + a.horas * 3600
    mejor, epoca = -1.0, 0
    if reanudar and vals:                       # no machacar el guardado con algo peor
        mejor = float(np.mean([evaluar(modelo, cv)[0] for _, cv in vals]))
        print(f'nota de partida {mejor:.3f}', flush=True)
    while time.monotonic() < t_fin:
        epoca += 1
        t0, suma = time.time(), 0.0
        for k, (x, y) in enumerate(carga):
            loss = perdida(modelo(pixel_values=x).logits, y, pesos)
            opt.zero_grad()
            loss.backward()
            opt.step()
            suma += float(loss)
            if k % 25 == 0:
                print(f'  época {epoca} lote {k}/{len(carga)} pérdida {float(loss):.3f}', flush=True)
            if time.monotonic() > t_fin:
                break
        print(f'época {epoca}: pérdida media {suma / (k + 1):.3f} ({(time.time() - t0) / 60:.1f} min)')

        miou_total = []
        for ruta, cv in vals:
            miou, iou, _ = evaluar(modelo, cv)
            miou_total.append(miou)
            print(f'  val {os.path.basename(os.path.dirname(ruta.rstrip("/")))}/'
                  f'{os.path.basename(ruta.rstrip("/"))}: mIoU {miou:.3f} | '
                  + ' '.join(f'{c} {float(v):.2f}' for c, v in zip(CLASES, iou)), flush=True)
        nota = float(np.mean(miou_total)) if miou_total else -epoca
        if nota > mejor:
            mejor = nota
            modelo.save_pretrained(a.salida)
            print(f'  guardado en {a.salida} (nota {nota:.3f})', flush=True)
    print('fin. mejor nota', mejor)


if __name__ == '__main__':
    main()
