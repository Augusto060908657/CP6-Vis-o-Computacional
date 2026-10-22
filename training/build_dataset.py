"""Monta o dataset YOLO (training/dataset) a partir das cenas + annotations.py.

Como há apenas 5 cenas, cada uma é expandida em várias variações offline
(recortes aleatórios que mantêm os objetos, ajustes de brilho/contraste,
espelhamento horizontal). Variações de matiz NÃO são aplicadas, pois
trocariam vermelho por verde.
"""
import random
import shutil
from pathlib import Path

from PIL import Image, ImageEnhance

from annotations import ANNOTATIONS, CLASSES, REF_H, REF_W, yolo_lines

ROOT = Path(__file__).resolve().parent.parent
SCENES = ROOT / 'public' / 'scenes'
OUT = Path(__file__).resolve().parent / 'dataset'
VARIANTS = 14          # variações por cena (além da original)
random.seed(7)


def boxes_for(scene_id):
    return [(CLASSES.index(n), x1 / REF_W, y1 / REF_H, x2 / REF_W, y2 / REF_H)
            for n, x1, y1, x2, y2 in ANNOTATIONS[scene_id]]


def write_label(path, boxes):
    lines = []
    for c, x1, y1, x2, y2 in boxes:
        x1, y1, x2, y2 = [min(1, max(0, v)) for v in (x1, y1, x2, y2)]
        if x2 - x1 < 0.004 or y2 - y1 < 0.004:
            continue
        lines.append(f'{c} {(x1 + x2) / 2:.6f} {(y1 + y2) / 2:.6f} {x2 - x1:.6f} {y2 - y1:.6f}')
    path.write_text('\n'.join(lines) + '\n')


def variant(img, boxes):
    w, h = img.size
    # recorte aleatório (escala 0.7-1.0) que mantém a maioria das caixas visíveis
    s = random.uniform(0.7, 1.0)
    cw, ch = int(w * s), int(h * s)
    x0, y0 = random.randint(0, w - cw), random.randint(0, h - ch)
    crop = img.crop((x0, y0, x0 + cw, y0 + ch)).resize((w, h), Image.BILINEAR)
    nb = []
    for c, x1, y1, x2, y2 in boxes:
        X1, Y1, X2, Y2 = x1 * w, y1 * h, x2 * w, y2 * h
        X1, X2 = max(X1, x0), min(X2, x0 + cw)
        Y1, Y2 = max(Y1, y0), min(Y2, y0 + ch)
        if X2 - X1 < 0.6 * (x2 - x1) * w or Y2 - Y1 < 0.6 * (y2 - y1) * h:
            continue  # objeto cortado demais
        nb.append((c, (X1 - x0) / cw, (Y1 - y0) / ch, (X2 - x0) / cw, (Y2 - y0) / ch))
    if random.random() < 0.5:  # espelhamento horizontal
        crop = crop.transpose(Image.FLIP_LEFT_RIGHT)
        nb = [(c, 1 - x2, y1, 1 - x1, y2) for c, x1, y1, x2, y2 in nb]
    crop = ImageEnhance.Brightness(crop).enhance(random.uniform(0.8, 1.2))
    crop = ImageEnhance.Contrast(crop).enhance(random.uniform(0.85, 1.15))
    crop = ImageEnhance.Color(crop).enhance(random.uniform(0.85, 1.15))  # saturação, não matiz
    return crop, nb


def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    for split in ('train', 'val'):
        (OUT / 'images' / split).mkdir(parents=True)
        (OUT / 'labels' / split).mkdir(parents=True)
    for sid in ANNOTATIONS:
        img = Image.open(SCENES / f'{sid}.jpg').convert('RGB')
        boxes = boxes_for(sid)
        # original: vai para train e val (val = as 5 cenas originais do jogo)
        for split in ('train', 'val'):
            img.save(OUT / 'images' / split / f'{sid}.jpg', quality=95)
            write_label(OUT / 'labels' / split / f'{sid}.txt', boxes)
        for i in range(VARIANTS):
            v, nb = variant(img, boxes)
            v.save(OUT / 'images' / 'train' / f'{sid}_v{i:02d}.jpg', quality=92)
            write_label(OUT / 'labels' / 'train' / f'{sid}_v{i:02d}.txt', nb)
    yaml = f"path: {OUT}\ntrain: images/train\nval: images/val\nnames:\n" + ''.join(f'  {i}: {n}\n' for i, n in enumerate(CLASSES))
    (Path(__file__).resolve().parent / 'data.yaml').write_text(yaml)
    print('train:', len(list((OUT / 'images' / 'train').glob('*.jpg'))), 'val:', len(list((OUT / 'images' / 'val').glob('*.jpg'))))


if __name__ == '__main__':
    main()
