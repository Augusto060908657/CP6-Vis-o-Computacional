"""Anotações manuais (bounding boxes) das 5 cenas do simulador.

As caixas foram marcadas em coordenadas de uma versão 1456x819 da imagem
(x1, y1, x2, y2) e são normalizadas para o formato YOLO ao gerar os labels.
"""
from pathlib import Path

CLASSES = ['red_light', 'green_light', 'pedestrian', 'crosswalk', 'dog', 'cat', 'traffic_cone']
REF_W, REF_H = 1456, 819

ANNOTATIONS = {
    'clear-green': [
        ('green_light', 826, 127, 855, 192),
        ('green_light', 1140, 246, 1165, 305),
        ('green_light', 1171, 250, 1197, 299),
        ('crosswalk', 270, 443, 1110, 482),
    ],
    'red-pedestrian': [
        ('red_light', 700, 27, 739, 110),
        ('red_light', 579, 245, 593, 273),
        ('red_light', 664, 247, 676, 273),
        ('red_light', 1272, 197, 1323, 232),
        ('pedestrian', 651, 276, 784, 490),
        ('pedestrian', 1195, 313, 1227, 397),
        ('pedestrian', 1228, 310, 1252, 382),
        ('pedestrian', 233, 340, 253, 393),
        ('pedestrian', 68, 318, 95, 368),
        ('crosswalk', 0, 474, 1456, 553),
    ],
    'dog-crossing': [
        ('dog', 460, 301, 647, 402),
        ('crosswalk', 405, 322, 1005, 355),
        ('pedestrian', 1207, 176, 1274, 350),
    ],
    'cat-cone': [
        ('green_light', 815, 14, 837, 61),
        ('green_light', 717, 116, 732, 144),
        ('cat', 602, 364, 682, 448),
        ('traffic_cone', 764, 376, 919, 450),
        ('crosswalk', 540, 303, 1000, 332),
    ],
    # Cena de chuva: semáforos vermelhos reais; a esfera verde na vitrine é o
    # "decoy" e NÃO deve ser rotulada como semáforo verde.
    'rainy-decoy': [
        ('red_light', 1000, 50, 1046, 163),
        ('red_light', 608, 193, 628, 235),
        ('red_light', 755, 240, 775, 277),
        ('red_light', 956, 194, 984, 236),
        ('pedestrian', 898, 283, 945, 413),
        ('pedestrian', 829, 305, 857, 375),
        ('pedestrian', 1023, 302, 1049, 360),
        ('crosswalk', 130, 485, 1456, 560),
    ],
}


def yolo_lines(scene_id):
    lines = []
    for name, x1, y1, x2, y2 in ANNOTATIONS[scene_id]:
        cx, cy = (x1 + x2) / 2 / REF_W, (y1 + y2) / 2 / REF_H
        w, h = (x2 - x1) / REF_W, (y2 - y1) / REF_H
        lines.append(f'{CLASSES.index(name)} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}')
    return lines


if __name__ == '__main__':
    from PIL import Image, ImageDraw
    root = Path(__file__).resolve().parent.parent
    out = Path('/tmp/check'); out.mkdir(exist_ok=True)
    colors = {'red_light': 'red', 'green_light': 'lime', 'pedestrian': 'yellow', 'crosswalk': 'cyan',
              'dog': 'orange', 'cat': 'magenta', 'traffic_cone': 'white'}
    for sid, boxes in ANNOTATIONS.items():
        img = Image.open(root / 'public' / 'scenes' / f'{sid}.jpg').convert('RGB').resize((REF_W, REF_H))
        d = ImageDraw.Draw(img)
        for name, x1, y1, x2, y2 in boxes:
            d.rectangle((x1, y1, x2, y2), outline=colors[name], width=3)
            d.text((x1 + 2, max(0, y1 - 12)), name, fill=colors[name])
        img.save(out / f'{sid}.png')
    print('ok')
