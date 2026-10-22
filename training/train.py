"""Treina o YOLOv8 para as classes do simulador de carro autônomo.

Uso (a partir da raiz do projeto):
    uv run python training/build_dataset.py   # gera training/dataset
    uv run python training/train.py            # treina e copia o modelo para models/

Classes: red_light, green_light, pedestrian, crosswalk, dog, cat, traffic_cone
Os nomes são reconhecidos por category_for() em server.py.
"""
import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
CLASSES = ['red_light', 'green_light', 'pedestrian', 'crosswalk', 'dog', 'cat', 'traffic_cone']

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=60)
parser.add_argument('--imgsz', type=int, default=960)
parser.add_argument('--batch', type=int, default=4)
parser.add_argument('--device', default=None, help="'cpu', '0'... (padrão: GPU se houver, senão CPU)")
args = parser.parse_args()

dataset = HERE / 'dataset'
data_yaml = HERE / 'data.yaml'
data_yaml.write_text(
    f"path: {dataset.as_posix()}\ntrain: images/train\nval: images/val\nnames:\n"
    + ''.join(f'  {i}: {name}\n' for i, name in enumerate(CLASSES))
)

model = YOLO(str(ROOT / 'models' / 'yolov8n.pt'))  # parte dos pesos COCO (fine-tuning)
model.train(
    data=str(data_yaml),
    epochs=args.epochs,
    imgsz=args.imgsz,
    batch=args.batch,
    device=args.device,
    workers=0,
    project=str(HERE / 'runs'),
    name='car_sim',
    exist_ok=True,
    patience=0,
    optimizer='AdamW',
    lr0=0.002,
    cos_lr=True,
    warmup_epochs=1,
    # Augmentations: SEM variação de matiz (hsv_h=0), senão vermelho vira verde.
    hsv_h=0.0, hsv_s=0.4, hsv_v=0.35,
    degrees=0.0, translate=0.1, scale=0.35, fliplr=0.5,
    mosaic=0.6, close_mosaic=10, mixup=0.0,
    plots=True,
    verbose=True,
)

best = HERE / 'runs' / 'car_sim' / 'weights' / 'best.pt'
target = ROOT / 'models' / 'autonomous-car-yolov8.pt'
shutil.copy(best, target)
print(f'Modelo treinado copiado para {target}')
