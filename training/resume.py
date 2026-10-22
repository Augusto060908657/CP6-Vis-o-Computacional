import shutil
from pathlib import Path
from ultralytics import YOLO
HERE = Path(__file__).resolve().parent
YOLO(str(HERE/'runs'/'car_sim'/'weights'/'last.pt')).train(resume=True)
shutil.copy(HERE/'runs'/'car_sim'/'weights'/'best.pt', HERE.parent/'models'/'autonomous-car-yolov8.pt')
print('COPIED')
