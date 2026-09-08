import json
from pathlib import Path

det_json_path = "/home/piet/Desktop/yolop/dataset/bdd100k/labels/train_det.json"
img_dir = "/home/piet/Desktop/yolop/dataset/bdd100k/images"

with open(det_json_path, 'r') as f:
    data = json.load(f)

sample = data[0]
print(f"Sample: {sample['name']}")
img_path = Path(img_dir) / sample['name']
print(f"Image exists: {img_path.exists()}")

import cv2
img = cv2.imread(str(img_path))
if img is not None:
    print(f"Image shape: {img.shape}")
else:
    print("Failed to load image")

for label in sample['labels']:
    if label['category'] == 'lane': continue
    box = label['box2d']
    print(f"Box: {box}")
    x1, y1, x2, y2 = box['x1'], box['y1'], box['x2'], box['y2']
    print(f"Valid: {x1 < x2 and y1 < y2 and x1 >= 0 and y1 >= 0}")
