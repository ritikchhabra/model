import json
from pathlib import Path

det_json_path = "/home/piet/Desktop/yolop/dataset/bdd100k/labels/train_det.json"

with open(det_json_path, 'r') as f:
    data = json.load(f)

for i, sample in enumerate(data):
    if i > 10: break
    print(f"Sample {i}: {sample['name']}")
    for j, label in enumerate(sample.get('labels', [])):
        print(f"  Label {j}: {label.get('category')} - Keys: {list(label.keys())}")
