import os
import json
import cv2
import numpy as np
from pathlib import Path
import argparse

def verify_detection(img_dir, det_json_path):
    print(f"Verifying detection: {det_json_path}")
    valid = 0
    invalid = 0
    missing = 0
    
    if not os.path.exists(det_json_path):
        print(f"Error: {det_json_path} not found")
        return 0, 0, 0

    with open(det_json_path, 'r') as f:
        data = json.load(f)
        
    det_categories = set()
    for sample in data[:100]:
        for label in sample.get('labels', []):
            if 'box2d' in label:
                det_categories.add(label['category'])

    for item in data:
        img_name = item['name']
        img_path = Path(img_dir) / img_name
        
        if not img_path.exists():
            missing += 1
            continue
        
        labels = item.get('labels', [])
        all_boxes_valid = True
        has_det_label = False
        
        for label in labels:
            cat = label.get('category')
            if cat in det_categories:
                has_det_label = True
                box = label.get('box2d')
                if not box:
                    all_boxes_valid = False
                    break
                x1, y1, x2, y2 = box['x1'], box['y1'], box['x2'], box['y2']
                if not (x1 < x2 and y1 < y2 and x1 >= 0 and y1 >= 0):
                    all_boxes_valid = False
                    break
        
        if has_det_label:
            if all_boxes_valid:
                valid += 1
            else:
                invalid += 1
        else:
            valid += 1
            
    return valid, invalid, missing

def verify_lanes(img_dir, lane_json_path):
    print(f"Verifying lanes: {lane_json_path}")
    valid = 0
    invalid = 0
    missing = 0
    
    if not os.path.exists(lane_json_path):
        print(f"Error: {lane_json_path} not found")
        return 0, 0, 0

    with open(lane_json_path, 'r') as f:
        data = json.load(f)
        
    for item in data:
        img_name = item['name']
        img_path = Path(img_dir) / img_name
        if not img_path.exists():
            missing += 1
            continue
        lanes = [l for l in item.get('labels', []) if l['category'] == 'lane']
        if not lanes:
            valid += 1
            continue
        all_lanes_valid = True
        for lane in lanes:
            poly = lane.get('poly2d')
            if not poly:
                all_lanes_valid = False
                break
            for p in poly:
                vertices = p.get('vertices', [])
                if not vertices:
                    all_lanes_valid = False
                    break
                for v in vertices:
                    if len(v) != 2 or not all(isinstance(coord, (int, float)) for coord in v):
                        all_lanes_valid = False
                        break
                if not all_lanes_valid: break
            if not all_lanes_valid: break
        if all_lanes_valid: valid += 1
        else: invalid += 1
    return valid, invalid, missing

def verify_drivable(img_dir, mask_dir):
    print(f"Verifying drivable area: {mask_dir}")
    valid = 0
    invalid = 0
    missing = 0
    
    img_files = list(Path(img_dir).glob("*.jpg"))
    
    mask_lookup = {}
    # mask_dir is already the split dir (e.g. dataset/bdd100k/drivable/train)
    if mask_dir.exists():
        for p in mask_dir.glob("*.png"):
            mask_lookup[p.name] = p

    for img_path in img_files:
        img_stem = img_path.stem
        mask_found = False
        # We don't know if it's train or val from the image filename alone
        # but we can check both common patterns in the current mask_dir
        for pattern in [f"{img_stem}_train_id.png", f"{img_stem}_val_id.png", f"{img_stem}_id.png"]:
            if pattern in mask_lookup:
                mask_path = mask_lookup[pattern]
                mask_found = True
                try:
                    mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
                    img = cv2.imread(str(img_path))
                    if mask is not None and img is not None and mask.shape[:2] == img.shape[:2]:
                        valid += 1
                    else:
                        invalid += 1
                except Exception:
                    invalid += 1
                break
        if not mask_found:
            missing += 1
            
    return valid, invalid, missing

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    args = parser.parse_args()
    dataset_root = Path(args.dataset)
    img_dir = dataset_root / "images"
    det_labels_dir = dataset_root / "labels"
    lane_labels_dir = dataset_root / "lanes"
    drivable_dir = dataset_root / "drivable"
    
    print("DATASET VERIFICATION")
    print("-" * 20)
    img_count = len(list(img_dir.glob("*.jpg")))
    print(f"Total Images Found: {img_count}")
    
    det_valid, det_invalid, det_missing = 0, 0, 0
    for json_file in det_labels_dir.glob("*.json"):
        v, i, m = verify_detection(img_dir, json_file)
        det_valid += v; det_invalid += i; det_missing += m
    print(f"\nDetection:\n  Valid: {det_valid}\n  Invalid: {det_invalid}\n  Missing: {det_missing}")
    
    lane_valid, lane_invalid, lane_missing = 0, 0, 0
    for json_file in lane_labels_dir.glob("*.json"):
        v, i, m = verify_lanes(img_dir, json_file)
        lane_valid += v; lane_invalid += i; lane_missing += m
    print(f"\nLane:\n  Valid: {lane_valid}\n  Invalid: {lane_invalid}\n  Missing: {lane_missing}")
    
    drv_valid, drv_invalid, drv_missing = 0, 0, 0
    for split in ["train", "val"]:
        v, i, m = verify_drivable(img_dir, drivable_dir / split)
        drv_valid += v; drv_invalid += i; drv_missing += m
    print(f"\nDrivable:\n  Valid: {drv_valid}\n  Invalid: {drv_invalid}\n  Missing: {drv_missing}")

if __name__ == "__main__":
    main()
