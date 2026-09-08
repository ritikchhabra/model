import os
import cv2
import torch
import numpy as np
import random
from pathlib import Path
import argparse
from yolop.datasets.bdd100k_dataset import YOLOPDataset

def preview_dataset(dataset_root, num_samples=5):
    print(f"Starting dataset preview on: {dataset_root}")
    dataset = YOLOPDataset(dataset_root, split='train')
    
    os.makedirs("logs/dataset_preview", exist_ok=True)
    
    indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))
    
    for i, idx in enumerate(indices):
        img, sample = dataset.get_preview_data(idx)
        img_name = sample['name']
        h, w = img.shape[:2]
        
        missing_labels = []

        # 1. Draw Drivable Area (Green overlay, 40% opacity)
        # We need to find the mask file for this image
        stem = Path(img_name).stem
        mask_path = dataset.drv_dir / f"{stem}_id.png"
        if not mask_path.exists():
            mask_path = dataset.drv_dir / f"{stem}_{dataset.split}_id.png"
        
        if mask_path.exists():
            mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
            if mask is not None:
                mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
                overlay = img.copy()
                # Assuming class 1 or something is drivable? 
                # Actually, we should check the class. For now, let's assume non-zero is drivable.
                # A better way: check the mask value for the drivable class.
                # In BDD, drivable is often a specific class.
                # Let's just overlay where mask > 0.
                overlay[mask > 0] = [0, 255, 0] # Green
                cv2.addWeighted(overlay, 0.4, img, 0.6, 0, img)
            else:
                missing_labels.append("drivable_mask")
        else:
            missing_labels.append("drivable_mask")

        # 2. Draw Detection (Red boxes)
        has_det = False
        for label in sample.get('labels', []):
            if 'box2d' in label:
                has_det = True
                box = label['box2d']
                x1 = int(box['x1'] * w / 640)
                y1 = int(box['y1'] * h / 384)
                x2 = int(box['x2'] * w / 640)
                y2 = int(box['y2'] * h / 384)
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2) # Red in RGB
                cv2.putText(img, label.get('category', 'obj'), (x1, y1-5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        if not has_det:
            missing_labels.append("detections")

        # 3. Draw Lanes (Blue lines)
        has_lane = False
        for label in sample.get('labels', []):
            if label.get('category') == 'lane' and 'poly2d' in label:
                has_lane = True
                for poly in label['poly2d']:
                    vertices = poly.get('vertices', [])
                    if len(vertices) >= 2:
                        pts = []
                        for v in vertices:
                            px = int(v[0] * w / 640)
                            py = int(v[1] * h / 384)
                            pts.append([px, py])
                        pts = np.array(pts, np.int32).reshape((-1, 1, 2))
                        cv2.polylines(img, [pts], False, (255, 255, 0), 2) # Cyan/Blue in RGB
        if not has_lane:
            missing_labels.append("lanes")

        # Save
        save_path = f"logs/dataset_preview/sample_{i+1:02d}.png"
        cv2.imwrite(save_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        
        status = f"[MISSING: {', '.join(missing_labels)}]" if missing_labels else "[OK]"
        print(f"Saved {save_path} | {img_name} {status}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", type=str, required=True)
    parser.add_argument("--num_samples", type=int, default=5)
    args = parser.parse_args()
    preview_dataset(args.dataset_root, args.num_samples)
