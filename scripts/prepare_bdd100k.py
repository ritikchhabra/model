import os
import json
import shutil
from pathlib import Path
import argparse

def prepare_dataset(archive_root, target_dir, limit=None):
    archive_root = Path(archive_root)
    target_dir = Path(target_dir)

    img_src_root = archive_root / "bdd100k/bdd100k/images/100k"
    det_label_dir = archive_root / "bdd100k_labels_release/bdd100k/labels"
    seg_label_dir = archive_root / "bdd100k_seg/bdd100k/seg/labels"

    subdirs = ["images", "labels", "lanes", "drivable"]
    for sd in subdirs:
        (target_dir / sd).mkdir(parents=True, exist_ok=True)

    print(f"Starting preparation...\nTarget: {target_dir}")

    train_json_path = det_label_dir / "bdd100k_labels_images_train.json"
    val_json_path = det_label_dir / "bdd100k_labels_images_val.json"

    with open(train_json_path, "r") as f:
        train_data = json.load(f)
    with open(val_json_path, "r") as f:
        val_data = json.load(f)

    print(f"Loaded {len(train_data)} train and {len(val_data)} val samples.")

    shutil.copy(train_json_path, target_dir / "labels/train_det.json")
    shutil.copy(val_json_path, target_dir / "labels/val_det.json")
    shutil.copy(train_json_path, target_dir / "lanes/train_lanes.json")
    shutil.copy(val_json_path, target_dir / "lanes/val_lanes.json")

    print("Indexing source images...")
    img_map = {}
    for split in ["train", "val", "test"]:
        split_dir = img_src_root / split
        if split_dir.exists():
            for p in split_dir.rglob("*.jpg"):
                img_map[p.name] = p

    print("Symlinking images...")
    count = 0
    for i, sample in enumerate(train_data + val_data):
        if limit and i >= limit:
            break
        img_name = sample["name"]
        img_path = img_map.get(img_name)
        
        if img_path:
            target_img_path = target_dir / "images" / img_name
            if not target_img_path.exists():
                os.symlink(img_path, target_img_path)
            count += 1
        else:
            print(f"Warning: {img_name} not found")

    print(f"Symlinked {count} images.")

    print("Organizing segmentation masks...")
    seg_counts = 0
    # Pre-index mask files for fast lookup
    mask_lookup = {}
    for split in ["train", "val"]:
        split_dir = seg_label_dir / split
        if split_dir.exists():
            for p in split_dir.glob("*.png"):
                # mask name: {img_stem}_{split}_id.png
                # We'll store it by the img_stem
                # For '0004a4c0-d4dff0ad_train_id.png', stem is '0004a4c0-d4dff0ad_train_id'
                # We'll use a prefix match or just check the two possibilities
                mask_lookup[p.name] = p

    # We iterate over symlinked images to find their masks
    for img_path in (target_dir / "images").glob("*.jpg"):
        img_stem = img_path.stem
        mask_found = False
        # Possible mask names for this image
        for split in ["train", "val"]:
            for pattern in [f"{img_stem}_{split}_id.png", f"{img_stem}_id.png"]:
                if pattern in mask_lookup:
                    mask_path = mask_lookup[pattern]
                    target_mask = target_dir / "drivable" / split / mask_path.name
                    if not target_mask.exists():
                        os.symlink(mask_path, target_mask)
                    mask_found = True
                    seg_counts += 1
                    break
            if mask_found: break
            
    print(f"Symlinked {seg_counts} segmentation masks.")
    print("Preparation complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=str, required=True)
    parser.add_argument("--target", type=str, required=True)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    prepare_dataset(args.archive, args.target, args.limit)
