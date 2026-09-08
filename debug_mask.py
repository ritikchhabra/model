import os
from pathlib import Path
seg_label_dir = Path("/home/piet/Desktop/yolop/archive/bdd100k_seg/bdd100k/seg/labels")
mask_lookup = {}
for split in ["train", "val"]:
    split_dir = seg_label_dir / split
    if split_dir.exists():
        for p in split_dir.glob("*.png"):
            mask_lookup[p.name] = p
print(f"Mask lookup keys (first 5): {list(mask_lookup.keys())[:5]}")
img_stem = "0004a4c0-d4dff0ad"
print(f"Checking pattern: {img_stem}_train_id.png")
print(f"In lookup: {f'{img_stem}_train_id.png' in mask_lookup}")
