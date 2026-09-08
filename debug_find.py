from pathlib import Path
import os

def find_image(img_name, img_src_root):
    for split in ["train", "val", "test"]:
        split_dir = img_src_root / split
        if not split_dir.exists():
            print(f"Split dir {split_dir} does not exist")
            continue
        print(f"Searching in {split_dir} for {img_name}")
        for p in split_dir.rglob(img_name):
            return p
    return None

img_src_root = Path("/home/piet/Desktop/yolop/archive/bdd100k/bdd100k/images/100k")
print(find_image("0000f77c-6257be58.jpg", img_src_root))
