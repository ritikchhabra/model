import os
import argparse
from pathlib import Path
from collections import Counter

def inspect_dataset(root_dir):
    print(f"Inspecting dataset at: {root_dir}")
    print("-" * 40)

    stats = {
        'image_dirs': [],
        'label_dirs': [],
        'extensions': Counter(),
        'file_counts': Counter(),
        'splits': Counter(),
        'annotation_files': []
    }

    root = Path(root_dir)
    
    # Walk through the directory
    for path in root.rglob('*'):
        if path.is_file():
            ext = path.suffix.lower()
            stats['extensions'][ext] += 1
            
            # Check for common annotation files
            if ext in ['.json', '.xml', '.txt'] and any(kw in path.name.lower() for kw in ['label', 'anno', 'bdd']):
                stats['annotation_files'].append(str(path))
            
            # Check for common image files
            if ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                # Try to detect split
                split = 'unknown'
                path_str = str(path).lower()
                if 'train' in path_str:
                    split = 'train'
                elif 'val' in path_str or 'valid' in path_str:
                    split = 'val'
                elif 'test' in path_str:
                    split = 'test'
                
                stats['splits'][split] += 1

        elif path.is_dir():
            # Identify potential image/label directories
            name = path.name.lower()
            if any(kw in name for kw in ['image', 'img']):
                stats['image_dirs'].append(str(path))
            elif any(kw in name for kw in ['label', 'anno']):
                stats['label_dirs'].append(str(path))

    print(f"Image/Label Directories Found:")
    print(f"  Image Dirs: {len(stats['image_dirs'])}")
    print(f"  Label Dirs: {len(stats['label_dirs'])}")
    print("\nFile Extensions Found:")
    for ext, count in stats['extensions'].items():
        print(f"  {ext}: {count}")
    
    print("\nSplits (Images):")
    for split, count in stats['splits'].items():
        print(f"  {split}: {count}")

    print("\nAnnotation Files Found:")
    print(f"  Count: {len(stats['annotation_files'])}")
    if stats['annotation_files']:
        print(f"  Example: {stats['annotation_files'][0]}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect BDD100K dataset archive")
    parser.add_argument("root", type=str, help="Path to the dataset root")
    args = parser.parse_args()
    
    if not os.path.exists(args.root):
        print(f"Error: Path {args.root} does not exist.")
    else:
        inspect_dataset(args.root)
