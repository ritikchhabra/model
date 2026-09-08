"""
BDD100K Dataset Verification Script.

Usage:
    python scripts/verify_bdd100k.py --archive archive

Reports:
    - image counts per split
    - label entry counts
    - matched image/label counts
    - missing images (label has no image)
    - missing labels (image has no label)
    - malformed label entries (missing box2d / poly2d)
    - drivable area and lane label stats
"""

import json
import argparse
from pathlib import Path
import sys


def verify_split(archive_root, split):
    print(f"\n{'='*60}")
    print(f"Split: {split.upper()}")
    print('='*60)

    img_dir = archive_root / 'bdd100k' / 'bdd100k' / 'images' / '100k' / split
    label_file = (archive_root / 'bdd100k_labels_release' / 'bdd100k' / 'labels'
                  / f'bdd100k_labels_images_{split}.json')

    errors = []

    # ── Images ──────────────────────────────────────────────────────
    if not img_dir.exists():
        errors.append(f"[FATAL] Image directory not found: {img_dir}")
        print('\n'.join(errors))
        return errors

    img_files = sorted(img_dir.glob('*.jpg'))
    img_stems  = {p.stem for p in img_files}
    print(f"Images on disk : {len(img_files)}")

    # ── Labels ──────────────────────────────────────────────────────
    if not label_file.exists():
        errors.append(f"[FATAL] Label file not found: {label_file}")
        print('\n'.join(errors))
        return errors

    with open(label_file) as f:
        all_labels = json.load(f)

    label_stems = {}
    for entry in all_labels:
        stem = Path(entry['name']).stem
        label_stems[stem] = entry

    print(f"Label entries  : {len(all_labels)}")

    # ── Matching ────────────────────────────────────────────────────
    matched = img_stems & set(label_stems.keys())
    img_no_label   = img_stems - set(label_stems.keys())
    label_no_img   = set(label_stems.keys()) - img_stems

    print(f"Matched        : {len(matched)}")
    if img_no_label:
        errors.append(f"[WARN] {len(img_no_label)} images have no label entry")
        print(f"  Images without labels: {len(img_no_label)}")
        for s in sorted(img_no_label)[:5]:
            print(f"    {s}.jpg")
        if len(img_no_label) > 5:
            print(f"    ... ({len(img_no_label)-5} more)")

    if label_no_img:
        print(f"  Labels without images (on disk): {len(label_no_img)}")

    # ── Label content analysis ──────────────────────────────────────
    n_det = n_drv = n_lane = 0
    n_malformed = 0
    for stem in matched:
        entry = label_stems[stem]
        labels = entry.get('labels', [])
        if not labels:
            n_malformed += 1
            continue
        for lbl in labels:
            cat = lbl.get('category', '')
            if cat == 'drivable area':
                n_drv += 1
            elif cat == 'lane':
                n_lane += 1
            elif 'box2d' in lbl:
                n_det += 1

    print(f"\nLabel content (matched samples):")
    print(f"  Detection boxes   : {n_det}")
    print(f"  Drivable polygons : {n_drv}")
    print(f"  Lane polylines    : {n_lane}")
    print(f"  Empty label files : {n_malformed}")

    if n_malformed > 0:
        errors.append(f"[WARN] {n_malformed} matched images have empty labels")

    if n_det == 0:
        errors.append(f"[ERROR] No detection boxes found in {split} set!")
    if n_drv == 0:
        errors.append(f"[ERROR] No drivable area labels found in {split} set!")
    if n_lane == 0:
        errors.append(f"[ERROR] No lane labels found in {split} set!")

    if errors:
        print(f"\nIssues:")
        for e in errors:
            print(f"  {e}")
    else:
        print(f"\n  [OK] No issues found.")

    return errors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--archive', default='archive',
                        help='Path to archive/ root')
    args = parser.parse_args()

    archive = Path(args.archive)
    if not archive.exists():
        print(f"[FATAL] Archive root not found: {archive.resolve()}")
        sys.exit(1)

    print(f"BDD100K Verification")
    print(f"Archive root: {archive.resolve()}")

    all_errors = []
    for split in ['train', 'val']:
        errs = verify_split(archive, split)
        all_errors.extend(errs)

    print(f"\n{'='*60}")
    if all_errors:
        print(f"Total issues found: {len(all_errors)}")
        for e in all_errors:
            print(f"  {e}")
        sys.exit(1)
    else:
        print("All checks passed.")


if __name__ == '__main__':
    main()
