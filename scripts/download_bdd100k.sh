#!/bin/bash

# BDD100K Dataset Downloader/Manager for YOLOP Project
# This script manages the directory structure and verifies downloaded files.
# BDD100K requires manual download from https://bdd-data.berkeley.edu/

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATASET_DIR="$PROJECT_ROOT/datasets/bdd100k"
IMAGES_DIR="$DATASET_DIR/images/100k"
LABELS_DIR="$DATASET_DIR/labels"

echo "=========================================================="
echo " BDD100K Dataset Manager"
echo "=========================================================="

# 1. Create directory structure
echo "[1/4] Creating directory structure..."
mkdir -p "$IMAGES_DIR/train"
mkdir -p "$IMAGES_DIR/val"
mkdir -p "$LABELS_DIR/det_20"
mkdir -p "$LABELS_DIR/drivable"
mkdir -p "$LABELS_DIR/lane"
mkdir -p "$LABELS_DIR/other"

# 2. Disk space check
echo "[2/4] Checking disk space..."
python3 -c "import shutil; usage = shutil.disk_usage('$DATASET_DIR'); print(f'Available space in dataset dir: {usage.free / (1024**3):.2f} GB')"

# 3. Check for existing files and verify
echo "[3/4] Verifying existing files..."
# We expect some files if the user has already started downloading.
# This is a placeholder for actual verification logic.
found_images=$(find "$IMAGES_DIR" -type f | wc -l)
found_labels=$(find "$LABELS_DIR" -type f | wc -l)

echo "    Found $found_images images."
echo "    Found $found_labels label files."

# 4. Manual Download Instructions
echo ""
echo "=========================================================="
echo " MANUAL DOWNLOAD REQUIRED"
echo "=========================================================="
echo "BDD100K is not publicly downloadable via direct link without authentication."
echo "Please follow these steps:"
echo "1. Go to: https://bdd-data.berkeley.edu/"
echo "2. Register and log in."
echo "3. Download the following components:"
echo "   - BDD100K Images (Train/Val)"
echo "   - BDD100K Annotations (Detection, Drivable Area, Lane)"
echo ""
echo "Once downloaded, move them to:"
echo "   - Images -> $IMAGES_DIR"
echo "   - Annotations -> $LABELS_DIR"
echo ""
echo "After moving files, run this script again to verify."
echo "=========================================================="

