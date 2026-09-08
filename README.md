# YOLOP — You Only Look Once for Panoptic Driving Perception

Multi-task driving perception model performing simultaneously:
1. **Object Detection** (10 classes: car, truck, bus, person, bike, motor, rider, traffic light, traffic sign, train)
2. **Drivable Area Segmentation** (3 classes: background, direct, alternative)
3. **Lane Detection** (binary mask)

## Project Structure

```
yolop/
├── train.py                  # Training script
├── validate.py               # Validation script
├── inference.py              # Inference + visualization
├── requirements.txt          # Dependencies
├── README.md
├── configs/
│   └── bdd100k.yaml          # Training configuration
├── yolop/
│   ├── backbone/darknet.py   # CSP-Darknet backbone
│   ├── model/yolop.py        # Full 3-head model
│   ├── losses/multi_task_loss.py
│   ├── datasets/bdd100k_dataset.py
│   └── utils/checkpoint.py
├── scripts/
│   ├── verify_bdd100k.py     # Dataset integrity check
│   ├── export_onnx.py        # ONNX export
│   └── ...
└── checkpoints/              # Saved checkpoints (gitignored)
```

## Environment Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Requirements:** Python 3.12, PyTorch 2.14+, CUDA 13.0

## Dataset

Uses BDD100K (archive/ directory). Verify before training:

```bash
python scripts/verify_bdd100k.py --archive /path/to/archive
```

## Training

```bash
# Full training
python train.py --config configs/bdd100k.yaml

# Resume from last checkpoint
python train.py --config configs/bdd100k.yaml --resume last.pt

# Resume from specific epoch
python train.py --config configs/bdd100k.yaml --resume checkpoints/epoch_005.pt

# Smoke test (15 images, 20 epochs overfit)
python train.py --smoke-test --smoke-n 15 --max-epochs 20
```

## Checkpoints

| File | Contents | Size |
|------|----------|------|
| `last.pt` | Full resumable state (epoch, model, optimizer, scheduler) | ~138 MB |
| `best.pt` | Model weights ONLY (`model.state_dict()`) | ~46 MB |
| `checkpoints/epoch_NNN.pt` | Full resumable state (keeps last 3) | ~138 MB |

## Inference

```bash
# Run on a directory of images
python inference.py --weights best.pt --dir path/to/images/ --n 5

# Run on a single image
python inference.py --weights best.pt --image path/to/img.jpg
```

Loads `best.pt` as:
```python
state_dict = torch.load("best.pt", map_location=device)
model.load_state_dict(state_dict)
model.eval()
```

## Model Architecture

```
Input [B, 3, 384, 640]
    ↓
CSP-Darknet Backbone
    ↓          ↓          ↓
  P3 (128ch) P4 (512ch) P5 (1024ch)
  96×160      48×80      24×40
    ↓          ↓          ↓
Det Head p3  Det Head p4  Det Head p5   → Object Detection
    ↓
Seg Head (upsample → 384×640)           → Drivable Area (3-class)
    ↓
Lane Head (upsample → 384×640)          → Lane Mask (binary)
```

## Loss

```
total = λ_det × det_loss + λ_drv × drv_loss + λ_lane × lane_loss

det_loss:  BCEWithLogitsLoss (objectness heatmap, pos_weight=10)
drv_loss:  CrossEntropyLoss  (3-class segmentation)
lane_loss: BCEWithLogitsLoss (binary mask, pos_weight=10)
```
