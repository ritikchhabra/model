"""
YOLOP Inference Script.

Usage:
    python inference.py --weights best.pt --image path/to/img.jpg
    python inference.py --weights best.pt --dir bdd100k_samples/ --n 5
"""

import os
import argparse
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path

from yolop.model.yolop import YOLOP


ORIG_H, ORIG_W = 720, 1280
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess(img_bgr, img_h=384, img_w=640):
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (img_w, img_h))
    img = img.astype(np.float32) / 255.0
    img = (img - MEAN) / STD
    tensor = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).float()
    return tensor


def visualise(orig_bgr, det_outs, drv_out, lane_out, img_h=384, img_w=640):
    h, w = orig_bgr.shape[:2]
    canvas = orig_bgr.copy()

    # ── Drivable area overlay ────────────────────────────────────────
    drv_prob  = F.softmax(drv_out[0], dim=0).cpu().numpy()  # [3, H, W]
    drv_class = np.argmax(drv_prob, axis=0)                  # [H, W]
    drv_resized = cv2.resize(drv_class.astype(np.uint8), (w, h),
                              interpolation=cv2.INTER_NEAREST)
    drv_overlay = canvas.copy()
    drv_overlay[drv_resized == 1] = [0, 200, 0]   # direct  = green
    drv_overlay[drv_resized == 2] = [0, 150, 80]  # alternative = teal
    canvas = cv2.addWeighted(drv_overlay, 0.45, canvas, 0.55, 0)

    # ── Lane overlay ─────────────────────────────────────────────────
    lane_mask = torch.sigmoid(lane_out[0, 0]).cpu().numpy()  # [H, W]
    lane_resized = cv2.resize(lane_mask, (w, h), interpolation=cv2.INTER_LINEAR)
    canvas[lane_resized > 0.4] = (canvas[lane_resized > 0.4] * 0.5
                                    + np.array([0, 0, 255]) * 0.5).astype(np.uint8)

    # ── Detection: top peaks from p3 heatmap ────────────────────────
    det_p3 = torch.sigmoid(det_outs['p3'][0, 0]).cpu().numpy()  # [H_s, W_s]
    flat = det_p3.flatten()
    top_k = min(20, len(flat))
    top_idx = np.argsort(flat)[-top_k:][::-1]
    gs_h, gs_w = det_p3.shape
    for idx in top_idx:
        conf = flat[idx]
        if conf < 0.3:
            continue
        gy, gx = divmod(int(idx), gs_w)
        cx = int(gx / gs_w * w)
        cy = int(gy / gs_h * h)
        bw = int(w * 0.07)
        bh = int(h * 0.10)
        x1, y1 = max(0, cx - bw//2), max(0, cy - bh//2)
        x2, y2 = min(w, cx + bw//2), min(h, cy + bh//2)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(canvas, f'{conf:.2f}', (x1, max(y1-4, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

    return canvas


def run(model, img_bgr, device, img_h=384, img_w=640):
    tensor = preprocess(img_bgr, img_h, img_w).to(device)
    with torch.no_grad():
        outputs = model(tensor)
    return outputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights',  default='best.pt')
    parser.add_argument('--config',   default='configs/bdd100k.yaml')
    parser.add_argument('--image',    default=None, help='Single image path')
    parser.add_argument('--dir',      default=None, help='Directory of images')
    parser.add_argument('--n',        type=int, default=5)
    parser.add_argument('--out-dir',  default='logs/inference_test')
    args = parser.parse_args()

    import yaml
    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load model
    model = YOLOP(
        num_classes_det=config['NUM_CLASSES_DET'],
        num_classes_seg=config['NUM_CLASSES_SEG'],
        img_size=(config['IMG_H'], config['IMG_W']),
    ).to(device)

    print(f"Loading weights: {args.weights}")
    state_dict = torch.load(args.weights, map_location=device, weights_only=True)
    # best.pt is a plain state_dict
    if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
        state_dict = state_dict['model_state_dict']
    model.load_state_dict(state_dict)
    model.eval()
    print("Model loaded.")

    # Collect images
    if args.image:
        img_paths = [Path(args.image)]
    elif args.dir:
        img_paths = sorted(Path(args.dir).glob('*.jpg'))[:args.n]
    else:
        # default: use bdd100k_samples if it exists
        img_paths = sorted(Path('bdd100k_samples').glob('*.jpg'))[:args.n]

    if not img_paths:
        print("No images found.")
        return

    os.makedirs(args.out_dir, exist_ok=True)
    img_h, img_w = config['IMG_H'], config['IMG_W']

    for img_path in img_paths:
        orig = cv2.imread(str(img_path))
        if orig is None:
            print(f"  [SKIP] Cannot read: {img_path}")
            continue

        outputs = run(model, orig, device, img_h, img_w)
        result  = visualise(orig, outputs['det'], outputs['drv'], outputs['lane'],
                             img_h, img_w)

        out_path = os.path.join(args.out_dir, f'pred_{img_path.name}')
        cv2.imwrite(out_path, result)
        print(f"  Saved: {out_path}")

    print("Done.")


if __name__ == '__main__':
    main()
