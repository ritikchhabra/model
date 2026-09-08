"""
YOLOP Training Script.

Usage:
    # Full training:
    python train.py --config configs/bdd100k.yaml

    # Resume from last checkpoint:
    python train.py --config configs/bdd100k.yaml --resume last.pt

    # Resume from specific epoch checkpoint:
    python train.py --config configs/bdd100k.yaml --resume checkpoints/epoch_005.pt

    # Overfit smoke-test (N images, 20 epochs):
    python train.py --config configs/bdd100k.yaml --smoke-test --smoke-n 15

    # Short 1-epoch training test:
    python train.py --config configs/bdd100k.yaml --max-epochs 1 --val-subset 50
"""

import os
import time
import argparse
import yaml
import torch
from torch.utils.data import DataLoader, Subset
from torch.amp import autocast, GradScaler

from yolop.model.yolop import YOLOP
from yolop.losses.multi_task_loss import YOLOPLoss
from yolop.utils.checkpoint import save_checkpoint, load_checkpoint
from yolop.datasets.bdd100k_dataset import BDD100KDataset


# ──────────────────────────────────────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, criterion, scaler, device, epoch):
    model.train()
    total_loss = det_sum = drv_sum = lane_sum = 0.0
    t0 = time.time()
    n  = len(loader)

    for i, batch in enumerate(loader):
        images  = batch['images'].to(device, non_blocking=True)
        targets = {
            'det': {
                scale: t.to(device, non_blocking=True)
                for scale, t in batch['targets']['det'].items()
            },
            'drv':  batch['targets']['drv'].to(device,  non_blocking=True),
            'lane': batch['targets']['lane'].to(device, non_blocking=True),
        }

        optimizer.zero_grad(set_to_none=True)

        with autocast('cuda', enabled=scaler.is_enabled()):
            outputs = model(images)
            loss, det_l, drv_l, lane_l = criterion(outputs, targets)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        det_sum    += det_l.item()
        drv_sum    += drv_l.item()
        lane_sum   += lane_l.item()

        if (i + 1) % max(1, n // 5) == 0:
            print(f"  [{i+1}/{n}] loss={loss.item():.4f}  "
                  f"det={det_l.item():.4f}  "
                  f"drv={drv_l.item():.4f}  "
                  f"lane={lane_l.item():.4f}")

    elapsed = time.time() - t0
    return {
        'total': total_loss / n,
        'det':   det_sum / n,
        'drv':   drv_sum / n,
        'lane':  lane_sum / n,
        'time':  elapsed,
    }


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = det_sum = drv_sum = lane_sum = 0.0
    n = len(loader)

    for batch in loader:
        images  = batch['images'].to(device, non_blocking=True)
        targets = {
            'det': {
                scale: t.to(device, non_blocking=True)
                for scale, t in batch['targets']['det'].items()
            },
            'drv':  batch['targets']['drv'].to(device,  non_blocking=True),
            'lane': batch['targets']['lane'].to(device, non_blocking=True),
        }
        with autocast('cuda', enabled=device.type == 'cuda'):
            outputs = model(images)
            loss, det_l, drv_l, lane_l = criterion(outputs, targets)

        total_loss += loss.item()
        det_sum    += det_l.item()
        drv_sum    += drv_l.item()
        lane_sum   += lane_l.item()

    return {
        'total': total_loss / n,
        'det':   det_sum / n,
        'drv':   drv_sum / n,
        'lane':  lane_sum / n,
    }


# ──────────────────────────────────────────────────────────────────────────────
def build_loaders(config, args):
    archive = args.archive_root or config['ARCHIVE_ROOT']
    img_h, img_w = config['IMG_H'], config['IMG_W']

    n_smoke = args.smoke_n if args.smoke_test else None

    train_ds = BDD100KDataset(archive, split='train',
                               img_size=(img_h, img_w),
                               max_samples=n_smoke)

    val_ds   = BDD100KDataset(archive, split='val',
                               img_size=(img_h, img_w),
                               max_samples=n_smoke)

    # Optionally limit val to a subset for speed
    val_subset = args.val_subset or config.get('VAL_SUBSET_SIZE')
    if val_subset and not args.smoke_test:
        val_ds = Subset(val_ds, list(range(min(int(val_subset), len(val_ds)))))

    nw = 0 if args.smoke_test else config['NUM_WORKERS']
    bs = args.batch_size or config['BATCH_SIZE']

    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,
                               num_workers=nw, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=bs, shuffle=False,
                               num_workers=nw, pin_memory=True)
    return train_loader, val_loader


# ──────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description='YOLOP Training')
    parser.add_argument('--config',       default='configs/bdd100k.yaml')
    parser.add_argument('--resume',       default=None, help='Path to checkpoint')
    parser.add_argument('--archive-root', default=None, help='Override ARCHIVE_ROOT')
    parser.add_argument('--smoke-test',   action='store_true',
                        help='Overfit on a tiny subset (smoke-test)')
    parser.add_argument('--smoke-n',      type=int, default=15,
                        help='Number of images for smoke-test')
    parser.add_argument('--max-epochs',   type=int, default=None)
    parser.add_argument('--batch-size',   type=int, default=None)
    parser.add_argument('--val-subset',   type=int, default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # ── Model ────────────────────────────────────────────────────────
    model = YOLOP(
        num_classes_det=config['NUM_CLASSES_DET'],
        num_classes_seg=config['NUM_CLASSES_SEG'],
        img_size=(config['IMG_H'], config['IMG_W']),
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params/1e6:.1f}M")

    # ── Optimiser & Scheduler ────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['LEARNING_RATE'],
        weight_decay=config['WEIGHT_DECAY'],
    )
    max_ep = args.max_epochs or (30 if args.smoke_test else config['EPOCHS'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max_ep, eta_min=1e-6
    )

    criterion = YOLOPLoss(
        lambda_det=config['LAMBDA_DET'],
        lambda_drv=config['LAMBDA_DRV'],
        lambda_lane=config['LAMBDA_LANE'],
        num_classes_det=config['NUM_CLASSES_DET'],
    ).to(device)

    scaler = GradScaler('cuda', enabled=device.type == 'cuda')

    # ── Resume ───────────────────────────────────────────────────────
    start_epoch = 0
    best_metric = float('inf')
    if args.resume:
        start_epoch, model, optimizer, scheduler, best_metric = \
            load_checkpoint(args.resume, model, optimizer, scheduler)
        start_epoch += 1  # next epoch

    # ── Data ─────────────────────────────────────────────────────────
    train_loader, val_loader = build_loaders(config, args)

    if args.smoke_test:
        print(f"\n{'='*60}")
        print(f"SMOKE TEST: {args.smoke_n} images, {max_ep} epochs")
        print('='*60)

    # ── Training loop ────────────────────────────────────────────────
    for epoch in range(start_epoch, max_ep):
        print(f"\nEpoch [{epoch+1}/{max_ep}]  lr={optimizer.param_groups[0]['lr']:.6f}")

        train_m = train_one_epoch(model, train_loader, optimizer, criterion,
                                   scaler, device, epoch)
        val_m   = validate(model, val_loader, criterion, device)

        scheduler.step()

        print(f"  TRAIN total={train_m['total']:.4f}  "
              f"det={train_m['det']:.4f}  drv={train_m['drv']:.4f}  "
              f"lane={train_m['lane']:.4f}  time={train_m['time']:.1f}s")
        print(f"  VAL   total={val_m['total']:.4f}  "
              f"det={val_m['det']:.4f}  drv={val_m['drv']:.4f}  "
              f"lane={val_m['lane']:.4f}")

        is_best = val_m['total'] < best_metric
        if is_best:
            best_metric = val_m['total']

        state = {
            'epoch':              epoch,
            'model_state_dict':   model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_metric':        best_metric,
            'train_metrics':      train_m,
            'val_metrics':        val_m,
            'config':             config,
        }
        save_checkpoint(
            state, is_best,
            checkpoint_dir=config['CHECKPOINT_DIR'],
            last_path=config['LAST_MODEL_PATH'],
            best_path=config['BEST_MODEL_PATH'],
            keep_last_n=config['KEEP_LAST_N'],
        )

    print("\nDone.")
    if args.smoke_test:
        print(f"Final train loss: {train_m['total']:.4f}  "
              f"(should be decreasing for overfit to work)")


if __name__ == '__main__':
    main()
