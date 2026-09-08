"""
Checkpoint utilities for YOLOP.

Rules:
  - last.pt      = full resumable state (epoch, model, optimizer, scheduler, metric, config)
  - best.pt      = model.state_dict() ONLY  (torch.save(model.state_dict(), path))
  - epoch_XXX.pt = full resumable state, keep only KEEP_LAST_N newest

Checkpoint sizes are printed after every save.
"""

import os
import glob
import torch
from pathlib import Path


def _size_mb(path):
    return os.path.getsize(path) / 1e6


def save_checkpoint(state, is_best, checkpoint_dir, last_path, best_path, keep_last_n=3):
    """
    state must contain:
        epoch, model_state_dict, optimizer_state_dict,
        scheduler_state_dict (or None), best_metric, config
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    epoch = state['epoch']

    # 1. Save last.pt (full resumable)
    torch.save(state, last_path)
    print(f"  last.pt  saved  ({_size_mb(last_path):.1f} MB)")

    # 2. Save best.pt — model weights ONLY
    if is_best:
        torch.save(state['model_state_dict'], best_path)
        print(f"  *** best.pt saved ({_size_mb(best_path):.1f} MB) ***")

    # 3. Save epoch_XXX.pt
    epoch_path = os.path.join(checkpoint_dir, f"epoch_{epoch:03d}.pt")
    torch.save(state, epoch_path)
    print(f"  epoch_{epoch:03d}.pt saved ({_size_mb(epoch_path):.1f} MB)")

    # 4. Prune old epoch checkpoints
    epoch_ckpts = sorted(glob.glob(os.path.join(checkpoint_dir, "epoch_*.pt")))
    if len(epoch_ckpts) > keep_last_n:
        for old in epoch_ckpts[:-keep_last_n]:
            os.remove(old)
            print(f"  deleted old checkpoint: {os.path.basename(old)}")


def load_checkpoint(path, model, optimizer=None, scheduler=None):
    """
    Returns (start_epoch, model, optimizer, scheduler, best_metric).
    Handles both full-state checkpoints and best.pt (model weights only).
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    print(f"Loading checkpoint: {path}  ({_size_mb(path):.1f} MB)")
    ckpt = torch.load(path, map_location='cpu', weights_only=True)

    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
        epoch       = ckpt.get('epoch', 0)
        best_metric = ckpt.get('best_metric', float('inf'))
        if optimizer is not None and 'optimizer_state_dict' in ckpt:
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        if scheduler is not None and 'scheduler_state_dict' in ckpt and ckpt['scheduler_state_dict']:
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        print(f"  Resuming from epoch {epoch+1}  (best_metric={best_metric:.4f})")
        return epoch, model, optimizer, scheduler, best_metric
    else:
        # best.pt: plain state_dict
        model.load_state_dict(ckpt)
        print("  Loaded model weights only (inference mode).")
        return 0, model, optimizer, scheduler, float('inf')
