import torch
import os
import glob
from pathlib import Path

def save_checkpoint(state, is_best, checkpoint_dir, last_path, best_path, keep_last_n=3):
    """
    Saves training state for resuming, the best model, and manages epoch checkpoints.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    epoch = state['epoch']
    
    # 1. Save last.pt (full resumable state)
    torch.save(state, last_path)
    
    # 2. Save best.pt (ONLY model state dict)
    if is_best:
        torch.save(state['model_state_dict'], best_path)
        print(f"*** Best model saved to {best_path} ***")

    # 3. Save epoch_XXX.pt
    epoch_filename = f"epoch_{epoch:03d}.pt"
    epoch_path = os.path.join(checkpoint_dir, epoch_filename)
    torch.save(state, epoch_path)
    print(f"Epoch checkpoint saved: {epoch_path}")

    # 4. Manage epoch checkpoints (KEEP_LAST_N)
    checkpoint_files = sorted(glob.glob(os.path.join(checkpoint_dir, "epoch_*.pt")))
    if len(checkpoint_files) > keep_last_n:
        to_delete = checkpoint_files[:-keep_last_n]
        for f in to_delete:
            os.remove(f)
            print(f"Deleted old checkpoint: {f}")

def load_checkpoint(path, model, optimizer=None, scheduler=None):
    """
    Loads a checkpoint. Returns (epoch, model, optimizer, scheduler, best_metric).
    """
    if not os.path.exists(path):
        print(f"Warning: Checkpoint {path} not found.")
        return 0, model, optimizer, scheduler, float('inf')

    print(f"Loading checkpoint: {path}")
    checkpoint = torch.load(path, map_location='cpu')
    
    # Handle both full state and model-only state (for best.pt)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        epoch = checkpoint.get('epoch', 0)
        best_metric = checkpoint.get('best_metric', float('inf'))
        if optimizer and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if scheduler and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        print(f"Resuming from epoch {epoch}")
        return epoch, model, optimizer, scheduler, best_metric
    else:
        # This is likely a best.pt (model-only)
        model.load_state_dict(checkpoint)
        print("Loaded model weights only (no training state).")
        return 0, model, optimizer, scheduler, float('inf')
