import torch
import argparse
import os
import time
from pathlib import Path
from yolop.model.yolop import YOLOP
from yolop.losses.multi_task_loss import YOLOPLoss
from yolop.utils.checkpoint import save_checkpoint, load_checkpoint
from yolop.datasets.bdd100k_dataset import YOLOPDataset
from torch.utils.data import DataLoader
import yaml
import subprocess

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    start_time = time.time()
    
    for batch in loader:
        images = batch['images'].to(device)
        targets = {k: {sk: sv.to(device) for sk, sv in v.items()} if isinstance(v, dict) else v.to(device) 
                   for k, v in batch['targets'].items()}
        
        optimizer.zero_grad()
        outputs = model(images)
        loss, det_loss, drv_loss, lane_loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
    elapsed = time.time() - start_time
    return running_loss / len(loader), elapsed

def validate_one_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_det = 0.0
    total_drv = 0.0
    total_lane = 0.0

    with torch.no_grad():
        for batch in loader:
            images = batch['images'].to(device)
            targets = {k: {sk: sv.to(device) for sk, sv in v.items()} if isinstance(v, dict) else v.to(device) 
                       for k, v in batch['targets'].items()}
            
            outputs = model(images)
            loss, det_loss, drv_loss, lane_loss = criterion(outputs, targets)
            
            total_loss += loss.item()
            total_det += det_loss.item()
            total_drv += drv_loss.item()
            total_lane += lane_loss.item()

    n = len(loader)
    return total_loss/n, total_det/n, total_drv/n, total_lane/n

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/bdd100k.yaml")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--dataset_root", type=str, default=None, help="Override dataset directory")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--skip-preview", action="store_true")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # 1. Dataset Root resolution
    dataset_root = args.dataset_root if args.dataset_root else config.get('DATASET_DIR')
    if not dataset_root:
        raise ValueError("Error: No dataset path provided in config or CLI.")
    
    dataset_path = Path(dataset_root)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Error: Dataset directory not found at {dataset_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = YOLOP(
        num_classes_det=config['NUM_CLASSES_DET'],
        num_classes_seg=config['NUM_CLASSES_SEG'],
        num_lanes=config['NUM_LANES']
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config['LEARNING_RATE'])
    criterion = YOLOPLoss(
        lambda_det=config['LAMBDA_DET'],
        lambda_drv=config['LAMBDA_DRV'],
        lambda_lane=config['LAMBDA_LANE']
    ).to(device)

    start_epoch = 0
    best_metric = float('inf')

    if args.resume:
        start_epoch, model, optimizer, _, best_metric = load_checkpoint(
            args.resume, model, optimizer, None, best_metric
        )
        start_epoch += 1

    if args.smoke_test:
        print("Running Smoke Test (Short)...")
        class DummyLoader:
            def __init__(self, size): self.size = size
            def __len__(self): return self.size
            def __iter__(self):
                for _ in range(self.size):
                    yield {'images': torch.randn(1, 3, 384, 640), 
                           'targets': {'det': {'p3':torch.randn(1,15,96,160), 'p4':torch.randn(1,15,48,80), 'p5':torch.randn(1,15,24,40)}, 
                                       'drv': {'p3':torch.randint(0,19,(1,96,160)), 'p4':torch.randint(0,19,(1,48,80))}, 
                                       'lane': torch.randn(1,4,96,160)}}
        
        loader = DummyLoader(5)
        for epoch in range(start_epoch, start_epoch + 2):
            avg_loss, epoch_time = train_one_epoch(model, loader, optimizer, criterion, device)
            print(f"Epoch [{epoch+1}/{config['EPOCHS']}] | Loss: {avg_loss:.4f} | Time: {epoch_time:.2f}s")
            
            state = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_metric': best_metric,
                'config': config
            }
            save_checkpoint(state, False, config['CHECKPOINT_DIR'], 
                           os.path.join(config['CHECKPOINT_DIR'], 'last.pt'),
                           os.path.join(config['CHECKPOINT_DIR'], 'best.pt'),
                           keep_last_n=config['KEEP_LAST_N'])
        return

    # 3. Preview Step
    if not args.skip_preview:
        print("Running Pre-training Dataset Preview...")
        try:
            # Use the venv python to run the preview script
            subprocess.run(["./.venv/bin/python3", "scripts/preview_dataset.py", "--dataset_root", str(dataset_path)], check=True)
        except subprocess.CalledProcessError as e:
            print(f"Preview failed: {e}")
            raise

    print("Starting Full Training...")
    train_dataset = YOLOPDataset(dataset_path, split='train', img_size=(config['IMG_H'], config['IMG_W']))
    train_loader = DataLoader(train_dataset, batch_size=config['BATCH_SIZE'], shuffle=True, num_workers=config['NUM_WORKERS'])

    # Validation Loader
    val_dataset = YOLOPDataset(dataset_path, split='val', img_size=(config['IMG_H'], config['IMG_W']))
    
    # Handle subset validation if config says so
    val_subset_size = config.get('VAL_SUBSET_SIZE')
    if val_subset_size and val_subset_size > 0:
        print(f"Validating on subset of size {val_subset_size}...")
        from torch.utils.data import Subset
        indices = list(range(min(val_subset_size, len(val_dataset))))
        val_dataset = Subset(val_dataset, indices)

    val_loader = DataLoader(val_dataset, batch_size=config['BATCH_SIZE'], shuffle=False, num_workers=config['NUM_WORKERS'])

    total_start_time = time.time()
    for epoch in range(start_epoch, config['EPOCHS']):
        avg_train_loss, train_epoch_time = train_one_epoch(model, train_loader, optimizer, criterion, device)
        
        # Run Validation
        avg_val_loss, val_det, val_drv, val_lane = validate_one_epoch(model, val_loader, criterion, device)
        
        print(f"Epoch [{epoch+1}/{config['EPOCHS']}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Det: {val_det:.3f} | Val Drv: {val_drv:.3f} | Val Lane: {val_lane:.3f} | Time: {train_epoch_time:.2f}s")
        
        current_metric = avg_val_loss 
        is_best = current_metric < best_metric
        if is_best:
            best_metric = current_metric

        state = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_metric': best_metric,
            'config': config
        }
        save_checkpoint(state, is_best, config['CHECKPOINT_DIR'], 
                       os.path.join(config['CHECKPOINT_DIR'], 'last.pt'),
                       os.path.join(config['CHECKPOINT_DIR'], 'best.pt'),
                       keep_last_n=config['KEEP_LAST_N'])
        
        print(f"Checkpoint size (last.pt): {os.path.getsize(os.path.join(config['CHECKPOINT_DIR'], 'last.pt'))/1e6:.2f} MB")

    total_elapsed = time.time() - total_start_time
    print(f"\nTraining Complete! Total Time: {total_elapsed:.2f}s")

if __name__ == "__main__":
    main()
