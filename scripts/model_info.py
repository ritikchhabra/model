import torch
import yaml
import argparse
import os
from yolop.model.yolop import YOLOP

def print_shapes(obj, indent=2):
    if isinstance(obj, torch.Tensor):
        print("  " * indent + f"shape: {tuple(obj.shape)}")
    elif isinstance(obj, dict):
        for k, v in obj.items():
            print("  " * indent + f"{k}:")
            print_shapes(v, indent + 1)
    else:
        print("  " * indent + f"value: {obj}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--config", type=str, default="configs/bdd100k.yaml")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    model = YOLOP(
        num_classes_det=config['NUM_CLASSES_DET'],
        num_classes_seg=config['NUM_CLASSES_SEG'],
        num_lanes=config['NUM_LANES']
    )
    
    if args.weights.endswith("best.pt"):
        model.load_state_dict(torch.load(args.weights, map_location='cpu'))
    else:
        checkpoint = torch.load(args.weights, map_location='cpu')
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Parameters: {total_params / 1e6:.2f} M")
    print(f"Trainable Parameters: {trainable_params / 1e6:.2f} M")
    print(f"FP32 weight size: ~{total_params * 4 / 1e6:.2f} MB")
    print(f"FP16 weight size: ~{total_params * 2 / 1e6:.2f} MB")

    # Input shape
    input_shape = (1, 3, 384, 640)
    model.eval()
    with torch.no_grad():
        dummy_input = torch.randn(*input_shape)
        outputs = model(dummy_input)
        
    print(f"Input shape: {input_shape}")
    print("Outputs:")
    print_shapes(outputs)

if __name__ == "__main__":
    main()
