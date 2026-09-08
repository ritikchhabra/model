import torch
import argparse
import os
import yaml
from yolop.model.yolop import YOLOP

def export(args):
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    img_h = config.get('IMG_H', 384)
    img_w = config.get('IMG_W', 640)

    model = YOLOP(
        num_classes_det=config.get('NUM_CLASSES_DET', 10),
        num_classes_seg=config.get('NUM_CLASSES_SEG', 3),
        img_size=(img_h, img_w)
    )
    
    if os.path.exists(args.weights):
        weights_data = torch.load(args.weights, map_location='cpu')
        if isinstance(weights_data, dict) and 'model_state_dict' in weights_data:
            model.load_state_dict(weights_data['model_state_dict'])
        else:
            model.load_state_dict(weights_data)
        print(f"Loaded weights from {args.weights}")
    else:
        print("Warning: Weights file not found. Exporting default model architecture.")
    
    model.eval()
    dummy_input = torch.randn(1, 3, img_h, img_w)
    
    onnx_path = args.output
    os.makedirs(os.path.dirname(onnx_path) if os.path.dirname(onnx_path) else '.', exist_ok=True)
    print(f"Exporting ONNX model to {onnx_path}...")
    
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['det_p3', 'det_p4', 'det_p5', 'drivable', 'lane'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'det_p3': {0: 'batch_size'},
            'det_p4': {0: 'batch_size'},
            'det_p5': {0: 'batch_size'},
            'drivable': {0: 'batch_size'},
            'lane': {0: 'batch_size'}
        }
    )
    
    size_mb = os.path.getsize(onnx_path) / (1024 * 1024)
    print(f"ONNX export successful: {onnx_path} ({size_mb:.2f} MB)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export YOLOP PyTorch model to ONNX format")
    parser.add_argument("--weights", type=str, default="best.pt")
    parser.add_argument("--output", type=str, default="checkpoints/yolop.onnx")
    parser.add_argument("--config", type=str, default="configs/bdd100k.yaml")
    args = parser.parse_args()
    export(args)
