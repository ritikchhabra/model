import torch
import argparse
import os
from yolop.model.yolop import YOLOP
import yaml

def export(args):
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
        import torch
        checkpoint = torch.load(args.weights, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
    
    model.eval()
    dummy_input = torch.randn(1, 3, 384, 640)
    
    onnx_path = args.output
    print(f"Exporting to {onnx_path}...")
    
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['det_p3', 'det_p4', 'det_p5', 'drv_p3', 'drv_p4', 'lane'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'det_p3': {0: 'batch_size'},
            'det_p4': {0: 'batch_size'},
            'det_p5': {0: 'batch_size'},
            'drv_p3': {0: 'batch_size'},
            'drv_p4': {0: 'batch_size'},
            'lane': {0: 'batch_size'}
        }
    )
    
    print(f"ONNX export complete. Size: {os.path.getsize(onnx_path)/1e6:.2f} MB")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--output", type=str, default="yolop.onnx")
    parser.add_argument("--config", type=str, default="configs/bdd100k.yaml")
    args = parser.parse_args()
    export(args)
