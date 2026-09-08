import torch
import numpy as np
import onnx
import onnxruntime as ort
import argparse
import yaml
import os
from yolop.model.yolop import YOLOP

def verify_onnx(args):
    print("=== Step 1: Validating ONNX graph structure ===")
    onnx_model = onnx.load(args.onnx)
    onnx.checker.check_model(onnx_model)
    print("ONNX model structure is valid.")

    print("=== Step 2: Preparing PyTorch Model ===")
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    img_h = config.get('IMG_H', 384)
    img_w = config.get('IMG_W', 640)

    py_model = YOLOP(
        num_classes_det=config.get('NUM_CLASSES_DET', 10),
        num_classes_seg=config.get('NUM_CLASSES_SEG', 3),
        img_size=(img_h, img_w)
    )
    if os.path.exists(args.weights):
        weights_data = torch.load(args.weights, map_location='cpu')
        if isinstance(weights_data, dict) and 'model_state_dict' in weights_data:
            py_model.load_state_dict(weights_data['model_state_dict'])
        else:
            py_model.load_state_dict(weights_data)
        print(f"Loaded PyTorch weights from {args.weights}")
    
    py_model.eval()

    # Generate test tensor
    test_input_np = np.random.randn(1, 3, img_h, img_w).astype(np.float32)
    test_input_torch = torch.from_numpy(test_input_np)

    # PyTorch inference
    with torch.no_grad():
        py_outputs = py_model(test_input_torch)

    py_det_p3 = py_outputs['det']['p3'].numpy()
    py_det_p4 = py_outputs['det']['p4'].numpy()
    py_det_p5 = py_outputs['det']['p5'].numpy()
    py_drv    = py_outputs['drv'].numpy()
    py_lane   = py_outputs['lane'].numpy()

    print("=== Step 3: Running ONNX Runtime Inference ===")
    session = ort.InferenceSession(args.onnx, providers=['CPUExecutionProvider'])
    ort_inputs = {session.get_inputs()[0].name: test_input_np}
    ort_outputs = session.run(None, ort_inputs)

    onnx_det_p3, onnx_det_p4, onnx_det_p5, onnx_drv, onnx_lane = ort_outputs

    print("=== Step 4: Comparing Outputs ===")
    det_p3_diff = np.max(np.abs(py_det_p3 - onnx_det_p3))
    det_p4_diff = np.max(np.abs(py_det_p4 - onnx_det_p4))
    det_p5_diff = np.max(np.abs(py_det_p5 - onnx_det_p5))
    drv_diff    = np.max(np.abs(py_drv - onnx_drv))
    lane_diff   = np.max(np.abs(py_lane - onnx_lane))

    print(f"Det P3 max absolute error    : {det_p3_diff:.6f}")
    print(f"Det P4 max absolute error    : {det_p4_diff:.6f}")
    print(f"Det P5 max absolute error    : {det_p5_diff:.6f}")
    print(f"Drivable area max abs error  : {drv_diff:.6f}")
    print(f"Lane line max abs error      : {lane_diff:.6f}")

    np.testing.assert_allclose(py_det_p3, onnx_det_p3, rtol=1e-3, atol=1e-4)
    np.testing.assert_allclose(py_det_p4, onnx_det_p4, rtol=1e-3, atol=1e-4)
    np.testing.assert_allclose(py_det_p5, onnx_det_p5, rtol=1e-3, atol=1e-4)
    np.testing.assert_allclose(py_drv, onnx_drv, rtol=1e-3, atol=1e-4)
    np.testing.assert_allclose(py_lane, onnx_lane, rtol=1e-3, atol=1e-4)

    print("\nSUCCESS: All 5 ONNX output tensors match PyTorch outputs within numerical tolerance!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test and compare exported ONNX model against PyTorch")
    parser.add_argument("--onnx", type=str, default="checkpoints/yolop.onnx")
    parser.add_argument("--weights", type=str, default="best.pt")
    parser.add_argument("--config", type=str, default="configs/bdd100k.yaml")
    args = parser.parse_args()
    verify_onnx(args)
