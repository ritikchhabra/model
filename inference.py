import os
import cv2
import torch
import numpy as np
from pathlib import Path
from yolop.model.yolop import YOLOP
import torchvision.transforms as transforms

def run_inference():
    os.makedirs("logs/inference_test", exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
    print("Loading model...")
    model = YOLOP(num_classes_det=10, num_classes_seg=19, num_lanes=4).to(device)
    try:
        model.load_state_dict(torch.load("checkpoints/best.pt", map_location=device, weights_only=True))
    except Exception as e:
        print(f"Warning: Failed to load best.pt ({e}). Using uninitialized weights.")
    model.eval()

    # Preprocessing
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((384, 640))
    ])

    image_paths = list(Path("bdd100k_samples").glob("*.jpg"))
    if not image_paths:
        print("No test images found.")
        return

    print(f"Running inference on {len(image_paths)} images...")
    with torch.no_grad():
        for i, img_path in enumerate(image_paths):
            print(f"Processing {img_path.name}")
            
            # Read image
            orig_img = cv2.imread(str(img_path))
            if orig_img is None:
                continue
            
            # Convert BGR to RGB and normalize
            img_rgb = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
            input_tensor = transform(img_rgb).unsqueeze(0).to(device)
            
            # Forward pass
            outputs = model(input_tensor)
            
            # Process outputs (naive heuristic)
            h, w = orig_img.shape[:2]
            overlay = orig_img.copy()

            # 1. Drivable Area (argmax of p3 output)
            drv_p3 = outputs["drv"]["p3"][0]  # [C, H, W]
            drv_mask = torch.argmax(drv_p3, dim=0).cpu().numpy()
            
            # Resize mask to original image size
            drv_mask_resized = cv2.resize(drv_mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
            
            # Overlay Drivable Area (Green)
            overlay[drv_mask_resized > 0] = [0, 255, 0]

            # 2. Lanes (threshold of lane output)
            lane_out = outputs["lane"][0]  # [C, H, W]
            lane_mask = torch.max(lane_out, dim=0)[0].cpu().numpy()
            lane_mask_resized = cv2.resize(lane_mask, (w, h), interpolation=cv2.INTER_LINEAR)
            
            # Overlay Lanes (Blue)
            overlay[lane_mask_resized > 0.5] = [255, 0, 0]

            # 3. Detections (find peaks in det_out p3)
            det_p3 = outputs["det"]["p3"][0] # [15, H, W]
            # Sum over classes (first 10 channels) to find objectness peak
            obj_map = torch.sum(det_p3[:10], dim=0).cpu().numpy()
            
            # Find top 5 peaks
            flat_indices = np.argsort(obj_map.flatten())[-5:]
            for idx in flat_indices:
                py, px = np.unravel_index(idx, obj_map.shape)
                val = obj_map[py, px]
                if val > 0.5: # arbitrary threshold
                    # Map to original image size
                    x_center = int(px * w / obj_map.shape[1])
                    y_center = int(py * h / obj_map.shape[0])
                    # Draw arbitrary box
                    box_w, box_h = int(w * 0.1), int(h * 0.1)
                    x1, y1 = max(0, x_center - box_w//2), max(0, y_center - box_h//2)
                    x2, y2 = min(w, x_center + box_w//2), min(h, y_center + box_h//2)
                    cv2.rectangle(orig_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            
            # Blend
            final_img = cv2.addWeighted(overlay, 0.4, orig_img, 0.6, 0)
            
            # Save
            out_path = f"logs/inference_test/pred_{img_path.name}"
            cv2.imwrite(out_path, final_img)
            print(f"Saved {out_path}")

if __name__ == "__main__":
    run_inference()
