
import os
import sys
import torch
import cv2
import numpy as np
import gradio as gr
from PIL import Image

# Ensure Metric3D is in path if needed/loading locally
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
METRIC3D_DIR = os.path.join(PROJECT_ROOT, "models", "Metric3D")

if METRIC3D_DIR not in sys.path:
    sys.path.append(METRIC3D_DIR)

# Global device variable
device = "cpu"
print(f"Running on device: {device}")

# Global model variable
model = None

def load_model():
    global model
    if model is None:
        print("Loading Metric3D ViT-Large model...")
        try:
             # Load from local source using torch.hub
            model = torch.hub.load(METRIC3D_DIR, 'metric3d_vit_large', source='local', pretrain=True)
        except Exception as e:
            print(f"Failed to load via torch.hub: {e}")
            print("Attempting manual import...")
            # Fallback manual import if hub fails
            try:
                from hubconf import metric3d_vit_large
                model = metric3d_vit_large(pretrain=True)
            except ImportError as ie:
                print(f"Manual import failed: {ie}")
                raise ie
        
        model.to(device).eval()
        print("Model loaded.")
    return model

def process_image(input_image):
    if input_image is None:
        return None
    
    # Load model if not loaded
    model = load_model()
    
    # Convert to RGB if needed
    if isinstance(input_image, Image.Image):
        input_image = np.array(input_image)
    
    rgb_origin = input_image # Already RGB from Gradio
    
    # Pre-processing from hubconf.py
    input_size = (616, 1064) # for vit model
    h, w = rgb_origin.shape[:2]
    scale = min(input_size[0] / h, input_size[1] / w)
    rgb = cv2.resize(rgb_origin, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR)
    
    # Padding
    padding = [123.675, 116.28, 103.53]
    h, w = rgb.shape[:2]
    pad_h = input_size[0] - h
    pad_w = input_size[1] - w
    pad_h_half = pad_h // 2
    pad_w_half = pad_w // 2
    
    rgb = cv2.copyMakeBorder(rgb, pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half, cv2.BORDER_CONSTANT, value=padding)
    pad_info = [pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half]

    # Normalize
    mean = torch.tensor([123.675, 116.28, 103.53]).float()[:, None, None]
    std = torch.tensor([58.395, 57.12, 57.375]).float()[:, None, None]
    rgb_tensor = torch.from_numpy(rgb.transpose((2, 0, 1))).float()
    rgb_tensor = torch.div((rgb_tensor - mean), std)
    rgb_tensor = rgb_tensor[None, :, :, :].to(device)

    # Inference
    with torch.no_grad():
        pred_depth, confidence, output_dict = model.inference({'input': rgb_tensor})

    # Extract Surface Normal
    if 'prediction_normal' in output_dict:
        pred_normal = output_dict['prediction_normal'][:, :3, :, :]
        # un pad
        pred_normal = pred_normal.squeeze()
        pred_normal = pred_normal[:, pad_info[0] : pred_normal.shape[1] - pad_info[1], pad_info[2] : pred_normal.shape[2] - pad_info[3]]
        
        # Resize back to original size
        # Use bilinear interpolation for resizing
        pred_normal = torch.nn.functional.interpolate(pred_normal[None, :, :, :], size=rgb_origin.shape[:2], mode='bilinear', align_corners=False).squeeze()
        
        # Visualize
        pred_normal_vis = pred_normal.cpu().numpy().transpose((1, 2, 0))
        pred_normal_vis = (pred_normal_vis * 255).astype(np.uint8)
        
        return Image.fromarray(pred_normal_vis)
    else:
        return None

# Gradio Interface
with gr.Blocks() as demo:
    gr.Markdown("# Metric3D Surface Normal Estimation")
    
    with gr.Row():
        input_img = gr.Image(label="Input Image", type="numpy")
        output_img = gr.Image(label="Surface Normal", type="pil")
    
    submit_btn = gr.Button("Estimate Surface Normal")
    submit_btn.click(fn=process_image, inputs=input_img, outputs=output_img)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0")
