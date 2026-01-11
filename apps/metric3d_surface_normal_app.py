
import os
import sys
import torch
import cv2
import numpy as np
import gradio as gr
from PIL import Image
import math

# Ensure Metric3D is in path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
METRIC3D_DIR = os.path.join(PROJECT_ROOT, "models", "Metric3D")

if METRIC3D_DIR not in sys.path:
    sys.path.append(METRIC3D_DIR)

from mono.model.monodepth_model import get_configured_monodepth_model
try:
    from mmcv.utils import Config
except:
    from mmengine import Config

# Global variables
model = None
cfg = None

# Checkpoint handling
CHECKPOINT_URL = "https://huggingface.co/JUGGHM/Metric3D/resolve/main/metric_depth_vit_large_800k.pth"
CHECKPOINT_DIR = os.path.join(os.path.expanduser("~"), ".cache", "torch", "hub", "checkpoints")
CHECKPOINT_FILENAME = "metric_depth_vit_large_800k.pth"
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, CHECKPOINT_FILENAME)

def ensure_checkpoint_exists():
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"Checkpoint not found at {CHECKPOINT_PATH}. Downloading...")
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
        try:
            torch.hub.download_url_to_file(CHECKPOINT_URL, CHECKPOINT_PATH)
            print("Download complete.")
        except Exception as e:
            print(f"Error downloading checkpoint: {e}")
            raise e
    else:
        print(f"Checkpoint found at {CHECKPOINT_PATH}")

ensure_checkpoint_exists()

CONFIG_PATH = os.path.join(METRIC3D_DIR, "mono/configs/HourglassDecoder/vit.raft5.large.py")

# --- Helper Functions from mono/utils/do_test_cpu.py ---

def build_camera_model(H : int, W : int, intrinsics : list) -> np.array:
    fx, fy, u0, v0 = intrinsics
    f = (fx + fy) / 2.0
    # principle point location
    x_row = np.arange(0, W).astype(np.float32)
    x_row_center_norm = (x_row - u0) / W
    x_center = np.tile(x_row_center_norm, (H, 1)) # [H, W]

    y_col = np.arange(0, H).astype(np.float32) 
    y_col_center_norm = (y_col - v0) / H
    y_center = np.tile(y_col_center_norm, (W, 1)).T # [H, W]

    # FoV
    fov_x = np.arctan(x_center / (f / W))
    fov_y = np.arctan(y_center / (f / H))

    cam_model = np.stack([x_center, y_center, fov_x, fov_y], axis=2)
    return cam_model

def resize_for_input(image, output_shape, intrinsic, canonical_shape, to_canonical_ratio):
    padding = [123.675, 116.28, 103.53]
    h, w, _ = image.shape
    resize_ratio_h = output_shape[0] / canonical_shape[0]
    resize_ratio_w = output_shape[1] / canonical_shape[1]
    to_scale_ratio = min(resize_ratio_h, resize_ratio_w)

    resize_ratio = to_canonical_ratio * to_scale_ratio

    reshape_h = int(resize_ratio * h)
    reshape_w = int(resize_ratio * w)

    pad_h = max(output_shape[0] - reshape_h, 0)
    pad_w = max(output_shape[1] - reshape_w, 0)
    pad_h_half = int(pad_h / 2)
    pad_w_half = int(pad_w / 2)

    # resize
    image = cv2.resize(image, dsize=(reshape_w, reshape_h), interpolation=cv2.INTER_LINEAR)
    # padding
    image = cv2.copyMakeBorder(
        image, 
        pad_h_half, 
        pad_h - pad_h_half, 
        pad_w_half, 
        pad_w - pad_w_half, 
        cv2.BORDER_CONSTANT, 
        value=padding)
    
    # Resize, adjust principle point
    intrinsic[2] = intrinsic[2] * to_scale_ratio
    intrinsic[3] = intrinsic[3] * to_scale_ratio

    cam_model = build_camera_model(reshape_h, reshape_w, intrinsic)
    cam_model = cv2.copyMakeBorder(
        cam_model, 
        pad_h_half, 
        pad_h - pad_h_half, 
        pad_w_half, 
        pad_w - pad_w_half, 
        cv2.BORDER_CONSTANT, 
        value=-1)

    pad=[pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half]
    label_scale_factor=1/to_scale_ratio
    return image, cam_model, pad, label_scale_factor

def transform_test_data_scalecano(rgb, intrinsic, data_basic):
    canonical_space = data_basic['canonical_space']
    forward_size = data_basic.crop_size
    mean = torch.tensor([123.675, 116.28, 103.53]).float()[:, None, None]
    std = torch.tensor([58.395, 57.12, 57.375]).float()[:, None, None]

    ori_h, ori_w, _ = rgb.shape
    ori_focal = (intrinsic[0] + intrinsic[1]) / 2
    canonical_focal = canonical_space['focal_length']

    cano_label_scale_ratio = canonical_focal / ori_focal

    canonical_intrinsic = [
        intrinsic[0] * cano_label_scale_ratio,
        intrinsic[1] * cano_label_scale_ratio,
        intrinsic[2],
        intrinsic[3],
    ]

    # resize
    rgb, cam_model, pad, resize_label_scale_ratio = resize_for_input(rgb, forward_size, canonical_intrinsic, [ori_h, ori_w], 1.0)

    # label scale factor
    label_scale_factor = cano_label_scale_ratio * resize_label_scale_ratio

    rgb = torch.from_numpy(rgb.transpose((2, 0, 1))).float()
    rgb = torch.div((rgb - mean), std)
    # rgb = rgb.cpu() # Already cpu
    
    cam_model = torch.from_numpy(cam_model.transpose((2, 0, 1))).float()
    cam_model = cam_model[None, :, :, :] #.cpu()
    cam_model_stacks = [
        torch.nn.functional.interpolate(cam_model, size=(cam_model.shape[2]//i, cam_model.shape[3]//i), mode='bilinear', align_corners=False)
        for i in [2, 4, 8, 16, 32]
    ]
    return rgb, cam_model_stacks, pad, label_scale_factor

def load_model():
    global model, cfg
    if model is None:
        print(f"Loading configuration from {CONFIG_PATH}")
        cfg = Config.fromfile(CONFIG_PATH)
        
        # Adjust config for CPU if needed (though patching removed hardcoded cuda)
        
        print("Building model...")
        model = get_configured_monodepth_model(cfg)
        model = model.cpu()
        
        print(f"Loading checkpoint from {CHECKPOINT_PATH}")
        checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu")
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
            
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        print("Model loaded successfully.")
    return model, cfg

def process_image(input_image):
    if input_image is None:
        return None
    
    model, cfg = load_model()
    
    if isinstance(input_image, Image.Image):
        input_image = np.array(input_image)
    
    # Input is RGB from Gradio
    rgb_origin = input_image
    
    # Intrinsic (Default approximation)
    intrinsic = [1000.0, 1000.0, rgb_origin.shape[1]/2, rgb_origin.shape[0]/2]
    
    # Transform
    rgb_input, _, pad, _ = transform_test_data_scalecano(rgb_origin, intrinsic, cfg.data_basic)
    
    # Inference
    data = dict(
        input=rgb_input.unsqueeze(0), # Add batch dim
    )
    
    with torch.no_grad():
        pred_depth, confidence, output_dict = model.inference(data)
    
    # Extract Normal
    if 'prediction_normal' in output_dict:
        normal_out = output_dict['prediction_normal'][0, :] # Take first from batch
        
        pred_normal = normal_out[:3, :, :]
        H_out, W_out = pred_normal.shape[1:]
        
        # Unpad
        pred_normal = pred_normal[:, pad[0]:H_out-pad[1], pad[2]:W_out-pad[3]]
        
        # Resize to original
        pred_normal = torch.nn.functional.interpolate(pred_normal[None, :], size=[rgb_origin.shape[0], rgb_origin.shape[1]], mode='bilinear', align_corners=True).squeeze()
        
        # Visualize
        pred_normal_vis = pred_normal.cpu().numpy().transpose((1, 2, 0))
        pred_normal_vis = (pred_normal_vis + 1) / 2
        pred_normal_vis = (pred_normal_vis * 255).clip(0, 255).astype(np.uint8)
        
        return Image.fromarray(pred_normal_vis)
        
    return None

# Gradio Interface
with gr.Blocks() as demo:
    gr.Markdown("# Metric3D Surface Normal Estimation (CPU)")
    
    with gr.Row():
        input_img = gr.Image(label="Input Image", type="numpy")
        output_img = gr.Image(label="Surface Normal", type="pil")
    
    submit_btn = gr.Button("Estimate Surface Normal")
    submit_btn.click(fn=process_image, inputs=input_img, outputs=output_img)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0")
