import cv2
import torch
import numpy as np
import os
import sys

# Ensure proper path for imports
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)
METRIC3D_DIR = os.path.join(PROJECT_ROOT, "models", "Metric3D")
sys.path.append(METRIC3D_DIR)

from mono.utils.do_test import transform_test_data_scalecano, get_prediction
from mono.model.monodepth_model import get_configured_monodepth_model
from mmengine import Config

# --- Constants ---
MODEL_TYPE = "ViT-Small"
CHECKPOINT_URL = "https://huggingface.co/JUGGHM/Metric3D/resolve/main/metric_depth_vit_small_800k.pth"
CHECKPOINT_FILENAME = "metric_depth_vit_small_800k.pth"
CHECKPOINT_DIR = os.path.join(os.path.expanduser("~"), ".cache", "torch", "hub", "checkpoints")
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, CHECKPOINT_FILENAME)
CONFIG_PATH = os.path.join(METRIC3D_DIR, "mono/configs/HourglassDecoder/vit.raft5.small.py")

DEVICE = "cpu"

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

def load_model():
    ensure_checkpoint_exists()
    print(f"Loading configuration from {CONFIG_PATH}")
    cfg = Config.fromfile(CONFIG_PATH)
    
    cfg.load_from = CHECKPOINT_PATH
    cfg.model.cudnn_benchmark = False
    cfg.model.pretrained = None
    # cfg.model.test_cfg.mode = "whole" # CAUSES ERROR on ViT-Small config
    
    print("Building model...")
    model = get_configured_monodepth_model(cfg)
    model = model.to(DEVICE).eval()
    return model, cfg

def process_frame(model, cfg, frame):
    # Input expected to be RGB
    rgb_origin = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Create dummy intrinsics: [fx, fy, cx, cy]
    h, w = rgb_origin.shape[:2]
    intrinsic = [1000.0, 1000.0, w / 2, h / 2]

    rgb_input, cam_models_stacks, pad, label_scale_factor = transform_test_data_scalecano(
        rgb_origin, 
        intrinsic, 
        cfg.data_basic
    )
    
    with torch.no_grad():
        # transform_test_data_scalecano returns tensors already on device (due to our patch) 
        # but rgb_input is [3, H, W], we need [1, 3, H, W]
        input_batch = rgb_input.unsqueeze(0)
        
        # cam_models_stacks is a list of [1, 4, H, W] tensors
        
        input_data = {
            'input': input_batch,
            'cam_model': cam_models_stacks
        }
        
        # Inference
        # model.inference returns (pred_depth, confidence, output_dict)
        pred_depth, confidence, pred_dict = model.inference(input_data)
        
        # Post-process normals
        if 'prediction_normal' in pred_dict:
            pred_normal = pred_dict['prediction_normal'] # (B, 3, H, W)
            if pred_normal.dim() == 4:
                pred_normal = pred_normal.squeeze(0)
                
            pred_normal = pred_normal.permute(1, 2, 0).cpu().numpy()
            
            # Unpad
            # pad is [pad_h_half, pad_h_remaining, pad_w_half, pad_w_remaining]
            pad_h_half, pad_h_rem, pad_w_half, pad_w_rem = pad
            h_pred, w_pred, _ = pred_normal.shape
            
            # Crop valid area
            pred_normal = pred_normal[pad_h_half : h_pred - pad_h_rem, pad_w_half : w_pred - pad_w_rem, :]
            
            # Resize back to original
            pred_normal = cv2.resize(pred_normal, (w, h), interpolation=cv2.INTER_LINEAR)
            
            # Visualize: (-1, 1) -> (0, 255)
            # Normals are in [-1, 1] range
            pred_normal_vis = (pred_normal + 1) / 2
            pred_normal_vis = (pred_normal_vis * 255).clip(0, 255).astype(np.uint8)
            
            # Convert back to BGR for display
            return cv2.cvtColor(pred_normal_vis, cv2.COLOR_RGB2BGR)
        else:
            # Fallback if no normal output (e.g. depth only model?)
            # But we requested surface normals.
            pass
            
    return frame

def main():
    try:
        model, cfg = load_model()
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    print("Opening camera (0)...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        return
        
    print("Starting video loop. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
            
        # Optional: scale down frame for speed before processing if needed
        # frame_small = cv2.resize(frame, (0,0), fx=0.5, fy=0.5)
        
        output = process_frame(model, cfg, frame)
        
        # Stack vertically or horizontally? Let's just show output
        # Combined view
        vis = np.hstack((frame, output))
        cv2.imshow('Metric3D Surface Normal Live (Left: Input, Right: Prediction)', vis)
        
        if cv2.waitKey(1) == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
