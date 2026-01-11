import cv2
import torch
import numpy as np
import os
import sys
import time

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
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "captured_images")

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
    
    print("Building model...")
    model = get_configured_monodepth_model(cfg)
    model = model.to(DEVICE).float().eval() # Ensure float32 logic on CPU
    return model, cfg

def ensure_frame_shape(frame):
    print(f"DEBUG: Input frame shape: {frame.shape}")
    # Handle flattened frames (RPi libcamerify/OpenCV quirk)
    if len(frame.shape) == 3 and frame.shape[0] == 1 and frame.shape[1] > 10000:
        flat_size = frame.shape[1]
        resolutions = [
            (1280, 720), (1920, 1080), (640, 480), (800, 600), (1024, 768), (1536, 864)
        ]
        for w_cand, h_cand in resolutions:
            if w_cand * h_cand == flat_size:
                print(f"DEBUG: Reshaping flattened frame to ({h_cand}, {w_cand}, 3)")
                return frame.reshape((h_cand, w_cand, 3))
    
    # 2D case
    if len(frame.shape) == 2 and frame.shape[0] == 1 and frame.shape[1] > 30000:
         flat_size = frame.shape[1] // 3
         resolutions = [
            (1280, 720), (1920, 1080), (640, 480), (800, 600), (1024, 768), (1536, 864)
        ]
         for w_cand, h_cand in resolutions:
            if w_cand * h_cand == flat_size:
                 print(f"DEBUG: Reshaping flattened 2D frame to ({h_cand}, {w_cand}, 3)")
                 return frame.reshape((h_cand, w_cand, 3))
    return frame

def open_camera():
    indices = [0, 1, -1]
    backends = [cv2.CAP_ANY, cv2.CAP_V4L2]
    
    for backend in backends:
        for index in indices:
            print(f"Trying camera index {index} with backend {backend}...")
            cap = cv2.VideoCapture(index, backend)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    print(f"Success! Camera found at index {index}.")
                    return cap
                else:
                    cap.release()
            
    return None

def process_image(model, cfg, frame):
    # Input expected to be RGB
    rgb_origin = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Check for valid image content
    if np.all(rgb_origin == 0):
        print("WARNING: Input image is completely black!")
    
    # Create dummy intrinsics
    h, w = rgb_origin.shape[:2]
    intrinsic = [1000.0, 1000.0, w / 2, h / 2]

    rgb_input, cam_models_stacks, pad, label_scale_factor = transform_test_data_scalecano(
        rgb_origin, 
        intrinsic, 
        cfg.data_basic
    )
    
    with torch.no_grad():
        input_batch = rgb_input.unsqueeze(0).to(DEVICE)
        cam_models_stacks = [c.unsqueeze(0).to(DEVICE) for c in cam_models_stacks] # do_test usually returns list of tensors
        
        input_data = {
            'input': input_batch,
            'cam_model': cam_models_stacks
        }
        
        pred_depth, confidence, pred_dict = model.inference(input_data)
        
        if 'prediction_normal' in pred_dict:
            pred_normal = pred_dict['prediction_normal']
            if pred_normal.dim() == 4:
                pred_normal = pred_normal.squeeze(0)
                
            pred_normal = pred_normal.permute(1, 2, 0).cpu().numpy()
            
            # Check for NaNs in output
            if np.isnan(pred_normal).any():
                print("WARNING: NaN values detected in predicted normals!")
                pred_normal = np.nan_to_num(pred_normal, nan=0.0)

            # Unpad
            pad_h_half, pad_h_rem, pad_w_half, pad_w_rem = pad
            h_pred, w_pred, _ = pred_normal.shape
            pred_normal = pred_normal[pad_h_half : h_pred - pad_h_rem, pad_w_half : w_pred - pad_w_rem, :]
            pred_normal = cv2.resize(pred_normal, (w, h), interpolation=cv2.INTER_LINEAR)
            
            # Visualize: (-1, 1) -> (0, 255)
            pred_normal_vis = (pred_normal + 1) / 2
            pred_normal_vis = (pred_normal_vis * 255).clip(0, 255).astype(np.uint8)
            
            return cv2.cvtColor(pred_normal_vis, cv2.COLOR_RGB2BGR)
        else:
            print("Model output did not contain 'prediction_normal'. Keys found:", pred_dict.keys())
            return None

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    try:
        model, cfg = load_model()
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    print("Searching for available camera...")
    cap = open_camera()
    
    if cap is None:
        print("Error: No working camera found.")
        return

    # Warmup
    print("Camera warming up (capturing 5 frames)...")
    last_frame = None
    for i in range(5):
        ret, frame = cap.read()
        if ret:
            last_frame = frame
            time.sleep(0.5)
        else:
            print(f"Failed to capture frame {i}")
            
    cap.release()
    
    if last_frame is None:
        print("Failed to capture any valid frames.")
        return

    last_frame = ensure_frame_shape(last_frame)
    
    input_path = os.path.join(OUTPUT_DIR, "capture_input.png")
    cv2.imwrite(input_path, last_frame)
    print(f"Saved input image to {input_path}")
    
    print("Running inference...")
    result = process_image(model, cfg, last_frame)
    
    if result is not None:
        output_path = os.path.join(OUTPUT_DIR, "capture_normal.png")
        cv2.imwrite(output_path, result)
        print(f"SUCCESS: Saved surface normal estimation to {output_path}")
    else:
        print("Inference failed to produce a result.")

if __name__ == "__main__":
    main()
