import cv2
import torch
import numpy as np
import os
import sys
import time
from mmengine import Config

# Ensure proper path for imports
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)
METRIC3D_DIR = os.path.join(PROJECT_ROOT, "models", "Metric3D")
if METRIC3D_DIR not in sys.path:
    sys.path.append(METRIC3D_DIR)

from mono.model.monodepth_model import get_configured_monodepth_model

# --- Constants & Configuration (MATCHING apps/metric3d_surface_normal_app.py) ---
MODEL_TYPE = "ViT-Large"  # Changed from Small to Large to match working app
CHECKPOINT_URL = "https://huggingface.co/JUGGHM/Metric3D/resolve/main/metric_depth_vit_large_800k.pth"
CHECKPOINT_FILENAME = "metric_depth_vit_large_800k.pth"
CHECKPOINT_DIR = os.path.join(os.path.expanduser("~"), ".cache", "torch", "hub", "checkpoints")
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, CHECKPOINT_FILENAME)
CONFIG_PATH = os.path.join(METRIC3D_DIR, "mono/configs/HourglassDecoder/vit.raft5.large.py")

DEVICE = "cpu"
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "captured_images")

# --- Helper Functions from mono/utils/do_test_cpu.py (Self-contained to ensure parity) ---
def build_camera_model(H : int, W : int, intrinsics : list) -> np.array:
    fx, fy, u0, v0 = intrinsics
    f = (fx + fy) / 2.0
    x_row = np.arange(0, W).astype(np.float32)
    x_row_center_norm = (x_row - u0) / W
    x_center = np.tile(x_row_center_norm, (H, 1)) # [H, W]

    y_col = np.arange(0, H).astype(np.float32) 
    y_col_center_norm = (y_col - v0) / H
    y_center = np.tile(y_col_center_norm, (W, 1)).T # [H, W]

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

    image = cv2.resize(image, dsize=(reshape_w, reshape_h), interpolation=cv2.INTER_LINEAR)
    image = cv2.copyMakeBorder(
        image, pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half, 
        cv2.BORDER_CONSTANT, value=padding)
    
    intrinsic[2] = intrinsic[2] * to_scale_ratio
    intrinsic[3] = intrinsic[3] * to_scale_ratio

    cam_model = build_camera_model(reshape_h, reshape_w, intrinsic)
    cam_model = cv2.copyMakeBorder(
        cam_model, pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half, 
        cv2.BORDER_CONSTANT, value=-1)

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

    rgb, cam_model, pad, resize_label_scale_ratio = resize_for_input(rgb, forward_size, canonical_intrinsic, [ori_h, ori_w], 1.0)
    label_scale_factor = cano_label_scale_ratio * resize_label_scale_ratio

    rgb = torch.from_numpy(rgb.transpose((2, 0, 1))).float()
    rgb = torch.div((rgb - mean), std)
    
    # Clamp inputs (prevent NaNs on CPU)
    rgb = torch.clamp(rgb, min=-10.0, max=10.0)
    
    cam_model = torch.from_numpy(cam_model.transpose((2, 0, 1))).float()
    cam_model = cam_model[None, :, :, :]
    cam_model_stacks = [
        torch.nn.functional.interpolate(cam_model, size=(cam_model.shape[2]//i, cam_model.shape[3]//i), mode='bilinear', align_corners=False)
        for i in [2, 4, 8, 16, 32]
    ]
    return rgb, cam_model_stacks, pad, label_scale_factor


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
    
    # Force load from CPU
    print(f"Loading checkpoint from {CHECKPOINT_PATH}")
    # We load state dict manually to be safe on CPU
    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu")
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint

    print("Building model...")
    cfg.model.cudnn_benchmark = False
    cfg.model.pretrained = None
    model = get_configured_monodepth_model(cfg)
    
    model.load_state_dict(state_dict, strict=False)
    model = model.cpu().eval()
    return model, cfg

def ensure_frame_shape(frame):
    print(f"DEBUG: Input frame shape: {frame.shape}")
    if len(frame.shape) == 3 and frame.shape[0] == 1 and frame.shape[1] > 10000:
        flat_size = frame.shape[1]
        resolutions = [(1280, 720), (1920, 1080), (640, 480), (800, 600), (1024, 768), (1536, 864)]
        for w_cand, h_cand in resolutions:
            if w_cand * h_cand == flat_size:
                print(f"DEBUG: Reshaping flattened frame to ({h_cand}, {w_cand}, 3)")
                return frame.reshape((h_cand, w_cand, 3))
    
    if len(frame.shape) == 2 and frame.shape[0] == 1 and frame.shape[1] > 30000:
         flat_size = frame.shape[1] // 3
         resolutions = [(1280, 720), (1920, 1080), (640, 480), (800, 600), (1024, 768), (1536, 864)]
         for w_cand, h_cand in resolutions:
            if w_cand * h_cand == flat_size:
                 print(f"DEBUG: Reshaping flattened 2D frame to ({h_cand}, {w_cand}, 3)")
                 return frame.reshape((h_cand, w_cand, 3))
    return frame

def open_camera():
    indices = [0, 1, -1]
    backends = [cv2.CAP_ANY, cv2.CAP_V4L2]
    # Quick fix: prioritize known working index/backend from logs if available
    # For now, stick to robust search
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
    rgb_origin = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    if np.all(rgb_origin == 0):
        print("WARNING: Input image is completely black!")
    
    h, w = rgb_origin.shape[:2]
    intrinsic = [1000.0, 1000.0, w / 2, h / 2]

    rgb_input, cam_models_stacks, pad, label_scale_factor = transform_test_data_scalecano(
        rgb_origin, intrinsic, cfg.data_basic
    )
    
    with torch.no_grad():
        input_batch = rgb_input.unsqueeze(0).to(DEVICE)
        cam_models_stacks = [c.unsqueeze(0).to(DEVICE) for c in cam_models_stacks]
        
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
            
            if np.isnan(pred_normal).any():
                print("WARNING: NaN values detected in predicted normals!")
                pred_normal = np.nan_to_num(pred_normal, nan=0.0)

            pad_h_half, pad_h_rem, pad_w_half, pad_w_rem = pad
            h_pred, w_pred, _ = pred_normal.shape
            pred_normal = pred_normal[pad_h_half : h_pred - pad_h_rem, pad_w_half : w_pred - pad_w_rem, :]
            pred_normal = cv2.resize(pred_normal, (w, h), interpolation=cv2.INTER_LINEAR)
            
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
    if last_frame is None:
        print("Failed to reshape frame (unknown format).")
        return
        
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
