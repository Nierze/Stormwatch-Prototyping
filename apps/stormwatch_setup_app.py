import gradio as gr
import cv2
import torch
import numpy as np
import os
import sys
import time
import ultralytics
from PIL import Image

# --- Path Setup ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)
METRIC3D_DIR = os.path.join(PROJECT_ROOT, "models", "Metric3D")
if METRIC3D_DIR not in sys.path:
    sys.path.append(METRIC3D_DIR)

# --- Imports from Metric3D ---
from mono.model.monodepth_model import get_configured_monodepth_model
try:
    from mmcv.utils import Config
except:
    from mmengine import Config

# --- Constants ---
GROUND_TRUTH_DIR = os.path.join(PROJECT_ROOT, "Ground truth")
GT_FILENAME = "surface_normal_gt.png"
YOLO_MODEL_PATH = os.path.join(PROJECT_ROOT, "deployment models", "best.pt")

DEVICE = "cpu"

# --- Globals ---
metric3d_model = None
metric3d_cfg = None
yolo_model = None
current_yolo_path = None

# =================================================================================================
# METRIC3D LOGIC (Adapted from apps/metric3d_single_capture.py)
# =================================================================================================

CHECKPOINT_URL = "https://huggingface.co/JUGGHM/Metric3D/resolve/main/metric_depth_vit_large_800k.pth"
CHECKPOINT_FILENAME = "metric_depth_vit_large_800k.pth"
CHECKPOINT_DIR = os.path.join(os.path.expanduser("~"), ".cache", "torch", "hub", "checkpoints")
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, CHECKPOINT_FILENAME)
CONFIG_PATH = os.path.join(METRIC3D_DIR, "mono/configs/HourglassDecoder/vit.raft5.large.py")

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

def load_metric3d_model():
    global metric3d_model, metric3d_cfg
    if metric3d_model is not None:
        return metric3d_model, metric3d_cfg
        
    ensure_checkpoint_exists()
    print(f"Loading configuration from {CONFIG_PATH}")
    cfg = Config.fromfile(CONFIG_PATH)
    
    print(f"Loading checkpoint from {CHECKPOINT_PATH}")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu")
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint

    print("Building Metric3D model...")
    cfg.model.cudnn_benchmark = False
    cfg.model.pretrained = None
    model = get_configured_monodepth_model(cfg)
    
    model.load_state_dict(state_dict, strict=False)
    model = model.cpu().eval()
    
    metric3d_model = model
    metric3d_cfg = cfg
    return model, cfg

# --- Preprocessing Helpers ---
def build_camera_model(H : int, W : int, intrinsics : list) -> np.array:
    fx, fy, u0, v0 = intrinsics
    f = (fx + fy) / 2.0
    x_row = np.arange(0, W).astype(np.float32)
    x_row_center_norm = (x_row - u0) / W
    x_center = np.tile(x_row_center_norm, (H, 1))
    y_col = np.arange(0, H).astype(np.float32) 
    y_col_center_norm = (y_col - v0) / H
    y_center = np.tile(y_col_center_norm, (W, 1)).T
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
    image = cv2.copyMakeBorder(image, pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half, cv2.BORDER_CONSTANT, value=padding)
    intrinsic[2] = intrinsic[2] * to_scale_ratio
    intrinsic[3] = intrinsic[3] * to_scale_ratio
    cam_model = build_camera_model(reshape_h, reshape_w, intrinsic)
    cam_model = cv2.copyMakeBorder(cam_model, pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half, cv2.BORDER_CONSTANT, value=-1)
    return image, cam_model, [pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half], 1/to_scale_ratio

def transform_test_data_scalecano(rgb, intrinsic, data_basic):
    canonical_space = data_basic['canonical_space']
    forward_size = data_basic.crop_size
    mean = torch.tensor([123.675, 116.28, 103.53]).float()[:, None, None]
    std = torch.tensor([58.395, 57.12, 57.375]).float()[:, None, None]
    ori_h, ori_w, _ = rgb.shape
    ori_focal = (intrinsic[0] + intrinsic[1]) / 2
    canonical_focal = canonical_space['focal_length']
    cano_label_scale_ratio = canonical_focal / ori_focal
    canonical_intrinsic = [intrinsic[0] * cano_label_scale_ratio, intrinsic[1] * cano_label_scale_ratio, intrinsic[2], intrinsic[3]]
    rgb, cam_model, pad, resize_label_scale_ratio = resize_for_input(rgb, forward_size, canonical_intrinsic, [ori_h, ori_w], 1.0)
    label_scale_factor = cano_label_scale_ratio * resize_label_scale_ratio
    rgb = torch.from_numpy(rgb.transpose((2, 0, 1))).float()
    rgb = torch.div((rgb - mean), std)
    rgb = torch.clamp(rgb, min=-10.0, max=10.0) # Clamp for CPU safety
    cam_model = torch.from_numpy(cam_model.transpose((2, 0, 1))).float()
    cam_model = cam_model[None, :, :, :]
    cam_model_stacks = [torch.nn.functional.interpolate(cam_model, size=(cam_model.shape[2]//i, cam_model.shape[3]//i), mode='bilinear', align_corners=False) for i in [2, 4, 8, 16, 32]]
    return rgb, cam_model_stacks, pad, label_scale_factor

def ensure_frame_shape(frame):
    if len(frame.shape) == 3 and frame.shape[0] == 1 and frame.shape[1] > 10000:
        flat_size = frame.shape[1]
        resolutions = [(1280, 720), (1920, 1080), (640, 480), (800, 600), (1024, 768), (1536, 864)]
        for w_cand, h_cand in resolutions:
            if w_cand * h_cand == flat_size:
                return frame.reshape((h_cand, w_cand, 3))
    if len(frame.shape) == 2 and frame.shape[0] == 1 and frame.shape[1] > 30000:
         flat_size = frame.shape[1] // 3
         resolutions = [(1280, 720), (1920, 1080), (640, 480), (800, 600), (1024, 768), (1536, 864)]
         for w_cand, h_cand in resolutions:
            if w_cand * h_cand == flat_size:
                 return frame.reshape((h_cand, w_cand, 3))
    return frame

def open_camera():
    indices = [0, 1, -1]
    backends = [cv2.CAP_ANY, cv2.CAP_V4L2]
    for backend in backends:
        for index in indices:
            cap = cv2.VideoCapture(index, backend)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    cap.release()
                    cap = cv2.VideoCapture(index, backend)
                    return cap
                else:
                    cap.release()
    return None

def capture_image_from_camera():
    cap = open_camera()
    if cap is None:
        return None, "Error: No camera found."
        
    print("Camera warming up...")
    last_frame = None
    for i in range(5):
        ret, frame = cap.read()
        if ret:
            last_frame = frame
            time.sleep(0.2)
            
    cap.release()
    
    if last_frame is None:
        return None, "Error: Failed to capture frames."
        
    last_frame = ensure_frame_shape(last_frame)
    # FLIP 180 degrees (as requested)
    last_frame = cv2.flip(last_frame, -1)
    
    return last_frame, "Success"

def run_metric3d_inference(frame):
    model, cfg = load_metric3d_model()
    rgb_origin = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    h, w = rgb_origin.shape[:2]
    intrinsic = [1000.0, 1000.0, w / 2, h / 2]

    rgb_input, cam_models_stacks, pad, _ = transform_test_data_scalecano(rgb_origin, intrinsic, cfg.data_basic)
    
    with torch.no_grad():
        input_batch = rgb_input.unsqueeze(0).to(DEVICE)
        cam_models_stacks = [c.unsqueeze(0).to(DEVICE) for c in cam_models_stacks]
        input_data = {'input': input_batch, 'cam_model': cam_models_stacks}
        
        _, _, pred_dict = model.inference(input_data)
        
        if 'prediction_normal' in pred_dict:
            pred_normal = pred_dict['prediction_normal']
            if pred_normal.dim() == 4: pred_normal = pred_normal.squeeze(0)
            pred_normal = pred_normal.permute(1, 2, 0).cpu().numpy()
            if np.isnan(pred_normal).any():
                pred_normal = np.nan_to_num(pred_normal, nan=0.0)

            pad_h_half, pad_h_rem, pad_w_half, pad_w_rem = pad
            h_pred, w_pred, _ = pred_normal.shape
            pred_normal = pred_normal[pad_h_half : h_pred - pad_h_rem, pad_w_half : w_pred - pad_w_rem, :]
            pred_normal = cv2.resize(pred_normal, (w, h), interpolation=cv2.INTER_LINEAR)
            
            pred_normal_vis = (pred_normal + 1) / 2
            pred_normal_vis = (pred_normal_vis * 255).clip(0, 255).astype(np.uint8)
            return cv2.cvtColor(pred_normal_vis, cv2.COLOR_RGB2BGR)
    return None

# =================================================================================================
# FLOOD ESTIMATION LOGIC (Adapted from apps/surface_flood_app.py)
# =================================================================================================

def load_yolo_model(model_path):
    global yolo_model, current_yolo_path
    if model_path == current_yolo_path and yolo_model is not None:
        return yolo_model
    try:
        print(f"Loading YOLO model from: {model_path}")
        model = ultralytics.YOLO(model_path)
        yolo_model = model
        current_yolo_path = model_path
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def find_longest_vertical_run(mask):
    h, w = mask.shape
    max_len = 0
    best_run = None
    for x in range(w):
        col = mask[:, x]
        if not np.any(col): continue
        padded = np.concatenate(([False], col, [False]))
        diff = np.diff(padded.astype(int))
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0]
        if len(starts) == 0: continue
        lengths = ends - starts
        best_idx = np.argmax(lengths)
        current_max = lengths[best_idx]
        if current_max > max_len:
            max_len = current_max
            best_run = (x, starts[best_idx], ends[best_idx])
    return best_run

def process_flood_surface(image, surface_map, model_path, channel_mode, invert_depth, structure_thresh, black_expansion, min_width, deep_threshold):
    model = load_yolo_model(model_path)
    if model is None: return image, None, f"Model not loaded: {model_path}"
    if image is None or surface_map is None: return None, None, "Missing images."

    h1, w1 = image.shape[:2]
    h2, w2 = surface_map.shape[:2]
    target_w = max(w1, w2)
    target_h = max(h1, h2)
    
    if (w1, h1) != (target_w, target_h):
        image = cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
    if (w2, h2) != (target_w, target_h):
        surface_map = cv2.resize(surface_map, (target_w, target_h), interpolation=cv2.INTER_CUBIC)

    results = model(image)
    if not results[0].masks: return image, None, "No flood detected."

    masks = results[0].masks.data.cpu().numpy()
    if masks.shape[1:] != image.shape[:2]:
        full_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        for m in masks:
            m_resized = cv2.resize(m, (image.shape[1], image.shape[0]))
            full_mask = np.maximum(full_mask, m_resized)
    else:
        full_mask = np.max(masks, axis=0)
    
    flood_mask = (full_mask > 0.5).astype(np.uint8)
    masked_depth = cv2.bitwise_and(surface_map, surface_map, mask=flood_mask)

    if channel_mode == "Blue": raw_values = surface_map[:, :, 0]
    elif channel_mode == "Green": raw_values = surface_map[:, :, 1]
    elif channel_mode == "Red": raw_values = surface_map[:, :, 2]
    else: raw_values = cv2.cvtColor(surface_map, cv2.COLOR_BGR2GRAY)
    
    masked_values = raw_values.astype(np.float32)
    abs_depths = masked_values / 255.0
    if invert_depth: abs_depths = 1.0 - abs_depths
    
    structure_overlap = (abs_depths > structure_thresh) & (flood_mask == 1)
    
    if black_expansion > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (int(black_expansion)*2+1, int(black_expansion)*2+1))
        dilated_structure = cv2.dilate(structure_overlap.astype(np.uint8), kernel, iterations=1)
        structure_overlap = (dilated_structure == 1) & (flood_mask == 1)

    overlay = surface_map.copy()
    surface_overlap = (flood_mask == 1) & (~structure_overlap)
    overlay[surface_overlap] = overlay[surface_overlap] * 0.5 + np.array([0, 255, 0]) * 0.5
    overlay[structure_overlap] = np.array([0, 0, 0])

    if min_width > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (int(min_width), 1))
        filtered_mask = cv2.morphologyEx(structure_overlap.astype(np.uint8), cv2.MORPH_OPEN, kernel)
        search_mask = (filtered_mask > 0)
    else:
        search_mask = structure_overlap
        
    longest_run = find_longest_vertical_run(search_mask)
    max_len = 0
    if longest_run:
        lx, ly_start, ly_end = longest_run
        cv2.line(overlay, (lx, ly_start), (lx, ly_end - 1), (255, 0, 0), 3)
        max_len = ly_end - ly_start

    flood_rows = np.where(np.any(flood_mask > 0, axis=1))[0]
    flood_height = (flood_rows[-1] - flood_rows[0] + 1) if len(flood_rows) > 0 else image.shape[0]
    relative_height = max_len / flood_height if flood_height > 0 else 0.0

    if relative_height < 0.10: severity = "Low"
    elif relative_height < deep_threshold: severity = "Medium"
    else: severity = "High"

    info = f"Severity: {severity}\nMax Vertical: {max_len}px\nRelative: {relative_height*100:.1f}%"
    
    # Needs to be RGB for Gradio
    overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
    masked_depth_rgb = cv2.cvtColor(masked_depth, cv2.COLOR_BGR2RGB)
    
    return overlay_rgb, masked_depth_rgb, info


# =================================================================================================
# GRADIO INTERFACE FUNCTIONS
# =================================================================================================

def setup_ground_truth_action():
    frame, msg = capture_image_from_camera()
    if frame is None:
        return None, msg

    print("Running Metric3D Ground Truth Estimation...")
    gt_normal = run_metric3d_inference(frame)
    if gt_normal is None:
        return None, "Inference Failed"
        
    # Save to Ground Truth folder
    os.makedirs(GROUND_TRUTH_DIR, exist_ok=True)
    gt_path = os.path.join(GROUND_TRUTH_DIR, GT_FILENAME)
    cv2.imwrite(gt_path, gt_normal)
    print(f"Saved Ground Truth to {gt_path}")
    
    # Convert BGR to RGB for Gradio
    gt_normal_rgb = cv2.cvtColor(gt_normal, cv2.COLOR_BGR2RGB)
    return gt_normal_rgb, "Success: Ground Truth Saved"

def estimate_flood_action(channel, invert, str_thresh, black_exp, min_w, deep_thresh):
    # 1. Capture Live Image
    frame, msg = capture_image_from_camera()
    if frame is None:
        return None, None, f"Camera Error: {msg}"
    
    # 2. Check for Ground Truth
    gt_path = os.path.join(GROUND_TRUTH_DIR, GT_FILENAME)
    if not os.path.exists(gt_path):
        return None, None, "Error: No Ground Truth found. Run Setup first."
        
    gt_normal = cv2.imread(gt_path)
    if gt_normal is None:
        return None, None, "Error: Could not read Ground Truth file."
        
    # 3. Run Flood Estimation
    # Frame is already flipped from capture function.
    # Frame is BGR from cv2, process_flood_surface expects RGB usually?
    # surface_flood_app.py uses gr.Image(type="numpy") which is RGB.
    # So we should convert frame to RGB before passing.
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # GT is read as BGR by cv2.imread. process_flood_surface does logic on it.
    # surface_flood_app.py passed numpy image from Gradio (RGB).
    # But Metric3D output was visualized as RGB.
    # If process_flood_surface logic (channels) depends on RGB/BGR...
    # It has "channel_mode" selector. "Blue", "Green", "Red". 
    # If we pass BGR, "Blue" is channel 0. 
    # If we pass RGB, "Blue" is channel 2.
    # The existing app code: 
    #    if channel_mode == "Blue": raw_values = surface_map[:, :, 0]
    # This implies channel 0 is Blue. 
    # However Gradio loads RGB. So channel 0 is Red.
    # WAIT. `surface_flood_app.py` logic:
    #    if channel_mode == "Blue": raw_values = surface_map[:, :, 0]
    # If input is RGB, [:,:,0] is Red.
    # So in the original app, "Blue" matches Red channel? That's confusing.
    # Actually, Metric3D surface normals are typically: R=X, G=Y, B=Z or similar.
    # Normals map: 
    #   (x, y, z) -> (+1, +1)/2 * 255.
    #   Usually we care about the "Green" channel (Y-axis, up/down?).
    # let's stick to passing RGB images to `process_flood_surface` to match original app behavior.
    # If verify original app had RGB input: `gr.Image(type="numpy")` -> RGB.
    # So I should convert GT to RGB too.
    gt_normal_rgb = cv2.cvtColor(gt_normal, cv2.COLOR_BGR2RGB)
    
    overlay, masked_depth, info = process_flood_surface(
        frame_rgb, gt_normal_rgb, YOLO_MODEL_PATH, channel, invert, str_thresh, black_exp, min_w, deep_thresh
    )
    
    return overlay, masked_depth, info

# --- GUI ---
def launch_app():
    with gr.Blocks(title="Stormwatch Setup") as demo:
        gr.Markdown("# Stormwatch Setup & Flood Estimation")
        
        with gr.Tabs():
            # TAB 1: Setup Ground Truth
            with gr.Tab("Setup Surface Ground Truth"):
                gr.Markdown("Capture a baseline image to generate the Surface Normal Ground Truth.")
                setup_btn = gr.Button("Setup Surface Ground Truth", variant="primary")
                gt_output = gr.Image(label="Ground Truth Result (Saved)", type="numpy")
                status_txt = gr.Textbox(label="Status")
                
                setup_btn.click(setup_ground_truth_action, outputs=[gt_output, status_txt])
                
            # TAB 2: Estimate Flood
            with gr.Tab("Estimate Flood"):
                gr.Markdown("Capture a live image and estimate flood severity using the saved Ground Truth.")
                
                with gr.Accordion("Flood Configuration", open=False):
                    channel = gr.Dropdown(choices=["Blue", "Green", "Red", "Grayscale"], value="Green", label="Depth Source Channel")
                    invert = gr.Checkbox(label="Invert Depth Input", value=False)
                    str_thresh = gr.Slider(minimum=0.0, maximum=1.0, value=0.05, step=0.05, label="Structure Threshold")
                    black_exp = gr.Slider(minimum=0, maximum=50, value=7, step=1, label="Black Expansion (px)")
                    min_w = gr.Slider(minimum=1, maximum=20, value=3, step=1, label="Min Width (px)")
                    deep_thresh = gr.Slider(minimum=0.1, maximum=1.0, value=0.3, step=0.05, label="Deep Threshold")
                
                est_btn = gr.Button("Estimate Flood", variant="primary")
                
                with gr.Row():
                    out_overlay = gr.Image(label="Flood Overlay")
                    out_masked = gr.Image(label="Masked Depth")
                
                out_info = gr.Textbox(label="Analysis Info")
                
                est_btn.click(
                    estimate_flood_action, 
                    inputs=[channel, invert, str_thresh, black_exp, min_w, deep_thresh],
                    outputs=[out_overlay, out_masked, out_info]
                )

    demo.launch(server_name="0.0.0.0")

if __name__ == "__main__":
    launch_app()
