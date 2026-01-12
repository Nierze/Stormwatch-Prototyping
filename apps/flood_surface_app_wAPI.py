import gradio as gr
import ultralytics
import cv2
import numpy as np
import requests
import json
from datetime import datetime

# Global Model Cache
CURRENT_MODEL = None
CURRENT_MODEL_PATH = None

def send_flood_report(api_url, api_key, severity, details):
    """
    Sends a flood report to the API.
    """
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "severity": severity,
        "timestamp": datetime.now().isoformat(),
        "details": details
    }
    
    try:
        response = requests.post(api_url, json=payload, headers=headers)
        if response.status_code == 200 or response.status_code == 201:
            return f"API Status: Success ({response.status_code})"
        else:
            return f"API Status: Failed ({response.status_code}) - {response.text[:50]}"
    except Exception as e:
        return f"API Status: Error - {str(e)}"
CURRENT_MODEL = None
CURRENT_MODEL_PATH = None

def get_model(model_path):
    global CURRENT_MODEL, CURRENT_MODEL_PATH
    
    if model_path == CURRENT_MODEL_PATH and CURRENT_MODEL is not None:
        return CURRENT_MODEL
        
    try:
        print(f"Loading model from: {model_path}")
        model = ultralytics.YOLO(model_path)
        CURRENT_MODEL = model
        CURRENT_MODEL_PATH = model_path
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None


def find_longest_vertical_run(mask):
    """
    Finds the longest vertical run of True values in a boolean 2D mask.
    Returns (x, start_y, end_y) or None if no run found.
    """
    # Simply iterate columns. For a typical image size, this is fast enough in numpy-ish ways?
    # Actually, pure python column loop with numpy ops is okay-ish.
    # But let's try a properly vectorized approach per column or just efficient logic.
    
    h, w = mask.shape
    max_len = 0
    best_run = None
    
    # We can perform a column-wise run-length encoding.
    # To do this efficiently without too many loops:
    # We can iterate columns (which is O(W)) and use np.diff for starts/ends.
    
    for x in range(w):
        col = mask[:, x]
        if not np.any(col):
            continue
            
        # Find runs
        # Pad with False to detect start/end at boundaries
        padded = np.concatenate(([False], col, [False]))
        diff = np.diff(padded.astype(int))
        
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0] # distinct from starts
        
        if len(starts) == 0:
            continue
            
        lengths = ends - starts
        best_idx = np.argmax(lengths)
        current_max = lengths[best_idx]
        
        if current_max > max_len:
            max_len = current_max
            best_run = (x, starts[best_idx], ends[best_idx])
            
    return best_run


def process_flood_surface(image, surface_map, model_path, channel_mode, invert_depth, structure_thresh, black_expansion, min_width, deep_threshold, api_key, api_url):
    """
    Process flood estimation using surface map as depth proxy.
    
    Args:
        image: Original RGB Flood Image.
        surface_map: Surface Estimation Image (Normals/Depth).
        model_path: Path to the YOLO .pt model file.
        high_threshold: Percentage (0.0-1.0) for 'High' depth cutoff.
        channel_mode: Which channel to use (Blue, Green, Red, Grayscale).
        invert_depth: If True, reverses the depth value (0 becomes 1, 1 becomes 0).
        structure_thresh: Threshold for determining structure vs surface.
        black_expansion: Dilation size for result.
        min_width: Minimum horizontal width (px) for a vertical structure to be considered for the red line.
        api_key: API Key for getting access
        api_url: URL to post to
    """
    model = get_model(model_path)
    
    if model is None:
        return image, None, f"Model not loaded. Failed to load from: {model_path}"
    
    if image is None or surface_map is None:
        return None, None, "Please provide both images."

    # SMART UPSCALING
    # "make sure both images are upscayled to same size before any inference or intersection happens"
    # We want to preserve the highest detail.
    h1, w1 = image.shape[:2]
    h2, w2 = surface_map.shape[:2]
    
    target_w = max(w1, w2)
    target_h = max(h1, h2)
    
    # Resize both if needed (forcing specific size might distort aspect ratio if they differ, 
    # but we assume they are pairs. Typically we align to the largest dimension).
    # To be safe against aspect ratio mismatch, we usually resize to the one with largest area or specific one.
    # Let's assume they are paired and just resize to the largest dimensions found.
    
    if (w1, h1) != (target_w, target_h):
        image = cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
        
    if (w2, h2) != (target_w, target_h):
        surface_map = cv2.resize(surface_map, (target_w, target_h), interpolation=cv2.INTER_CUBIC)

    # Run Inference
    results = model(image)
    
    if not results[0].masks:
        return image, None, "No flood detected."

    # Aggregate Flood Mask
    masks = results[0].masks.data.cpu().numpy()
    if masks.shape[1:] != image.shape[:2]:
        full_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        for m in masks:
            m_resized = cv2.resize(m, (image.shape[1], image.shape[0]))
            full_mask = np.maximum(full_mask, m_resized)
    else:
        full_mask = np.max(masks, axis=0)

    # Threshold binary mask
    flood_mask = (full_mask > 0.5).astype(np.uint8)
    
    # Create Masked Depth View
    masked_depth = cv2.bitwise_and(surface_map, surface_map, mask=flood_mask)

    # Extract Depth Proxy Channel
    if channel_mode == "Blue":
        raw_values = surface_map[:, :, 0]
    elif channel_mode == "Green":
        raw_values = surface_map[:, :, 1]
    elif channel_mode == "Red":
        raw_values = surface_map[:, :, 2]
    else: # Grayscale
        raw_values = cv2.cvtColor(surface_map, cv2.COLOR_BGR2GRAY)
    
    # Mask to flood area only
    masked_values = raw_values.astype(np.float32)
    
    # INTERSECTION VISUALIZATION
    # "If the flood mask intersects with the [non-surface], I want to see the color change"
    
    # 1. Normalize
    abs_depths = masked_values / 255.0
    if invert_depth:
        abs_depths = 1.0 - abs_depths
        
    # 2. Define Intersection Areas
    # Zone 1: Flood on Surface (Ground) -> safe/low -> Green
    # Zone 2: Flood on Non-Surface (Intersection with Structure) -> critical -> Red
    
    surface_overlap = (abs_depths <= structure_thresh) & (flood_mask == 1)
    structure_overlap = (abs_depths > structure_thresh) & (flood_mask == 1)
    
    # Zones
    
    # Expand Critical Area (Dilation)
    # "for every pureblack intersected area, I want the neighboring n pixel to be black"
    if black_expansion > 0:
        kernel_size = int(black_expansion)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size*2+1, kernel_size*2+1))
        # Dilate the boolean mask (convert to uint8 first)
        structure_mask_uint8 = structure_overlap.astype(np.uint8)
        dilated_structure = cv2.dilate(structure_mask_uint8, kernel, iterations=1)
        # Update the boolean mask
        structure_overlap = (dilated_structure == 1)
        # Note: This might expand OUTSIDE the flood mask. 
        # User said "neighboring n pixel to be black".
        # If we want it constrained to flood, we'd AND it with flood_mask.
        # But usually "expand danger zone" implies the danger is bigger.
        # Let's constrain it to the Flood Mask? 
        # "I want the neighboring n pixel to be black" - logically if the flood is there.
        # If the flood is NOT there, we shouldn't paint it black (it's not flood).
        # So we constrain to flood_mask.
        structure_overlap = structure_overlap & (flood_mask == 1)

    # Apply Overlay
    # "place the flood segmentation mask on the depth map for the results"
    overlay = surface_map.copy()
    
    # Green for Surface Overlap (Flood on Ground)
    # We update surface_overlap to exclude the new expanded black area
    surface_overlap = (flood_mask == 1) & (~structure_overlap)
    
    overlay[surface_overlap] = overlay[surface_overlap] * 0.5 + np.array([0, 255, 0]) * 0.5 
    
    # Black for Structure Overlap (Flood on Structure)
    overlay[structure_overlap] = np.array([0, 0, 0])

    # Find and draw the longest vertical line
    # structure_overlap is the boolean mask of black areas
    
    # Filter by minimum width if needed
    if min_width > 1:
        # Create a horizontal kernel
        kernel_width = int(min_width)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_width, 1))
        # Convert to uint8 for morphology
        mask_uint8 = structure_overlap.astype(np.uint8)
        # Morphological Opening (Erosion followed by Dilation) removes small foreground objects
        # In this case, horizontal runs shorter than kernel_width will be removed.
        filtered_mask = cv2.morphologyEx(mask_uint8, cv2.MORPH_OPEN, kernel)
        search_mask = (filtered_mask > 0)
    else:
        search_mask = structure_overlap
        
    longest_run = find_longest_vertical_run(search_mask)
    
    if longest_run:
        lx, ly_start, ly_end = longest_run
        # Draw red line
        # Note: ly_end is exclusive index from diff, but for drawing we want the pixel coordinates.
        # So we draw from (lx, ly_start) to (lx, ly_end - 1)
        # color=(0, 0, 255) because usually OpenCV is BGR. Wait. 
        # The app reads "Original RGB Flood Image" via gr.Image(type="numpy").
        # Gradio returns RGB.
        # So Red is (255, 0, 0).
        cv2.line(overlay, (lx, ly_start), (lx, ly_end - 1), (255, 0, 0), 3)
    
    # Statistics
    max_len = 0
    if longest_run:
        _, ly_start, ly_end = longest_run
        max_len = ly_end - ly_start
    
    
    img_h = image.shape[0]
    
    # Calculate flood mask height
    # Find rows where flood exists
    flood_rows = np.where(np.any(flood_mask > 0, axis=1))[0]
    
    if len(flood_rows) > 0:
        flood_height = flood_rows[-1] - flood_rows[0] + 1
    else:
        flood_height = img_h # Fallback if no flood or empty
        
    relative_height = 0.0
    if flood_height > 0:
        relative_height = max_len / flood_height
        
    # Severity Heuristics
    if relative_height < 0.10:
        severity = "Low"
    elif relative_height < deep_threshold:
        severity = "Medium"
    else:
        severity = "High"
    
    # API Call
    api_status = send_flood_report(api_url, api_key, severity, {"max_vertical": int(max_len), "relative_height": float(relative_height)})
    
    info = (
        f"Flood Severity Analysis:\n\n"
        f"Severity: {severity.upper()}\n"
        f"Max Vertical Interaction: {max_len} px\n"
        f"Flood Vertical Span: {flood_height} px\n"
        f"Relative Depth: {relative_height*100:.1f}%\n"
        f"(Length of red line relative to flood height)\n\n"
        f"{api_status}"
    )
    
    return overlay, masked_depth, info

def launch_app():
    with gr.Blocks(title="Surface Flood Estimator") as demo:
        gr.Markdown("# Surface-Based Flood Depth Estimation")
        gr.Markdown("Upload an image and its corresponding Surface Estimation Map. The app estimates flood severity based on how much 'Non-Surface' component (walls, obstacles) is covered by water.")
        
        with gr.Row():
            with gr.Column():
                input_img = gr.Image(label="Flood Image", type="numpy")
                surface_img = gr.Image(label="Surface/Depth Map (Dark=Surface)", type="numpy")
                
                gr.Markdown("### Model Configuration")
                model_path_input = gr.Textbox(
                    label="YOLO Model Path", 
                    value=''
                )
                structure_thresh = gr.Slider(minimum=0.0, maximum=1.0, value=0.05, step=0.05, label="Surface Brightness Cutoff (Pixels darker than this are Ground)")
                black_expansion = gr.Slider(minimum=0, maximum=50, value=7, step=1, label="Critical Area Expansion (px)")
                min_width = gr.Slider(minimum=1, maximum=20, value=3, step=1, label="Min Vertical Structure Width (px)")
                
                deep_threshold = gr.Slider(minimum=0.1, maximum=1.0, value=0.3, step=0.05, label="Deep Flood Threshold (Relative Height)")
                
                channel = gr.Dropdown(choices=["Blue", "Green", "Red", "Grayscale"], value="Green", label="Depth Source Channel")
                invert = gr.Checkbox(label="Invert Depth Input", value=False)
                
                gr.Markdown("### API Configuration")
                api_key_input = gr.Textbox(label="API Key", value="sk_1768192573271_2bhobz0gvp7")
                api_url_input = gr.Textbox(label="API URL", value="https://www.stormwatch.app/docs")
                
                submit_btn = gr.Button("Estimate Flood & Send API Report")
            
            with gr.Column():
                output_img = gr.Image(label="Estimation Result", height=400)
                masked_depth_img = gr.Image(label="Flood - Depth Only", height=400)
                stats = gr.Textbox(label="Statistics & API Status", lines=10)
        
        submit_btn.click(
            fn=process_flood_surface,
            inputs=[input_img, surface_img, model_path_input, channel, invert, structure_thresh, black_expansion, min_width, deep_threshold, api_key_input, api_url_input],
            outputs=[output_img, masked_depth_img, stats]
        )
    
    demo.launch(share=False, server_name="0.0.0.0")

if __name__ == "__main__":
    launch_app()
