import gradio as gr
import cv2
import numpy as np
import os
import time
from threading import Lock

try:
    from hailo_platform import (HEF, VDevice, HailoStreamInterface, InferVStreams, 
                                ConfigureParams, InputVStreamParams, OutputVStreamParams, FormatType)
    HAILO_AVAILABLE = True
except ImportError:
    HAILO_AVAILABLE = False
    print("Hailo Platform not installed. This script will only work on a device with Hailo software installed.")

# Global Model Cache
CURRENT_MODEL = None
CURRENT_MODEL_PATH = None

class HailoResults:
    def __init__(self, masks, orig_shape):
        self.masks = HailoMasks(masks, orig_shape)

class HailoMasks:
    def __init__(self, data, orig_shape):
        self.data = data
        self.orig_shape = orig_shape

class HailoYOLOSeg:
    def __init__(self, model_path):
        if not HAILO_AVAILABLE:
            raise ImportError("Hailo Platform not available.")
        
        self.model_path = model_path
        self.lock = Lock()
        
        # Initialize resources
        self.target = VDevice()
        self.hef = HEF(self.model_path)
        
        # Configure Network Group
        configure_params = ConfigureParams.create_from_hef(
            hef=self.hef, 
            interface=HailoStreamInterface.PCIe
        )
        self.network_groups = self.target.configure(self.hef, configure_params)
        self.network_group = self.network_groups[0]
        self.network_group_params = self.network_group.create_params()
        
        # Create VStream Params - Input (UINT8 for images) and Output (FLOAT32 for easy math)
        self.input_vstream_params = InputVStreamParams.make(
            self.network_group, 
            format_type=FormatType.UINT8
        )
        self.output_vstream_params = OutputVStreamParams.make(
            self.network_group, 
            format_type=FormatType.FLOAT32
        )
        
        # Get Info
        self.input_vstream_infos = self.hef.get_input_vstream_infos()
        self.output_vstream_infos = self.hef.get_output_vstream_infos()
        
        # Assume single input
        self.input_info = self.input_vstream_infos[0]
        self.input_shape = self.input_info.shape 
        self.input_height = self.input_shape[0]
        self.input_width = self.input_shape[1]
        
        # Initialize Pipeline Context
        self.pipeline = InferVStreams(
            self.network_group, 
            self.input_vstream_params, 
            self.output_vstream_params
        )
        # Enter the context manually to keep it open
        self.pipeline.__enter__()

    def __del__(self):
        # Clean up
        if hasattr(self, 'pipeline') and self.pipeline:
             self.pipeline.__exit__(None, None, None)

    def __call__(self, image, conf=0.25, iou=0.7):
        with self.lock:
            # Preprocess
            h, w = image.shape[:2]
            processed_img = cv2.resize(image, (self.input_width, self.input_height))
            
            # Ensure proper contiguous array layout for HailoRT
            input_tensor = np.expand_dims(processed_img, axis=0)
            input_tensor = np.ascontiguousarray(input_tensor, dtype=np.uint8)
            
            input_data = {self.input_info.name: input_tensor}
            
            # Inference
            results = self.pipeline.infer(input_data)
            
            # Post-process logic...
            return self._post_process(results, h, w, conf, iou)

    def _post_process(self, results, h, w, conf, iou):
        # Post-process
        preds = None
        protos = None
        
        for name, data in results.items():
            if data.ndim == 4:
                 if data.shape[1] == 32:
                     protos = data[0] # remove batch
            elif data.ndim == 3:
                preds = data[0] # remove batch
                
        # Fallback heuristics
        if preds is None or protos is None:
             for name, data in results.items():
                d = data[0]
                if d.size == 32 * 160 * 160: 
                    protos = d.reshape(32, 160, 160)
                else:
                    preds = d
        
        if preds is None or protos is None:
            print("Could not identify YOLOv8 output tensors.")
            return [HailoResults(None, (h, w))]
            
        if preds.shape[0] < preds.shape[1]: 
             preds = preds.T
             
        boxes = preds[:, 0:4]
        scores = preds[:, 4:84]
        mask_coeffs = preds[:, 84:]
        
        class_ids = np.argmax(scores, axis=1)
        max_scores = np.max(scores, axis=1)
        
        valid_indices = max_scores > conf
        
        boxes = boxes[valid_indices]
        scores = max_scores[valid_indices]
        class_ids = class_ids[valid_indices]
        mask_coeffs = mask_coeffs[valid_indices]
        
        if len(boxes) == 0:
            return [HailoResults(None, (h, w))]
            
        indices = cv2.dnn.NMSBoxes(
            bboxes=boxes.tolist(), 
            scores=scores.tolist(),
            score_threshold=conf,
            nms_threshold=iou
        )
        
        final_masks = []
        
        if len(indices) > 0:
            indices = indices.flatten()
            for i in indices:
                coeffs = mask_coeffs[i]
                box = boxes[i]
                
                mask = np.matmul(coeffs, protos.reshape(32, -1)).reshape(160, 160)
                mask = 1 / (1 + np.exp(-mask))
                
                full_mask = cv2.resize(mask, (self.input_width, self.input_height))
                full_mask = (full_mask > 0.5).astype(np.uint8)
                
                bx, by, bw, bh = box
                x1 = int(bx - bw/2)
                y1 = int(by - bh/2)
                x2 = int(bx + bw/2)
                y2 = int(by + bh/2)
                
                x1 = max(0, x1); y1 = max(0, y1)
                x2 = min(self.input_width, x2); y2 = min(self.input_height, y2)
                
                instance_mask = np.zeros((self.input_height, self.input_width), dtype=np.uint8)
                instance_mask[y1:y2, x1:x2] = full_mask[y1:y2, x1:x2]
                
                final_instance_mask = cv2.resize(instance_mask, (w, h), interpolation=cv2.INTER_NEAREST)
                final_masks.append(final_instance_mask)
        
        if not final_masks:
            return [HailoResults(None, (h, w))]
            
        masks_array = np.array(final_masks)
        
        class TensorMock:
            def __init__(self, array):
                self.array = array
            def cpu(self):
                return self
            def numpy(self):
                return self.array
        
        results_wrapper = HailoResults(TensorMock(masks_array), (h, w))
        return [results_wrapper]

def get_model(model_path):
    global CURRENT_MODEL, CURRENT_MODEL_PATH
    
    if model_path == CURRENT_MODEL_PATH and CURRENT_MODEL is not None:
        return CURRENT_MODEL
        
    try:
        print(f"Loading Hailo HEF from: {model_path}")
        # Clean up old model if exists
        if CURRENT_MODEL is not None:
             del CURRENT_MODEL
             CURRENT_MODEL = None
             
        model = HailoYOLOSeg(model_path)
        CURRENT_MODEL = model
        CURRENT_MODEL_PATH = model_path
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return None

def find_longest_vertical_run(mask):
    """
    Finds the longest vertical run of True values in a boolean 2D mask.
    Returns (x, start_y, end_y) or None if no run found.
    """
    h, w = mask.shape
    max_len = 0
    best_run = None
    
    # Iterate columns
    for x in range(w):
        col = mask[:, x]
        if not np.any(col):
            continue
            
        # Find runs
        padded = np.concatenate(([False], col, [False]))
        diff = np.diff(padded.astype(int))
        
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0]
        
        if len(starts) == 0:
            continue
            
        lengths = ends - starts
        best_idx = np.argmax(lengths)
        current_max = lengths[best_idx]
        
        if current_max > max_len:
            max_len = current_max
            best_run = (x, starts[best_idx], ends[best_idx])
            
    return best_run

def process_flood_surface(image, surface_map, model_path, channel_mode, invert_depth, structure_thresh, black_expansion, min_width, deep_threshold):
    """
    Process flood estimation using surface map as depth proxy, with Hailo inference.
    """
    # Check if HEF file
    if not model_path.endswith('.hef'):
        return image, None, "Please provide a valid .hef file path."

    model = get_model(model_path)
    
    if model is None:
        return image, None, f"Model not loaded. Failed to load from: {model_path}"
    
    if image is None or surface_map is None:
        return None, None, "Please provide both images."

    # SMART UPSCALING
    h1, w1 = image.shape[:2]
    h2, w2 = surface_map.shape[:2]
    
    target_w = max(w1, w2)
    target_h = max(h1, h2)
    
    if (w1, h1) != (target_w, target_h):
        image = cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
        
    if (w2, h2) != (target_w, target_h):
        surface_map = cv2.resize(surface_map, (target_w, target_h), interpolation=cv2.INTER_CUBIC)

    # Run Inference
    results = model(image)
    
    # Check for masks
    if results[0].masks.data is None:
        return image, None, "No flood detected or Inference Failed."
        
    # Aggregate Flood Mask
    masks = results[0].masks.data.cpu().numpy()
    
    if masks is None or masks.size == 0:
        return image, None, "No flood detected."

    # Convert masks to single mask
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
    abs_depths = masked_values / 255.0
    if invert_depth:
        abs_depths = 1.0 - abs_depths
        
    surface_overlap = (abs_depths <= structure_thresh) & (flood_mask == 1)
    structure_overlap = (abs_depths > structure_thresh) & (flood_mask == 1)
    
    if black_expansion > 0:
        kernel_size = int(black_expansion)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size*2+1, kernel_size*2+1))
        structure_mask_uint8 = structure_overlap.astype(np.uint8)
        dilated_structure = cv2.dilate(structure_mask_uint8, kernel, iterations=1)
        structure_overlap = (dilated_structure == 1)
        structure_overlap = structure_overlap & (flood_mask == 1)

    overlay = surface_map.copy()
    
    surface_overlap = (flood_mask == 1) & (~structure_overlap)
    
    overlay[surface_overlap] = overlay[surface_overlap] * 0.5 + np.array([0, 255, 0]) * 0.5 
    overlay[structure_overlap] = np.array([0, 0, 0])

    # Find and draw the longest vertical line
    if min_width > 1:
        kernel_width = int(min_width)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_width, 1))
        mask_uint8 = structure_overlap.astype(np.uint8)
        filtered_mask = cv2.morphologyEx(mask_uint8, cv2.MORPH_OPEN, kernel)
        search_mask = (filtered_mask > 0)
    else:
        search_mask = structure_overlap
        
    longest_run = find_longest_vertical_run(search_mask)
    
    if longest_run:
        lx, ly_start, ly_end = longest_run
        cv2.line(overlay, (lx, ly_start), (lx, ly_end - 1), (255, 0, 0), 3)
    
    # Statistics
    max_len = 0
    if longest_run:
        _, ly_start, ly_end = longest_run
        max_len = ly_end - ly_start
    
    img_h = image.shape[0]
    flood_rows = np.where(np.any(flood_mask > 0, axis=1))[0]
    
    if len(flood_rows) > 0:
        flood_height = flood_rows[-1] - flood_rows[0] + 1
    else:
        flood_height = img_h
        
    relative_height = 0.0
    if flood_height > 0:
        relative_height = max_len / flood_height
        
    if relative_height < 0.10:
        severity = "Low"
    elif relative_height < deep_threshold:
        severity = "Medium"
    else:
        severity = "High"
    
    info = (
        f"Flood Severity Analysis:\n\n"
        f"Severity: {severity.upper()}\n"
        f"Max Vertical Interaction: {max_len} px\n"
        f"Flood Vertical Span: {flood_height} px\n"
        f"Relative Depth: {relative_height*100:.1f}%\n"
    )
    
    return overlay, masked_depth, info

def launch_app():
    with gr.Blocks(title="Surface Flood Estimator (Hailo)") as demo:
        gr.Markdown("# Surface-Based Flood Depth Estimation (Hailo Accelerated)")
        gr.Markdown("Upload an image and its corresponding Surface Estimation Map. This version uses a compiled HEF model running on a Hailo accelerator.")
        
        with gr.Row():
            with gr.Column():
                input_img = gr.Image(label="Flood Image", type="numpy")
                surface_img = gr.Image(label="Surface/Depth Map (Dark=Surface)", type="numpy")
                
                gr.Markdown("### Model Configuration")
                model_path_input = gr.Textbox(
                    label="Hailo Model Path (.hef)", 
                    value='yolov8m_seg.hef',
                    placeholder="/path/to/your/model.hef"
                )
                structure_thresh = gr.Slider(minimum=0.0, maximum=1.0, value=0.05, step=0.05, label="Surface Brightness Cutoff")
                black_expansion = gr.Slider(minimum=0, maximum=50, value=7, step=1, label="Critical Area Expansion (px)")
                min_width = gr.Slider(minimum=1, maximum=20, value=3, step=1, label="Min Vertical Structure Width (px)")
                
                deep_threshold = gr.Slider(minimum=0.1, maximum=1.0, value=0.3, step=0.05, label="Deep Flood Threshold")
                
                channel = gr.Dropdown(choices=["Blue", "Green", "Red", "Grayscale"], value="Green", label="Depth Source Channel")
                invert = gr.Checkbox(label="Invert Depth Input", value=False)
                
                submit_btn = gr.Button("Estimate Flood Depth")
            
            with gr.Column():
                output_img = gr.Image(label="Estimation Result", height=400)
                masked_depth_img = gr.Image(label="Flood - Depth Only", height=400)
                stats = gr.Textbox(label="Statistics", lines=8)
        
        submit_btn.click(
            fn=process_flood_surface,
            inputs=[input_img, surface_img, model_path_input, channel, invert, structure_thresh, black_expansion, min_width, deep_threshold],
            outputs=[output_img, masked_depth_img, stats]
        )
    
    demo.launch(share=False, server_name="0.0.0.0")

if __name__ == "__main__":
    launch_app()
