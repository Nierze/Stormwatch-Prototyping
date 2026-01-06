import gradio as gr
import ultralytics
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the YOLO model
# Assuming the path exists as verified in earlier steps
MODEL_PATH = '/home/nirze/PycharmProjects/Stormwatch Prototyping/models/GENERAL MODEL 4/runs/segment/train/weights/best.pt'
try:
    model = ultralytics.YOLO(MODEL_PATH)
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

def get_depth_level(hue_value):
    """
    Map Hue value to Depth Level.
    Assumption based on user input:
    - Red (~0-20, 160-180) is Near (Low Depth Impact/Distance)
    - Purple (~270) is Far (High Depth Impact/Distance)
    
    We will map Hue 0 -> 270 to a scalar 0 -> 1.
    Note: OpenCV Hue is 0-179.
    Red: 0 or 179.
    Purple: ~130-150? (Violet is ~270 deg -> 135 in OpenCV)
    
    Let's use a simpler heuristic:
    Dist = Hue Value (if we treat Red as 0 and Purple as 'Max').
    But Red wraps around.
    
    Let's analyze the Blue channel instead, as reasoned:
    Red=(255,0,0), Purple=(128,0,128).
    Blue channel: Low -> Near, High -> Far.
    This is robust if the heatmap transitions Red->...->Purple without other high-blue colors (like pure Blue).
    If it's a standard heatmap (Magma/Inferno?), it usually goes Black->Red->Yellow->White? NO.
    User said Red->Purple. Maybe Turbo or Jet?
    Jet: Blue->Cyan->Green->Yellow->Red. (Inverse?)
    User: "redder part is, the nearer... farther it is, the purplish".
    This sounds like Red->Blue/Purple spectral.
    
    Let's rely on Blue Channel intensity for "Farness".
    Normalized Blue (0-255) / 255.0.
    """
    # Using Blue channel intensity as proxy for "Farness" (Purple has high Blue, Red has low Blue)
    return hue_value # Placeholder, logic in process function

def process_flood_depth(image, depth_heatmap, high_threshold):
    if model is None:
        return image, "Model not loaded."
    
    if image is None or depth_heatmap is None:
        return None, "Please provide both images."

    # Resize depth heatmap to match image
    if image.shape[:2] != depth_heatmap.shape[:2]:
        depth_heatmap = cv2.resize(depth_heatmap, (image.shape[1], image.shape[0]))

    # Run Inference
    results = model(image)
    
    if not results[0].masks:
        return image, "No flood detected."

    # Aggregate Flood Mask
    # masks.data is (N, H, W). Max projection to combine multiple objects.
    masks = results[0].masks.data.cpu().numpy()
    if masks.shape[1:] != image.shape[:2]:
         # Resize masks to original image string if needed (YOLO sometimes resizes)
         # But usually ultralytics handles this if we use specialized methods. 
         # Manual resize:
         full_mask = np.zeros(image.shape[:2], dtype=np.uint8)
         for m in masks:
             m_resized = cv2.resize(m, (image.shape[1], image.shape[0]))
             full_mask = np.maximum(full_mask, m_resized)
    else:
        full_mask = np.max(masks, axis=0)

    # Threshold binary mask
    flood_mask = (full_mask > 0.5).astype(np.uint8)

    # Depth Logic
    # Extract Blue Channel from Depth Heatmap (BGR in OpenCV)
    # B is channel 0
    blue_channel = depth_heatmap[:, :, 0]
    
    # Mask the blue channel with flood area
    # only consider flood pixels
    flood_depths = blue_channel.astype(np.float32) / 255.0
    
    # Create Output Overlay
    overlay = image.copy()
    
    # Colors (BGR)
    # Low (Near/Red -> Low Blue): Green (0, 255, 0)
    # Medium: Yellow (0, 255, 255)
    # High (Far/Purple -> High Blue): Red (0, 0, 255)
    
    # We classify each pixel
    # Dynamic Thresholds based on user input
    # high_threshold comes from slider (e.g. 0.8)
    # low_threshold scaled proportionally (e.g. 0.4)
    low_threshold = high_threshold / 2.0
    
    low_mask = (flood_depths < low_threshold) & (flood_mask == 1)
    med_mask = (flood_depths >= low_threshold) & (flood_depths < high_threshold) & (flood_mask == 1)
    high_mask = (flood_depths >= high_threshold) & (flood_mask == 1)
    
    # Apply colors
    overlay[low_mask] = overlay[low_mask] * 0.5 + np.array([0, 255, 0]) * 0.5 # Green
    overlay[med_mask] = overlay[med_mask] * 0.5 + np.array([0, 255, 255]) * 0.5 # Yellow
    overlay[high_mask] = overlay[high_mask] * 0.5 + np.array([0, 0, 255]) * 0.5 # Red/Orange
    
    # Add count/info text
    low_pct = np.sum(low_mask) / np.sum(flood_mask) * 100 if np.sum(flood_mask) > 0 else 0
    med_pct = np.sum(med_mask) / np.sum(flood_mask) * 100 if np.sum(flood_mask) > 0 else 0
    high_pct = np.sum(high_mask) / np.sum(flood_mask) * 100 if np.sum(flood_mask) > 0 else 0
    
    info = f"Flood Depth Distribution:\nLow: {low_pct:.1f}%\nMedium: {med_pct:.1f}%\nHigh: {high_pct:.1f}%"
    
    return overlay, info

def launch_app():
    iface = gr.Interface(
        fn=process_flood_depth,
        inputs=[
            gr.Image(label="Flood Image", type="numpy"), 
            gr.Image(label="Depth Heatmap", type="numpy"),
            gr.Slider(minimum=0.5, maximum=0.95, value=0.8, step=0.05, label="High Depth Threshold (Higher = More Lenient)")
        ],
        outputs=[
            gr.Image(label="Flood Depth Estimation"),
            gr.Textbox(label="Statistics")
        ],
        title="Flood Depth Estimation",
        description="Upload a flood image and its corresponding depth heatmap (Red=Near, Purple=Far). Adjust the Leniency slider to control how strict better rating is."
    )
    iface.launch(share=False, server_name="0.0.0.0")

if __name__ == "__main__":
    launch_app()
