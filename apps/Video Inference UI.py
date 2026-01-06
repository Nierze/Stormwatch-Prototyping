import gradio as gr
from ultralytics import YOLO
import cv2

# Cache loaded models
model_cache = {}
# Global flag for stopping stream
stop_flag = {"stop": False}

def stream_predict(model_path: str, video_path: str, conf: float = 0.25, imgsz: int = 640, classes_text: str = ""):
    # Reset stop flag
    stop_flag["stop"] = False

    # Load or fetch cached model
    if model_path not in model_cache:
        model_cache[model_path] = YOLO(model_path)
    model = model_cache[model_path]

    # Parse classes
    classes = None
    if classes_text.strip():
        try:
            classes = [int(x.strip()) for x in classes_text.split(",") if x.strip().isdigit()]
        except ValueError:
            raise ValueError("Classes must be comma-separated integers (e.g. '0,2,5').")

    # Stream results
    for results in model.predict(source=video_path, stream=True, conf=conf, imgsz=imgsz, classes=classes):
        if stop_flag["stop"]:
            break
        frame = results.plot()  # annotated frame
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        yield frame  # yield to Gradio

def stop_stream():
    stop_flag["stop"] = True
    return None  # clear output frame


with gr.Blocks() as demo:
    gr.Markdown("## YOLO Video Streaming (Frame-by-Frame with Stop Button)")

    with gr.Row():
        model_path = gr.Textbox(label="Path to YOLO model (.pt)")

    with gr.Row():
        input_vid = gr.Video(label="Upload a Video")
        output_frames = gr.Image(label="Streamed Frames (updates live)")

    with gr.Accordion("Advanced", open=False):
        conf = gr.Slider(0.0, 1.0, value=0.25, step=0.01, label="Confidence threshold")
        imgsz = gr.Slider(320, 1280, value=640, step=32, label="Image size (square)")
        classes_text = gr.Textbox(value="", label="Classes (comma-separated). Blank = all")

    with gr.Row():
        run_btn = gr.Button("Start Streaming")
        stop_btn = gr.Button("Stop Streaming")

    run_btn.click(
        fn=stream_predict,
        inputs=[model_path, input_vid, conf, imgsz, classes_text],
        outputs=output_frames,
    )

    stop_btn.click(
        fn=stop_stream,
        inputs=[],
        outputs=output_frames,
    )

if __name__ == "__main__":
    demo.launch()
