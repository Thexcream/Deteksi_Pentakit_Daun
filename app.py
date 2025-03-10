import gradio as gr
from ultralytics import YOLO
import cv2
import tempfile

# Load model
model = YOLO("best.onnx", task="detect")  # Ganti dengan best.pt jika perlu

def detect_objects(
    video, 
    frame_skip=10, 
    resize_dim=640,
    conf_threshold=0.5
):
    # Baca video
    cap = cv2.VideoCapture(video)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # File output sementara
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp:
        output_path = temp.name

    # Inisialisasi video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Proses setiap N frame
        if frame_count % frame_skip == 0:
            # Resize untuk percepatan
            resized_frame = cv2.resize(frame, (resize_dim, resize_dim))
            
            # Prediksi dengan confidence threshold
            results = model(
                resized_frame,
                conf=conf_threshold  # Filter confidence
            )
            
            # Resize kembali ke ukuran asli
            annotated_frame = cv2.resize(results[0].plot(), (width, height))
            out.write(annotated_frame)
        else:
            out.write(frame)
            
        frame_count += 1

    cap.release()
    out.release()
    return output_path

# Interface Gradio
iface = gr.Interface(
    fn=detect_objects,
    inputs=[
        gr.Video(label="Input Video"),
        gr.Slider(1, 30, 10, label="Frame Skip"),
        gr.Slider(320, 1280, 640, 32, label="Resize Dimensi"),
        gr.Slider(0.0, 1.0, 0.5, 0.05, label="Confidence Threshold")
    ],
    outputs=gr.Video(label="Output Video"),
    title="YOLOv8 Video Detection",
    description="Deteksi penyakit tanaman tomat dengan YOLOv8"
)

if __name__ == "__main__":
    iface.launch()
