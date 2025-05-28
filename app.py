import gradio as gr
import numpy as np
import tensorflow as tf
import cv2
from PIL import Image
from pathlib import Path
import os

# Suppress TensorFlow warnings and configure CPU threads
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.config.threading.set_inter_op_parallelism_threads(2)
tf.config.threading.set_intra_op_parallelism_threads(2)

# Load model
try:
    model = tf.keras.models.load_model("deepfake_model.h5")
    model.compile(optimizer='adam', 
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
except Exception as e:
    raise gr.Error(f"Model loading failed: {str(e)}")

IMG_SIZE = 128
MAX_FRAMES = 20

# Predict for images
def predict_image(image_path):
    try:
        img = Image.open(image_path)
        img = img.resize((IMG_SIZE, IMG_SIZE))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        pred = model.predict(img_array, verbose=0)[0][0]
        return {"prediction": "Fake" if pred <= 0.5 else "Real"}
    except Exception as e:
        raise gr.Error(f"Image processing error: {str(e)}")

# Predict for videos
def predict_video(video_path):
    try:
        cap = cv2.VideoCapture(video_path)
        fake_votes = 0
        total_frames = 0

        while total_frames < MAX_FRAMES:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = frame.astype("float32") / 255.0
            frame = np.expand_dims(frame, axis=0)

            pred = model.predict(frame, verbose=0)[0][0]
            if pred <= 0.5:
                fake_votes += 1
            total_frames += 1

        cap.release()
        ratio = fake_votes / total_frames if total_frames > 0 else 0
        return {"prediction": "Fake" if ratio > 0.5 else "Real"}
    except Exception as e:
        raise gr.Error(f"Video processing error: {str(e)}")

# Unified handler for image/video
def process_file(file_path):
    try:
        if not file_path:
            return "", "No file uploaded"

        file_path = str(file_path)
        ext = Path(file_path).suffix.lower()

        if ext in ['.jpg', '.jpeg', '.png']:
            result = predict_image(file_path)
        elif ext in ['.mp4', '.avi', '.mov']:
            result = predict_video(file_path)
        else:
            return "", f"Unsupported file type: {ext}"

        return result["prediction"], ""
    except Exception as e:
        return "", str(e)

# Gradio interface
with gr.Blocks(title="Deepfake Detector") as interface:
    gr.Markdown("## Deepfake Detection")
    gr.Markdown("Upload an image (JPEG/PNG) or video (MP4/AVI/MOV)")

    with gr.Row():
        with gr.Column():
            file_input = gr.File(
                label="Upload Media",
                file_types=["image", "video"],
                type="filepath"
            )
            submit_btn = gr.Button("Analyze", variant="primary")

        with gr.Column():
            output_text = gr.Textbox(label="Prediction")
            error_output = gr.Textbox(visible=False, label="Error")

    submit_btn.click(
        fn=process_file,
        inputs=file_input,
        outputs=[output_text, error_output],
    )

interface.launch(
    show_error=True,
    server_name="0.0.0.0",
    server_port=7860
)
