import streamlit as st
from ultralytics import YOLO
import os
from PIL import Image

# ---------------------------------------------------------
# FORCE CPU MODE (Fix for “Torch not compiled with CUDA”)
# ---------------------------------------------------------
DEVICE = "cpu"   # Do NOT change this unless you have GPU + CUDA PyTorch

# ---------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------
st.set_page_config(page_title="YOLO Object Detection", layout="wide")
st.title("🔍 YOLOv8 Object Detection App")

st.write("Upload an image and perform object detection using YOLOv8.")

# ---------------------------------------------------------
# Load model
# ---------------------------------------------------------
model_path = st.text_input("Enter YOLO model path:", "yolo11m.pt")

if not os.path.exists(model_path):
    st.error("Model file not found. Place the YOLO weights (.pt) file in the project folder.")
else:
    model = YOLO(model_path)
    model.to(DEVICE)

# ---------------------------------------------------------
# File uploader
# ---------------------------------------------------------
uploaded_img = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_img is not None:
    image = Image.open(uploaded_img)
    st.image(image, caption="Uploaded Image", width=550)

    if st.button("Start Inference"):
        try:
            st.write("Running inference...")

            # Run YOLO inference on CPU
            results = model.predict(image, device=DEVICE)

            # Save output
            output_img_path = "output.jpg"
            results[0].save(output_img_path)

            st.success("Inference completed successfully!")
            st.image(output_img_path, caption="Detected Output", width=550)

        except Exception as e:
            st.error(f"Error during inference: {str(e)}")
