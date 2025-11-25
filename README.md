# 1. Project Title
YOLOv8 Object Detection Web App using Streamlit
# 2. Abstract / Introduction

This project implements a real-time Object Detection System using the YOLOv8 model trained on the COCO dataset.
The application allows users to upload images and perform object detection via a Streamlit web interface, running inside a Conda environment in Visual Studio Code (VS Code).

The system detects objects in images, draws bounding boxes, and displays confidence scores. This project follows the exact execution requirements:
✔ Runs locally (not Colab)
✔ Uses Conda environment
✔ Executed & deployed using streamlit run app.py
✔ Includes all required screenshots and documentation

# 3. Dataset & YOLO Model Details (COCO)
COCO Dataset (Common Objects in Context)

Contains 80 object classes

118k training images

5k validation images

Everyday objects: person, bottle, car, laptop, etc.

YOLOv8 Model

Version: YOLOv8n/s/m/l

Pretrained on COCO

Supports real-time detection

High accuracy, low inference time

Loaded using the Ultralytics YOLO library

# 4. Environment Setup
Step 1 — Install Anaconda

Download from: https://www.anaconda.com/

Step 2 — Create Conda Environment
conda create -n yolo-env python=3.10
conda activate yolo-env

Step 3 — Install Dependencies
pip install -r requirements.txt

requirements.txt includes:
streamlit
ultralytics
opencv-python
numpy
Pillow
torch
torchvision
torchaudio

# 5. GPU Installation Steps OR CPU Installation Steps
CPU Installation (Most Common)

Run:

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install ultralytics streamlit opencv-python numpy Pillow

GPU Installation (Only for NVIDIA CUDA Systems)

Check CUDA version:

nvidia-smi


For CUDA 12.1:

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install ultralytics streamlit opencv-python numpy Pillow


If you install the CPU version of torch, you must run YOLO on CPU (device="cpu").

# 6. How to Run in VS Code using Conda
Step 1 — Open VS Code
Step 2 — Open your project folder
Step 3 — Select Conda interpreter
Ctrl + Shift + P → "Python: Select Interpreter"
Choose: yolo-env

Step 4 — Activate environment in terminal
conda activate yolo-env

Step 5 — Run the app
streamlit run app.py

# 7. How to Deploy using Streamlit

Once you run:

streamlit run app.py


Streamlit launches automatically on:

http://localhost:8501


You will see:
✔ App UI
✔ Upload section
✔ Detection output

# 8. Output Screenshots (Mandatory)

Add all screenshots inside /Screenshots/ folder.

Required screenshots:

Conda environment activation in VS Code terminal
→ Screenshots/conda_env.png

Running Streamlit command
→ Screenshots/run_streamlit.png

Streamlit UI opened in browser
→ Screenshots/ui.png

Object detection result
→ Screenshots/detection.png

## 9. Enhancements / Innovations Added

(Write the ones you implemented. Examples below.)

Added confidence score slider

Added bounding box color customization

Supports multiple image batch processing

Clean dark UI theme

Automatically saves detected results

Added CPU-safe YOLO inference (no CUDA required)

Error handling for invalid paths

## 10. Results & Conclusion

This project successfully demonstrates real-time object detection using YOLOv8 + Streamlit, running entirely in a local Conda environment inside VS Code.
The system accurately detects COCO objects and presents results through a user-friendly web interface.
## Screenshot:
<img width="1919" height="901" alt="Screenshot 2025-11-24 224811" src="https://github.com/user-attachments/assets/7e063289-9fce-417e-9eb4-84dedbda54d0" />

<img width="1761" height="319" alt="Screenshot 2025-11-24 224914" src="https://github.com/user-attachments/assets/326b34ca-71cc-4443-95b8-2922bdc1ba00" />
