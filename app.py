import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os
from datetime import datetime
from ultralytics import YOLO
import pandas as pd

# ---------- Settings ----------
BIN_DIR = "bin"
os.makedirs(BIN_DIR, exist_ok=True)

# Model mapping - Yeh sahi rahega
MODELS = {
    "Nano (fastest)": "yolov8n.pt",
    "Small": "yolov8s.pt",
    "Medium": "yolov8m.pt",
    "Large": "yolov8l.pt",
    "Extra Large (most accurate)": "yolov8x.pt"
}

@st.cache_resource
def load_model(model_path):
    return YOLO(model_path)

def save_image(img_rgb, original_name):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"detected_{timestamp}_{original_name}"
    path = os.path.join(BIN_DIR, filename)
    Image.fromarray(img_rgb).save(path)
    return path, filename

def get_history():
    if not os.path.exists(BIN_DIR):
        return []
    files = [f for f in os.listdir(BIN_DIR) if f.endswith(('.jpg','.jpeg','.png','.webp'))]
    files.sort(key=lambda x: os.path.getmtime(os.path.join(BIN_DIR, x)), reverse=True)
    return files[:12]

# ---------- Page Config ----------
st.set_page_config(page_title="Object Detection", page_icon="🔍", layout="wide")
st.title("🔍 Object Detection")
st.markdown("Upload images, select model, click each image to detect")

# ---------- Sidebar ----------
with st.sidebar:
    st.markdown("""
        <style>
            [data-testid="stSidebar"] {
                min-width: 350px;
                max-width: 350px;
            }
        </style>
    """, unsafe_allow_html=True)
    
    st.markdown("### ⚙️ Settings")
   
    # Model options with descriptions
    model_options = {
        "Nano (fastest)": {
            "file": "yolov8n.pt",
            "desc": "⚡⚡Fastest, least accurate",
            "speed": "🚀 100%",
            "accuracy": "🎯 60%"
        },
        "Small": {
            "file": "yolov8s.pt",
            "desc": "⚡ Fast & balanced",
            "speed": "🚀 85%",
            "accuracy": "🎯 75%"
        },
        "Medium": {
            "file": "yolov8m.pt",
            "desc": "⚖️ Good balance",
            "speed": "🚀 70%",
            "accuracy": "🎯 85%"
        },
        "Large": {
            "file": "yolov8l.pt",
            "desc": "🎯 High accuracy",
            "speed": "🐢 50%",
            "accuracy": "🎯 92%"
        },
        "Extra Large (most accurate)": {
            "file": "yolov8x.pt",
            "desc": "🏆 Maximum accuracy",
            "speed": "🐢 30%",
            "accuracy": "🎯 98%"
        }
    }
   
    model_choice = st.selectbox("Model", list(model_options.keys()), index=0)
    selected = model_options[model_choice]
    
    st.caption(selected["desc"])
    col1, col2 = st.columns(2)
    with col1:
        st.metric("⚡ Speed", selected["speed"])
    with col2:
        st.metric("🎯 Accuracy", selected["accuracy"])
   
    conf_thresh = st.slider("Confidence Threshold", 0.0, 1.0, 0.25, 0.01)
    st.caption("Higher = fewer but more accurate detections")
   
    if os.path.exists(BIN_DIR):
        total_images = len([f for f in os.listdir(BIN_DIR) if f.endswith(('.jpg','.jpeg','.png','.webp'))])
        st.metric("📊 Total Detections", total_images)
   
    if st.button("🗑️ Clear All History", type="secondary", use_container_width=True):
        if os.path.exists(BIN_DIR):
            for f in os.listdir(BIN_DIR):
                try:
                    os.remove(os.path.join(BIN_DIR, f))
                except:
                    pass
        st.success("✅ History cleared!")
        st.rerun()
   
    st.markdown("---")
    st.caption("🔍 YOLOv8 | 80 Classes")

# ---------- Main Area ----------
uploaded_files = st.file_uploader(
    "Choose image(s) - multiple allowed",
    type=["jpg", "jpeg", "png", "webp"],
    accept_multiple_files=True
)

if uploaded_files:
    # ==================== FIXED LINE ====================
    model_path = MODELS[model_choice]                    # ← Yeh sahi tarika hai
    model = load_model(model_path)
    # ===================================================

    for uploaded_file in uploaded_files:
        with st.expander(f"📷 {uploaded_file.name}", expanded=True):
            col1, col2 = st.columns(2)
           
            # Original image
            image = Image.open(uploaded_file)
            col1.image(image, caption="Original", use_container_width=True)
           
            # Detection
            with st.spinner("Detecting..."):
                results = model(np.array(image), conf=conf_thresh)
                annotated = results[0].plot()
                annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
           
            col2.image(annotated_rgb, caption="Detected", use_container_width=True)
           
            # Save and download
            save_path, filename = save_image(annotated_rgb, uploaded_file.name)
            with open(save_path, "rb") as f:
                col2.download_button("📥 Download Result", f, file_name=filename, use_container_width=True)
           
            # Detection table
            boxes = results[0].boxes
            if len(boxes) > 0:
                data = []
                for box in boxes:
                    cls_name = results[0].names[int(box.cls)]
                    conf = box.conf[0].item()
                    data.append({"Object": cls_name, "Confidence": f"{conf:.2%}"})
                st.dataframe(pd.DataFrame(data), use_container_width=True)
            else:
                st.info("No objects detected")

# ---------- History Section ----------
st.markdown("---")
st.markdown("### 📜 Recent Detections")
history = get_history()
if history:
    cols = st.columns(4)
    for i, fname in enumerate(history):
        with cols[i % 4]:
            img_path = os.path.join(BIN_DIR, fname)
            st.image(img_path, use_container_width=True)
            st.caption(fname[:25] + "..." if len(fname) > 25 else fname)
else:
    st.info("No history yet. Upload images to see them here.")
