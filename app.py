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
st.set_page_config(
    page_title="Object Detection",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------- Custom CSS (Modern + Responsive) ----------
st.markdown("""
    <style>
        /* Sidebar fixed width */
        section[data-testid="stSidebar"] {
            min-width: 350px !important;
            max-width: 350px !important;
        }
        
        /* Bigger and nicer Upload area */
        .stFileUploader > div {
            border: 2px dashed #555 !important;
            border-radius: 12px !important;
            padding: 2rem !important;
            background-color: #1e1e1e !important;
        }
        
        /* Make images look better */
        .stImage img {
            border-radius: 10px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.3);
        }
        
        /* History section */
        .stExpander {
            border-radius: 10px;
        }
        
        /* Button styling */
        .stButton button {
            border-radius: 8px;
        }
        
        /* Dark theme improvements */
        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 2rem;
        }
    </style>
""", unsafe_allow_html=True)

# ---------- Title ----------
st.title("🔍 Object Detection")
st.markdown("Upload images • Select model • Get instant detections")

# ---------- Sidebar ----------
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    
    model_options = {
        "Nano (fastest)": {"file": "yolov8n.pt", "desc": "⚡ Fastest, good for quick testing", "speed": "🚀 100%", "accuracy": "🎯 60%"},
        "Small": {"file": "yolov8s.pt", "desc": "⚡ Fast & balanced", "speed": "🚀 85%", "accuracy": "🎯 75%"},
        "Medium": {"file": "yolov8m.pt", "desc": "⚖️ Best balance", "speed": "🚀 70%", "accuracy": "🎯 85%"},
        "Large": {"file": "yolov8l.pt", "desc": "🎯 High accuracy", "speed": "🐢 50%", "accuracy": "🎯 92%"},
        "Extra Large (most accurate)": {"file": "yolov8x.pt", "desc": "🏆 Maximum accuracy", "speed": "🐢 30%", "accuracy": "🎯 98%"}
    }

    model_choice = st.selectbox("Select Model", list(model_options.keys()), index=0)
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
        total = len([f for f in os.listdir(BIN_DIR) if f.endswith(('.jpg','.jpeg','.png','.webp'))])
        st.metric("📊 Total Detections", total)

    if st.button("🗑️ Clear All History", type="secondary", use_container_width=True):
        if os.path.exists(BIN_DIR):
            for f in os.listdir(BIN_DIR):
                try:
                    os.remove(os.path.join(BIN_DIR, f))
                except:
                    pass
        st.success("✅ History cleared successfully!")
        st.rerun()

    st.markdown("---")
    st.caption("🔍 YOLOv8 • 80 Classes")

# ---------- Main Area ----------
uploaded_files = st.file_uploader(
    "Drop your image(s) here or click to browse",
    type=["jpg", "jpeg", "png", "webp"],
    accept_multiple_files=True,
    help="Supports JPG, PNG, WEBP • Max 200MB per file"
)

if uploaded_files:
    model_path = MODELS[model_choice]
    model = load_model(model_path)

    for uploaded_file in uploaded_files:
        with st.expander(f"📷 {uploaded_file.name}", expanded=True):
            image = Image.open(uploaded_file).convert("RGB")
            
            # Responsive columns (stack on mobile)
            col1, col2 = st.columns(2)
            
            with col1:
                st.image(image, caption="📌 Original Image", use_container_width=True)
            
            with col2:
                with st.spinner("🔄 Running YOLOv8 detection..."):
                    results = model(np.array(image), conf=conf_thresh)
                    annotated = results[0].plot()
                    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                
                st.image(annotated_rgb, caption="✅ Detected Objects", use_container_width=True)

                # Download
                save_path, filename = save_image(annotated_rgb, uploaded_file.name)
                with open(save_path, "rb") as f:
                    st.download_button(
                        label="📥 Download Detected Image",
                        data=f,
                        file_name=filename,
                        use_container_width=True
                    )

            # Detection Table
            boxes = results[0].boxes
            if len(boxes) > 0:
                data = []
                for box in boxes:
                    cls_name = results[0].names[int(box.cls)]
                    conf = box.conf[0].item()
                    data.append({"Object": cls_name, "Confidence": f"{conf:.2%}"})
                st.dataframe(pd.DataFrame(data), use_container_width=True, hide_index=True)
            else:
                st.info("No objects detected above the confidence threshold.")

# ---------- History Section ----------
st.markdown("---")
st.subheader("📜 Recent Detections")

history = get_history()

if history:
    cols = st.columns(4)   # 4 columns on desktop, automatically stacks on mobile
    for i, fname in enumerate(history):
        with cols[i % 4]:
            img_path = os.path.join(BIN_DIR, fname)
            st.image(img_path, use_container_width=True)
            st.caption(fname[:25] + "..." if len(fname) > 25 else fname)
else:
    st.info("No detection history yet. Upload some images to see them here.")
