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

# ---------- Custom CSS for Responsive & Beautiful Interface ----------
st.markdown("""
    <style>
        /* GLOBAL STYLES */
        .stApp {
            background: linear-gradient(135deg, #f5f7fc 0%, #e9eef5 100%);
        }
        
        /* Main content container */
        .main .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        
        /* Cards for upload & results */
        .upload-card, .detection-card {
            background: white;
            border-radius: 24px;
            padding: 1.5rem;
            margin-bottom: 1.5rem;
            box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.05), 0 8px 10px -6px rgba(0, 0, 0, 0.02);
            transition: transform 0.2s ease;
        }
        
        /* Sidebar responsiveness */
        section[data-testid="stSidebar"] {
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(8px);
            border-right: 1px solid rgba(0,0,0,0.05);
        }
        
        @media (max-width: 768px) {
            section[data-testid="stSidebar"] {
                min-width: 100% !important;
                max-width: 100% !important;
                width: 100% !important;
                position: fixed;
                z-index: 1000;
                height: auto;
                border-radius: 0 0 20px 20px;
                box-shadow: 0 4px 12px rgba(0,0,0,0.1);
            }
            /* Push main content down when sidebar is expanded on mobile */
            .stApp > header {
                background: transparent;
            }
        }
        
        /* UPLOAD AREA - Beautiful styling */
        .stFileUploader > div {
            border: 2px dashed #cbd5e1 !important;
            border-radius: 32px !important;
            padding: 3rem 1.5rem !important;
            background: linear-gradient(145deg, #ffffff, #f8fafc) !important;
            transition: all 0.3s ease;
            cursor: pointer;
        }
        .stFileUploader > div:hover {
            border-color: #3b82f6 !important;
            background: #ffffff !important;
            box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.1);
        }
        /* Upload icon and text */
        .stFileUploader label {
            font-size: 1.2rem !important;
            font-weight: 600 !important;
            color: #1e293b !important;
        }
        .stFileUploader div[data-testid="stMarkdownContainer"] p {
            font-size: 0.95rem;
            color: #64748b;
        }
        
        /* Responsive image columns: stack on small screens */
        @media (max-width: 640px) {
            .stColumn {
                width: 100% !important;
                flex: auto !important;
            }
            .stColumns {
                flex-direction: column;
            }
        }
        
        /* History gallery responsive grid */
        .history-grid {
            display: flex;
            flex-wrap: wrap;
            gap: 1rem;
            margin-top: 1rem;
        }
        .history-item {
            flex: 1 1 calc(25% - 1rem);
            min-width: 140px;
            background: white;
            border-radius: 20px;
            padding: 0.75rem;
            box-shadow: 0 4px 12px rgba(0,0,0,0.05);
            transition: all 0.2s;
        }
        .history-item:hover {
            transform: translateY(-3px);
            box-shadow: 0 12px 20px -8px rgba(0,0,0,0.15);
        }
        @media (max-width: 768px) {
            .history-item {
                flex: 1 1 calc(50% - 1rem);
            }
        }
        @media (max-width: 480px) {
            .history-item {
                flex: 1 1 100%;
            }
        }
        
        /* Buttons & Metrics improvements */
        .stButton button {
            border-radius: 40px !important;
            font-weight: 500 !important;
            transition: 0.2s;
        }
        .stMetric {
            background: #f1f5f9;
            border-radius: 20px;
            padding: 0.5rem;
            text-align: center;
        }
        
        /* Better expander style */
        .streamlit-expanderHeader {
            background: white;
            border-radius: 20px;
            font-weight: 600;
            border: 1px solid #e2e8f0;
        }
        
        /* Footer */
        .footer {
            text-align: center;
            margin-top: 2rem;
            padding: 1rem;
            color: #64748b;
            font-size: 0.8rem;
        }
    </style>
""", unsafe_allow_html=True)

# ---------- Title with branding ----------
st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <h1 style="font-size: 2.5rem; font-weight: 700; background: linear-gradient(135deg, #2563eb, #7c3aed); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">🔍 Object Detection</h1>
        <p style="color: #475569;">Upload images, choose YOLOv8 model, and detect objects instantly</p>
    </div>
""", unsafe_allow_html=True)

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
    st.caption("Higher = fewer but more confident detections")

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
        st.success("✅ History cleared!")
        st.rerun()

    st.markdown("---")
    st.caption("🔍 YOLOv8 • 80 Classes")

# ---------- Main Area - Upload (Beautiful Card) ----------
with st.container():
    st.markdown('<div class="upload-card">', unsafe_allow_html=True)
    uploaded_files = st.file_uploader(
        "📤 Drag & drop or click to upload images",
        type=["jpg", "jpeg", "png", "webp"],
        accept_multiple_files=True,
        help="Supports JPG, PNG, WEBP • Multiple files allowed • Max 25MB each"
    )
    st.markdown('<p style="text-align: center; color: #94a3b8; margin-top: -0.5rem;">✨ High-quality object detection with YOLOv8</p>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ---------- Detection Processing ----------
if uploaded_files:
    model_path = MODELS[model_choice]
    model = load_model(model_path)

    for uploaded_file in uploaded_files:
        with st.expander(f"📷 {uploaded_file.name}", expanded=True):
            image = Image.open(uploaded_file).convert("RGB")
            
            # Two columns – responsive (stack automatically via CSS)
            cols = st.columns(2)
            
            # Original Image
            with cols[0]:
                st.image(image, caption="📌 Original Image", use_container_width=True)
            
            # Detected Image
            with cols[1]:
                with st.spinner("🔄 Detecting objects with YOLOv8..."):
                    results = model(np.array(image), conf=conf_thresh)
                    annotated = results[0].plot()
                    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                
                st.image(annotated_rgb, caption="✅ Detected Objects", use_container_width=True)

                # Download Button
                save_path, filename = save_image(annotated_rgb, uploaded_file.name)
                with open(save_path, "rb") as f:
                    st.download_button(
                        "📥 Download Detected Image",
                        f,
                        file_name=filename,
                        use_container_width=True
                    )

            # Detection Results Table
            boxes = results[0].boxes
            if len(boxes) > 0:
                data = []
                for box in boxes:
                    cls_name = results[0].names[int(box.cls)]
                    conf_val = box.conf[0].item()
                    data.append({"Object": cls_name, "Confidence": f"{conf_val:.2%}"})
                st.dataframe(pd.DataFrame(data), use_container_width=True, hide_index=True)
            else:
                st.info("🔍 No objects detected above confidence threshold. Try lowering the threshold or use a larger model.")

# ---------- History Section (Responsive Grid) ----------
st.markdown("---")
st.subheader("📜 Recent Detections")

history = get_history()

if history:
    # Build a responsive gallery without columns limitations
    st.markdown('<div class="history-grid">', unsafe_allow_html=True)
    for fname in history:
        img_path = os.path.join(BIN_DIR, fname)
        st.markdown(f'''
            <div class="history-item">
                <img src="data:image/png;base64,{__import__('base64').b64encode(open(img_path, "rb").read()).decode()}" style="width:100%; border-radius:16px; margin-bottom:8px;">
                <div style="font-size:0.75rem; color:#475569; text-align:center;">{fname[:20]}{"..." if len(fname)>20 else ""}</div>
            </div>
        ''', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
else:
    st.info("📭 No detections yet. Upload images above to see your history here.")

# Footer
st.markdown('<div class="footer">Powered by YOLOv8 • Streamlit • Responsive design for all devices</div>', unsafe_allow_html=True)
