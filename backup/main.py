# main.py
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from ultralytics import YOLO
import cv2
import numpy as np
import base64
import os
import webbrowser
import threading
import time

app = FastAPI(title="YOLOv8 Object Detection API")

# Enable CORS for frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the model once at startup
model = YOLO("yolov8n.pt")

# Function: Server start hone ke baad browser khud open karega
def open_browser():
    time.sleep(1.5)  # Server ko start hone ke liye thoda time dena
    webbrowser.open("http://127.0.0.1:8000")

# Startup Event: Background thread mein browser open karne wali command chalana
@app.on_event("startup")
def startup_event():
    threading.Thread(target=open_browser).start()

@app.post("/detect")
async def detect_objects(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File provided is not an image.")

    try:
        # Read image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Run YOLOv8 detection
        results = model(img)
        result = results[0]

        # Extract detections
        detections = []
        for box in result.boxes:
            class_id = int(box.cls[0].item())
            detections.append({
                "class_name": model.names[class_id],
                "confidence": round(box.conf[0].item() * 100, 2)
            })

        # Generate annotated image
        annotated_img = result.plot()
        
        # Encode image to Base64 to send to frontend
        _, buffer = cv2.imencode('.jpg', annotated_img)
        img_base64 = base64.b64encode(buffer).decode('utf-8')

        return {
            "detections": detections,
            "image_base64": f"data:image/jpeg;base64,{img_base64}"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Frontend directory ko mount karna taake API frontend ko serve kar sake
if os.path.exists("frontend"):
    app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")
else:
    print("Warning: 'frontend' folder not found. Web interface will not load properly.")