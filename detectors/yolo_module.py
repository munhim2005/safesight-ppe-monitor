"""
vision_module.py
----------------
Handles the YOLOv8 PPE detection model.
Separated from the main app for easier retraining/swapping of weights.
"""

from ultralytics import YOLO
from pathlib import Path
from huggingface_hub import hf_hub_download
import torch
from PIL import Image, ImageDraw
import cv2
from config import BOX_THRESHOLD

# Set device once for the module
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class VisionModule:
    def __init__(self, repo_id="killuminati1/construction-ppe-yolov8", filename="best.pt"):
        print(f"[Vision] Loading YOLOv8 from {repo_id}...")

        weights_path = hf_hub_download(repo_id=repo_id, filename=filename)
        self.model = YOLO(weights_path).to(DEVICE)
        
        # 1. Map 'helmet' to 'hard_hat' directly in the names dictionary.
        # This ensures result.plot() automatically draws the correct label text.
        self.names = self.model.names
        for k, v in list(self.names.items()):
            if v.lower() == 'helmet':
                self.names[k] = 'hard_hat'
            elif v.lower() == 'no-helmet':
                self.names[k] = 'no-hard_hat'
                
        print(f"[Vision] Model loaded on {DEVICE}")

    def get_detections(self, pil_img):
        results_list = self.model(pil_img, conf=BOX_THRESHOLD)  # returns list of Results
        result = results_list[0]            # single image inference

        # If there are detections
        if result.boxes is not None and len(result.boxes) > 0:
            
            # 2. Find all positive classes currently detected in this frame
            detected_positives = set()
            for cls in result.boxes.cls:
                label = self.names[int(cls)].lower()
                if not label.startswith('no-'):
                    detected_positives.add(label)

            # 3. Filter out negative classes (e.g., 'no-boots') if the positive ('boots') exists
            keep_indices = []
            for i, cls in enumerate(result.boxes.cls):
                label = self.names[int(cls)].lower()
                
                if label.startswith('no-'):
                    positive_version = label[3:]  # removes 'no-' (e.g., 'no-boots' -> 'boots')
                    if positive_version in detected_positives:
                        continue  # Skip adding this to the keep list
                        
                keep_indices.append(i)

            # Apply the filter to the result boxes so overlapping bad boxes aren't drawn
            result.boxes = result.boxes[keep_indices]

            # Annotate the image using the filtered boxes
            annotated_img = result.plot()  # returns NumPy array (BGR)

            # Convert boxes to list of dicts for MLP
            detections = []
            for box, conf, cls in zip(result.boxes.xyxy, result.boxes.conf, result.boxes.cls):
                x0, y0, x1, y1 = box.tolist()
                detections.append({
                    "label": self.names[int(cls)].lower(),
                    "confidence": float(conf),
                    "box": [x0, y0, x1, y1]
                })

            # Convert annotated image to PIL (for Gradio)
            from PIL import Image
            annotated_img = Image.fromarray(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB))

            return annotated_img, detections
        else:
            # No detections
            return pil_img, []