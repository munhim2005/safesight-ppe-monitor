"""
models.py
---------
Loads every model SafeSight depends on exactly once, at import time:
  - YOLOv8 PPE detector       (yolo_module.VisionModule)
  - CLIP scene/activity validator (clip_validator.CLIPValidator)
  - Decision tree severity classifier (decision_tree.DecisionTree)

Other modules should import the already-loaded instances from here
rather than constructing their own.
"""

from config import DEVICE
from detectors.yolo_module import VisionModule
from detectors.clip_validator import CLIPValidator
from severity.decision_tree import DecisionTree
from detectors.pose_module import PoseModule

vision_sys = VisionModule()
pose_sys = PoseModule()

print("Loading Custom NumPy Decision Tree...")
severity_model = DecisionTree(max_depth=8, min_samples_split=5)

try:
    severity_model.load("dt_model.npz")
    print("NumPy Decision Tree loaded successfully.")
except FileNotFoundError:
    print("\n[!] CRITICAL ERROR: 'dt_model.npz' not found. Run decision_tree.py first.\n")

clip_validator = CLIPValidator(device=DEVICE)
