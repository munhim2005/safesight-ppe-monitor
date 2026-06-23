"""
config.py
---------
Shared constants for SafeSight: device selection, detection thresholds,
and the color map used when drawing bounding boxes on frames.
"""

import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BOX_THRESHOLD = 0.40

LABEL_COLORS = {
    "worker":              (180, 180, 180),
    "hard_hat":            (0,   200,  80),
    "vest":                (0,   180, 255),
    "glass":               (0,   220, 180),
    "mask":                (200, 200,   0),
    "glove":               (255, 220,   0),
    "boots":               (100, 100, 255),
    "ear-protection":      (50,  150, 255),
    "no-hard_hat":         (255,  50,  50),
    "no-vest":             (255,  50,  50),
    "no-glass":            (255, 100,  50),
    "no-mask":             (255, 100,  50),
    "no-glove":            (255, 100,  50),
    "no-boots":            (255, 100,  50),
    "no-ear-protection":   (255, 100,  50),
    "circular_saw":        (255,  50, 255),
    "welding_equipment":   (255, 100, 255),
    "fire_extinguisher":   (255,   0,   0),
    "fire_prevention_net": (0,   255,   0),
}
DEFAULT_COLOR = (160, 160, 160)
