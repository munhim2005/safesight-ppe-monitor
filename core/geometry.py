"""
geometry.py
-----------
Spatial helper functions: bounding-box overlap math and label-to-color
lookup used when annotating frames.
"""

from config import LABEL_COLORS, DEFAULT_COLOR


def pick_color(label: str):
    return LABEL_COLORS.get(label.lower(), DEFAULT_COLOR)


def calculate_ioa(ppe_box, person_box):
    """
    Intersection over Area (of the PPE box) — how much of the PPE
    bounding box overlaps the worker's bounding box. Used to decide
    whether a detected PPE item is actually being worn by this worker
    (as opposed to lying nearby, or belonging to a different worker).
    """
    x_left = max(ppe_box[0], person_box[0])
    y_top = max(ppe_box[1], person_box[1])
    x_right = min(ppe_box[2], person_box[2])
    y_bottom = min(ppe_box[3], person_box[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    ppe_area = (ppe_box[2] - ppe_box[0]) * (ppe_box[3] - ppe_box[1])

    if ppe_area == 0:
        return 0.0

    return intersection_area / ppe_area
