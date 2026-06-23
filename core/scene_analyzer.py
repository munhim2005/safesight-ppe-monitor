"""
scene_analyzer.py
------------------
The core decision-making logic of SafeSight.

Takes YOLO's raw detections for a single frame, figures out which PPE
items belong to which worker (via IoA overlap), fuses that with CLIP's
independent scene read, runs the severity classifier, and produces a
structured scene report ready for display or video summarization.
"""

from datetime import datetime

import numpy as np

from core.geometry import calculate_ioa
from models import severity_model, clip_validator, pose_sys
from detectors.clip_validator import ACTIVITY_PPE_RULES, UNIVERSAL_REQUIRED, get_scene_activity_and_ppe

SEV_VALS = {"NONE": 0, "LOW": 1, "MEDIUM": 2, "HIGH": 3, "STOP_WORK": 4}

# Order the severity model's feature vector expects.
SEVERITY_FEATURE_ORDER = [
    "hard_hat", "vest", "glass", "glove", "boots", "ear_prot", "mask", 
    "using_tool", "using_hot_work", "fire_safety", "legs_visible", "torso_visible",
]

# PPE / equipment that sits on the ground or workbench rather than on
# the worker's body — these can never overlap a person's bounding box,
# so they're attributed to every worker in frame rather than matched
# by IoA.
ENVIRONMENT_OR_TOOL_LABELS = {
    "circular_saw", "welding_equipment", "fire_extinguisher", "fire_prevention_net"
}

# Minimum fraction of a PPE box that must overlap a worker's box for
# that PPE item to count as "worn by this worker".
PPE_OVERLAP_THRESHOLD = 0.40


def _match_ppe_to_worker(person: dict, ppes: list) -> list:
    """Returns the list of PPE/equipment labels associated with one worker."""
    worn_labels = []
    for ppe in ppes:
        label = ppe["label"].lower()
        if label in ENVIRONMENT_OR_TOOL_LABELS:
            worn_labels.append(label)
        elif calculate_ioa(ppe["box"], person["box"]) > PPE_OVERLAP_THRESHOLD:
            worn_labels.append(label)
    return worn_labels


def _crop_worker(person: dict, original_pil_image, pad_fraction: float = 0.25):
    """Crops the worker out of the full frame with a small padding margin."""
    w, h = original_pil_image.size
    x0, y0, x1, y1 = person["box"]
    pad_x, pad_y = int((x1 - x0) * pad_fraction), int((y1 - y0) * pad_fraction)
    crop_box = (
        max(0, int(x0 - pad_x)), max(0, int(y0 - pad_y)),
        min(w, int(x1 + pad_x)), min(h, int(y1 + pad_y)),
    )
    return original_pil_image.crop(crop_box)


def _classify_severity(worker_json: dict) -> str:
    """Runs the decision-tree severity classifier on one worker's fused PPE data."""
    ordered_features = [worker_json[key] for key in SEVERITY_FEATURE_ORDER]
    input_array = np.array(ordered_features, dtype=float)
    return severity_model.predict(input_array)


def _build_worker_entry(worker_id: str, worn_labels: list, worker_json: dict, visibility: dict) -> tuple:
    """
    Runs severity classification + violation text generation for one worker.
    Returns (worker_dict, violation_dict_or_None, severity_label).
    """
    legs_visible = visibility.get("legs_visible", True)
    torso_visible = visibility.get("torso_visible", True)

    severity_label = _classify_severity(worker_json)
    is_violation = severity_label != "NONE"

    activity = worker_json.get("activity", "unknown")
    missing_required = worker_json.get("missing_required", [])
    unconfirmed = []

    # Manually check baseline gear in case CLIP's activity rules didn't catch it
    if worker_json.get("hard_hat", 0.0) == 0.0:
        if "hard_hat" not in missing_required:
            missing_required.append("hard_hat")
        else:
            unconfirmed.append("hard_hat")

    if worker_json.get("boots", 0.0) == 0.0:
        if "boots" not in missing_required:
            missing_required.append("boots")
    if "boots" in missing_required and not legs_visible:
        missing_required.remove("boots")
        if "boots" not in unconfirmed:
            unconfirmed.append("boots")

    # Vest/glove/mask/glass are torso/hand/face-region items — same gating.
    for ppe_key in ("vest", "glove", "mask", "glass", "ear_prot"):
        if ppe_key in missing_required and not torso_visible:
            missing_required.remove(ppe_key)
            unconfirmed.append(ppe_key)

    if missing_required:
        is_violation = True
        formatted_missing = [item.replace("_", " ").title() for item in missing_required]
        desc_text = f"[{activity.upper()}] Missing: {', '.join(formatted_missing)}"

        # Elevate severity if required hot-work gear is missing
        if activity != "light_labor" and severity_label in ["NONE", "LOW", "MEDIUM"]:
            severity_label = "HIGH"
    elif unconfirmed:
        formatted_unconfirmed = [item.replace("_", " ").title() for item in unconfirmed]
        desc_text = f"[{activity.upper()}] Unconfirmed (partially out of frame): {', '.join(formatted_unconfirmed)}"
    else:
        desc_text = f"[{activity.upper()}] Compliant"

    activity_rules = ACTIVITY_PPE_RULES.get(activity, {})
    activity_req = activity_rules.get("required", [])
    activity_opt = activity_rules.get("optional", [])
    all_required = list(set(UNIVERSAL_REQUIRED + activity_req))

    confirmed_ppe = [
        k for k in all_required + activity_opt
        if worker_json.get(k, 0.0) == 1.0
    ]

    worker_entry = {
        "worker_id": worker_id,
        "activity": activity,
        "required_ppe": all_required,
        "optional_ppe": activity_opt,
        "confirmed_ppe": confirmed_ppe,
        "missing_required": missing_required,
        "ppe_worn": worn_labels,
        "violation": is_violation,
        "severity": severity_label,
        "fully_visible": legs_visible and torso_visible,
    }

    violation_entry = None
    if is_violation:
        violation_entry = {
            "worker_id": worker_id,
            "osha_code": "OSHA 1926",
            "description": desc_text,
            "severity": severity_label,
            "activity": activity,
        }

    return worker_entry, violation_entry, severity_label


def analyze_scene(detections: list, original_pil_image, camera_id: str = "CAM-01") -> dict:
    """
    Full per-frame analysis pipeline: matches PPE to workers, fuses with
    CLIP, classifies severity per worker, and rolls everything up into
    a scene-level report.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    people = [d for d in detections if d["label"].lower() == "worker"]
    ppes = [d for d in detections if d["label"].lower() != "worker"]

    w, h = original_pil_image.size
    if not people and ppes:
        # No worker box detected, but PPE/tools are visible — assume a
        # single worker spanning the whole frame so we still report on it.
        people = [{"label": "worker", "confidence": 1.0, "box": [0, 0, w, h]}]

    scene_report = {
        "camera_id": camera_id,
        "timestamp": timestamp,
        "scene_summary": f"Detected {len(people)} worker(s) in frame.",
        "workers": [],
        "violations": [],
        "overall_risk": "NONE",
        "stop_work_required": False,
        "notes": "Analyzed locally via Custom NumPy Decision Tree.",
    }

    highest_severity_val = 0

    for i, person in enumerate(people):
        worker_id = f"Worker {chr(65 + i)}"
        worn_labels = _match_ppe_to_worker(person, ppes)

        worker_crop = _crop_worker(person, original_pil_image)

        print(f"[{worker_id}] Fusing YOLO Data with CLIP Inference...")
        print(f"[DEBUG] worn_labels passed to CLIP: {worn_labels}")
        worker_json = clip_validator.generate_hybrid_json(worker_crop, worn_labels)

        visibility = pose_sys.get_visibility(original_pil_image, person["box"])
        worker_json["legs_visible"] = 1.0 if visibility.get("legs_visible", True) else 0.0
        worker_json["torso_visible"]  =1.0 if visibility.get("torso_visible", True) else 0.0

        worker_entry, violation_entry, severity_label = _build_worker_entry(
            worker_id, worn_labels, worker_json, visibility
        )

        if SEV_VALS[severity_label] > highest_severity_val:
            highest_severity_val = SEV_VALS[severity_label]
            scene_report["overall_risk"] = severity_label
            if severity_label == "STOP_WORK":
                scene_report["stop_work_required"] = True

        scene_report["workers"].append(worker_entry)
        if violation_entry:
            scene_report["violations"].append(violation_entry)

    worker_activities = [w.get("activity", "light_labor") for w in scene_report["workers"]]
    scene_activity, scene_unified_ppe = get_scene_activity_and_ppe(worker_activities)
    scene_report["scene_activity"] = scene_activity
    scene_report["scene_unified_ppe"] = scene_unified_ppe

    return scene_report
