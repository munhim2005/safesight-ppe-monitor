"""
video_pipeline.py
------------------
Frame-sampling pipeline that drives analyze_scene() over an uploaded
video file, and builds the running Markdown summary shown in the
Gradio UI while processing. Live webcam streaming is handled directly
in app.py via Gradio's browser-native webcam input.
"""

import os
from collections import Counter
from datetime import datetime

import cv2
import numpy as np
from PIL import Image

from detectors.clip_validator import ACTIVITY_RISK_RANK
from models import vision_sys
from core.scene_analyzer import analyze_scene

MIN_ACTIVITY_FRAMES = 2 #min number of frames an activty should be detected to be considered real-detection

RISK_EMOJI = {"NONE": "🟢", "LOW": "🟡", "MEDIUM": "🟠", "HIGH": "🔴", "STOP_WORK": "🚨"}
ACTIVITY_ICON = {
    "welding":     "🔥 WELDING",
    "cutting":     "🪚 CUTTING",
    "light_labor": "🏗️ LIGHT LABOR",
}


def _derive_final_activity(activity_window: list) -> tuple:
    """
    Derives a final activity decision from a rolling window of per-frame
    activities. light_labor is treated as background — any hazardous
    activity seen in the window wins. Returns (final_activity, counts_dict).
    """
    counts = Counter(activity_window)
    hazardous = {a: c for a, c in counts.items() if a != "light_labor"}

    confirm_hazardous = {a: c for a, c in hazardous.items() if c> MIN_ACTIVITY_FRAMES}

    if not confirm_hazardous:
        final = "light_labor"
    else:
        final = max(confirm_hazardous.keys(), key=lambda a: ACTIVITY_RISK_RANK.get(a, 0))

    return final, dict(counts)


def _build_video_summary(frame_idx, src_fps, step, sample_fps,
                          all_labels, all_violations,
                          partial=False, sampled_count=0,
                          activity_window=None):
    # Only keep a "no-X" label if the positive "X" was never detected by YOLO
    clean_labels = set()
    for label in all_labels:
        if label.startswith("no-"):
            positive_version = label[3:]
            if positive_version not in all_labels:
                clean_labels.add(label)
        else:
            clean_labels.add(label)

    ui_labels = sorted(lbl.replace("_", " ").replace("-", " ").title() for lbl in clean_labels)
    status = "⏳ Processing..." if partial else "✅ Complete"

    lines = [
        f"**{status}** · Frames read: `{frame_idx}` · Sampled: `{sampled_count}`",
        f"**Sample rate:** {sample_fps} fps (every {step} frames)",
        f"**Detected classes:** {', '.join(ui_labels) if ui_labels else 'None'}",
        "---",
    ]

    if activity_window:
        final_activity, counts = _derive_final_activity(activity_window)
        total = len(activity_window)

        breakdown_parts = []
        for act, cnt in sorted(counts.items(), key=lambda x: x[1], reverse=True):
            icon = ACTIVITY_ICON.get(act, act.upper())
            pct = int(100 * cnt / total)
            breakdown_parts.append(f"{icon}: {cnt}/{total} ({pct}%)")

        decision_label = "Current Activity Decision" if partial else "Final Activity Decision"
        lines.append(f"### 🎯 {decision_label}")
        lines.append("\n".join(breakdown_parts))

        verdict_icon = ACTIVITY_ICON.get(final_activity, final_activity.upper())
        if final_activity != "light_labor" and counts.get(final_activity, 0) < counts.get("light_labor", 0):
            lines.append(
                f"\n**→ Final: {verdict_icon}** "
                f"*(hazardous activity detected — overrides light labor background)*"
            )
        else:
            lines.append(f"\n**→ Final: {verdict_icon}**")
        lines.append("---")

    if all_violations:
        sev_counts = Counter(v.get("severity", "MEDIUM") for v in all_violations)
        severity_order = ["NONE", "LOW", "MEDIUM", "HIGH", "STOP_WORK"]
        breakdown = "  ".join(
            f"{RISK_EMOJI.get(s, '🟠')} {s}: {c}"
            for s, c in sorted(sev_counts.items(), key=lambda x: severity_order.index(x[0]))
        )
        heading = "⏳ Violations so far" if partial else "⚠️ Violations Found"
        lines.append(f"### {heading}: {len(all_violations)}")
        lines.append(breakdown)
        lines.append("---")

        for v in all_violations:
            sev = v.get("severity", "MEDIUM")
            activity = v.get("activity", "unknown").replace("_", " ").upper()
            desc = v.get("description", "")
            missing_part = desc.split("] ", 1)[-1] if "]" in desc else desc
            screenshot = "📸" if v.get("screenshot") else ""
            lines.append(
                f"{RISK_EMOJI.get(sev, '🟠')} **{v.get('timestamp', '?')}** &nbsp;·&nbsp; "
                f"**{v.get('worker_id', '')}** &nbsp;·&nbsp; "
                f"🔧 {activity} &nbsp;·&nbsp; "
                f"{missing_part} {screenshot}"
            )
            lines.append("")
    else:
        label = "No violations so far" if partial else "No violations detected"
        lines.append(f"### ✅ {label}")

    return "\n\n".join(lines)


def _equipment_visible_for_activity(activity: str, detected_labels: set) -> bool:
    """Whether the tool/equipment matching this activity is actually visible in frame."""
    if activity == "welding":
        return any("weld" in l or "torch" in l for l in detected_labels)
    if activity == "cutting":
        return any("saw" in l or "circular" in l for l in detected_labels)
    return False  # light_labor has no associated equipment


# Maps CLIP's internal "missing_required" PPE names to the matching
# YOLO label, so the video summary's class list stays consistent
# regardless of which model actually flagged the absence.
_MISSING_LABEL_ALIASES = {
    "ear_prot": "ear-protection",
    "ear-protection": "ear-protection",
    "ear_protection": "ear-protection",
    "gloves": "glove",
}


def process_video(video_path, sample_fps, camera_id, summary_every):
    if video_path is None:
        yield None, "No video provided.", "Upload a video first."
        return

    cap = cv2.VideoCapture(video_path)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    step = max(1, int(src_fps / sample_fps))

    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("annotated_vids", exist_ok=True)
    out_path = f"annotated_vids/annotated_output_{timestamp_str}.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    writer = cv2.VideoWriter(out_path, fourcc, src_fps, (W, H))

    all_violations = []
    all_screenshots = []
    all_labels = set()
    activity_window = []
    frame_idx = 0
    sampled_count = 0
    last_annotated = None
    last_screenshot_count = -999
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % step == 0:
            pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            annotated_pil, detections = vision_sys.get_detections(pil)
            last_annotated = cv2.cvtColor(np.array(annotated_pil), cv2.COLOR_RGB2BGR)

            for d in detections:
                all_labels.add(d["label"])

            scene = analyze_scene(detections, pil, camera_id or "CAM-01")
            activity_window.append(scene.get("scene_activity", "light_labor"))

            for worker in scene.get("workers", []):
                for missing in worker.get("missing_required", []):
                    base = _MISSING_LABEL_ALIASES.get(missing, missing)
                    all_labels.add(f"no-{base}")

            if scene.get("violations"):
                ts = f"{frame_idx / src_fps:.1f}s"
                detected_labels = {d["label"].lower() for d in detections}

                needs_screenshot = any(
                    _equipment_visible_for_activity(v.get("activity", "light_labor"), detected_labels)
                    for v in scene["violations"]
                )
                screenshot_path = None
                cooldown_samples = int(sample_fps * 5) #5second cooldown
                # Only take a screenshot IF there is a violation AND the cooldown has passed
                if needs_screenshot and (sampled_count - last_screenshot_count) >= cooldown_samples:
                    os.makedirs("screenshots", exist_ok=True)
                    screenshot_path = f"screenshots/violation_{timestamp_str}_{frame_idx}.jpg"
                    annotated_pil.save(screenshot_path)
                    all_screenshots.append(screenshot_path)
                    
                    # Reset the cooldown timer
                    last_screenshot_count = sampled_count 
                    
                    print(f"[SCREENSHOT] Saved at {ts}")

                for v in scene["violations"]:
                    v["timestamp"] = ts
                    v["screenshot"] = screenshot_path
                    all_violations.append(v)

            sampled_count += 1
            if sampled_count % int(summary_every) == 0:
                yield None, _build_video_summary(
                    frame_idx, src_fps, step, sample_fps,
                    all_labels, all_violations,
                    partial=True, sampled_count=sampled_count,
                    activity_window=activity_window,
                ), all_screenshots

        writer.write(last_annotated if last_annotated is not None else frame)
        frame_idx += 1

    cap.release()
    writer.release()

    yield out_path, _build_video_summary(
        frame_idx, src_fps, step, sample_fps,
        all_labels, all_violations,
        partial=False, sampled_count=sampled_count,
        activity_window=activity_window,
    ), all_screenshots