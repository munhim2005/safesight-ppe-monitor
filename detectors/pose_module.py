"""
pose_module.py
--------------
Wraps YOLOv8-pose to answer one narrow question: which body regions of
a worker are actually visible in frame? This feeds the severity model
a real "visibility" signal instead of assuming every detected worker
box is a fully-visible body.

We deliberately do NOT use this for facial orientation/identity — PPE
(helmets, welding masks, safety glasses) is designed to obscure facial
keypoints, so pose confidence on the face is structurally unreliable
in exactly the scenes this app needs to handle. We only trust it for
torso/hip/knee/ankle visibility, which PPE doesn't obscure.
"""

from ultralytics import YOLO
from config import DEVICE

# Standard YOLOv8-pose (COCO) keypoint indices we actually care about.
KP_SHOULDER_L, KP_SHOULDER_R = 5, 6
KP_HIP_L, KP_HIP_R = 11, 12
KP_KNEE_L, KP_KNEE_R = 13, 14
KP_ANKLE_L, KP_ANKLE_R = 15, 16

# Minimum keypoint confidence to count as "visible" rather than guessed.
KEYPOINT_CONF_THRESHOLD = 0.5


class PoseModule:
    def __init__(self, model_name: str = "yolov8n-pose.pt"):
        print(f"[Pose] Loading YOLOv8-pose ({model_name})...")
        self.model = YOLO(model_name).to(DEVICE)
        print(f"[Pose] Model loaded on {DEVICE}")

    def get_visibility(self, pil_img, person_box: list) -> dict:
        """
        Runs pose estimation on the full frame and returns a visibility
        dict for whichever detected skeleton best matches person_box.

        Returns:
            {
                "torso_visible": bool,
                "legs_visible": bool,
                "pose_detected": bool,   # False if no matching skeleton found at all
            }
        """
        results = self.model(pil_img, verbose=False)
        result = results[0]

        if result.keypoints is None or len(result.keypoints) == 0:
            return {"torso_visible": True, "legs_visible": True, "pose_detected": False}

        best_match = self._match_skeleton_to_box(result, person_box)
        if best_match is None:
            return {"torso_visible": True, "legs_visible": True, "pose_detected": False}

        kp_xy = best_match.xy[0]      # (17, 2) pixel coords
        kp_conf = best_match.conf[0]  # (17,) confidence per keypoint

        def kp_ok(idx: int) -> bool:
            return float(kp_conf[idx]) >= KEYPOINT_CONF_THRESHOLD

        torso_visible = kp_ok(KP_SHOULDER_L) or kp_ok(KP_SHOULDER_R) or kp_ok(KP_HIP_L) or kp_ok(KP_HIP_R)
        legs_visible = (
            kp_ok(KP_ANKLE_L) or kp_ok(KP_ANKLE_R)
        )

        return {"torso_visible": torso_visible, "legs_visible": legs_visible, "pose_detected": True}

    def _match_skeleton_to_box(self, result, person_box: list):
        """
        YOLOv8-pose detects every person in the frame independently of
        our PPE-model's worker boxes. Pick whichever detected skeleton's
        bounding box overlaps person_box the most.
        """
        if result.boxes is None or len(result.boxes) == 0:
            return None

        from core.geometry import calculate_ioa

        best_ioa = 0.0
        best_idx = None
        for i, box in enumerate(result.boxes.xyxy):
            ioa = calculate_ioa(box.tolist(), person_box)
            if ioa > best_ioa:
                best_ioa = ioa
                best_idx = i

        if best_idx is None or best_ioa < 0.3:
            return None

        return result.keypoints[best_idx]