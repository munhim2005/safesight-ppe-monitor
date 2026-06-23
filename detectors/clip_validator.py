"""
clip_backup.py
----------------
A lightweight Vision-Language model to double-check YOLO detections.
Uses OpenAI's CLIP to:
  1. Independently scan for ALL PPE items (fills gaps YOLO missed)
  2. Classify the worker's activity
  3. Determine what required PPE is missing based on that activity
"""

import torch
from transformers import CLIPProcessor, CLIPModel

# ── PPE Prompts ───────────────────────────────────────────────────────────────
# CLIP checks every one of these unconditionally, regardless of what YOLO found.
# If YOLO already confirmed an item, the CLIP result is ignored for that item.
# If YOLO missed an item but CLIP finds it, it gets filled in.

CLIP_PPE_PROMPTS = {
    "hard_hat": (
        "a construction worker wearing a hard hat or safety helmet on their head",
        "a construction worker with a completely bare head, no helmet or hard hat of any kind"
    ),
    "vest": (
        "a construction worker wearing a bright reflective high-visibility safety vest over their torso",
        "a construction worker with no vest, just a plain shirt, jacket, or coveralls"
    ),
    "glass": (
        "a worker with face or eye protection: safety glasses, goggles, a face shield, or a welding helmet visor",
        "a worker with a completely bare unprotected face and open eyes, no eyewear of any kind"
    ),
    "glove": (
        "a construction worker wearing work gloves, welding gloves, or protective gloves on their hands",
        "a construction worker with completely bare hands, no gloves of any kind"
    ),
    "boots": (
        "a construction worker wearing heavy steel toe boots, safety boots, or thick work boots",
        "a construction worker wearing sneakers, sandals, dress shoes, or light footwear"
    ),
    "ear_prot": (
        "a construction worker wearing earplugs inserted in their ears or ear muffs over their ears",
        "a construction worker with completely bare ears, no hearing protection of any kind"
    ),
    "mask": (
        "a construction worker wearing a respirator, dust mask, welding shield, or protective face covering",
        "a construction worker with a completely bare face, no mask, respirator, or face covering"
    ),
}

# ── Activity-Specific PPE Rules ───────────────────────────────────────────────
# Defines what PPE is REQUIRED for each activity.
# Items not listed here are not flagged as violations for that activity.
# Universal items (hard_hat, boots) are enforced separately below.

ACTIVITY_PPE_RULES = {
    'welding': {
        # No vest — welders wear coveralls/fire-resistant jackets instead
        'required': ['glass', 'glove', 'mask'],
        'optional': ['ear_prot'],
    },
    'cutting': {
        'required': ['glass', 'glove'],
        'optional': ['ear_prot', 'mask'],
    },
    'light_labor': {
        # Standard site work — vest is required here
        'required': ['vest'],
        'optional': ['glove', 'mask'],
    },
}

# Always required regardless of activity
# Always required regardless of activity
UNIVERSAL_REQUIRED = ['hard_hat', 'boots']

# ── Scene-level activity helpers ──────────────────────────────────────────────
# Risk rank — higher = more hazardous. Scene takes the max of all workers.
ACTIVITY_RISK_RANK = {
    'light_labor': 0,
    'cutting':     1,
    'welding':     2,
}

def get_scene_activity_and_ppe(worker_activities: list) -> tuple:
    """
    Derives a single scene-level activity and its unified required PPE
    from a list of per-worker activities.

    Rules:
      - light_labor is background noise — ignored when hazardous work is present
      - If only one hazardous activity type → use it
      - If multiple hazardous types → highest risk wins for the activity label,
        but required PPE is the UNION of all hazardous activities present
      - If no hazardous activity at all → light_labor

    Returns: (scene_activity_str, unified_required_ppe_list)
    """
    hazardous = [a for a in worker_activities if a != 'light_labor']
    unique_hazardous = set(hazardous)

    if not unique_hazardous:
        scene_activity = 'light_labor'
        unified_ppe = list(set(
            UNIVERSAL_REQUIRED + ACTIVITY_PPE_RULES['light_labor']['required']
        ))
    elif len(unique_hazardous) == 1:
        scene_activity = unique_hazardous.pop()
        unified_ppe = list(set(
            UNIVERSAL_REQUIRED + ACTIVITY_PPE_RULES[scene_activity]['required']
        ))
    else:
        # Mixed hazardous — highest risk label, but union of ALL required PPE
        scene_activity = max(unique_hazardous, key=lambda a: ACTIVITY_RISK_RANK.get(a, 0))
        all_required = []
        for act in unique_hazardous:
            all_required += ACTIVITY_PPE_RULES.get(act, {}).get('required', [])
        unified_ppe = list(set(UNIVERSAL_REQUIRED + all_required))
        print(f"[SCENE] Mixed activities {unique_hazardous} → '{scene_activity}', "
              f"unified PPE: {unified_ppe}")

    return scene_activity, unified_ppe

# ── CLIPValidator ─────────────────────────────────────────────────────────────

class CLIPValidator:
    def __init__(self, device=None):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        print("[CLIP] Loading Lightweight Fallback Model...")
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        print("[CLIP] Fallback Ready.")

    def _score(self, image, pos_prompt, neg_prompt):
        """
        Runs a single binary CLIP check.
        Returns (positive_prob, negative_prob).
        """
        inp = self.processor(
            text=[pos_prompt, neg_prompt],
            images=image,
            return_tensors="pt",
            padding=True
        )
        inp = {k: v.to(self.device) for k, v in inp.items()}

        with torch.no_grad():
            probs = self.model(**inp).logits_per_image.softmax(dim=1).cpu().numpy()[0]

        return float(probs[0]), float(probs[1])

    def classify_activity(self, cropped_pil_image):
        """
        Classifies what the worker is doing.
        Returns: (activity_str, confidence_float)
        activity_str is one of: 'welding', 'cutting', 'light_labor'
        """
        prompts = [
            # Welding - tool-focused
            "a photo of welding torch actively welding metal, bright blue-white electric arc burning at the torch tip on the metal joint, glowing molten weld pool, hot sparks shooting straight down vertically from the electrode, intense arc light, close-up on the welding tool and arc",
            
            # Circular Saw Cutting
            "a photo of circular saw actively cutting metal or wood, large flat spinning toothed blade cutting through the material, sawdust and metal shavings ejecting sideways, close-up on the spinning blade and cutting action, no arc",
            
            # General work
            "a photo of construction worker doing general site work, standing walking or carrying materials, no active power tool, no sparks, no arc",

            # Idle / inspecting — no tool running at all
            "a photo of a worker standing still or holding a piece of metal or material, inspecting or positioning it by hand, no power tool switched on, no sparks, no glowing metal, no arc, nothing actively happening",
        ]
        inp = self.processor(
            text=prompts,
            images=cropped_pil_image,
            return_tensors="pt",
            padding=True
        )
        inp = {k: v.to(self.device) for k, v in inp.items()}

        with torch.no_grad():
            probs = self.model(**inp).logits_per_image.softmax(dim=1).cpu().numpy()[0]

        activities = ['welding', 'cutting', 'light_labor', 'light_labor']
        best_idx = int(probs.argmax())
        best_conf = float(probs[best_idx])

        # If CLIP isn't confident about a hazardous activity, don't guess — default to light_labor.
        # Hazardous activities (welding/grinding/cutting) need a clear visual signal.
        HAZARDOUS_CONFIDENCE_THRESHOLD = 0.35
        if activities[best_idx] != 'light_labor' and best_conf < HAZARDOUS_CONFIDENCE_THRESHOLD:
            print(f"[CLIP] Low confidence ({best_conf:.2f}) for '{activities[best_idx]}' → defaulting to 'light_labor'")
            return 'light_labor', best_conf

        return activities[best_idx], best_conf

    def generate_hybrid_json(self, cropped_pil_image, yolo_labels):
        """
        Full pipeline:
          1. Load YOLO's findings
          2. CLIP independently scans for ALL 7 PPE items
          3. Merge: present if EITHER source confirms it
          4. Classify activity
          5. Determine missing required PPE based on activity + universal rules

        Returns a dict ready for the MLP and the violation reporter.
        """

        # ── Step 1: Load YOLO findings ─────────────────────────────────────────
        yolo_result = {
            "hard_hat":   1.0 if 'hard_hat'        in yolo_labels else 0.0,
            "vest":       1.0 if 'vest'             in yolo_labels else 0.0,
            "glass":      1.0 if 'glass'            in yolo_labels else 0.0,
            "glove":      1.0 if 'glove'            in yolo_labels else 0.0,
            "boots":      1.0 if 'boots'            in yolo_labels else 0.0,
            "ear_prot":   1.0 if 'ear-protection'   in yolo_labels else 0.0,
            "mask":       1.0 if 'mask'             in yolo_labels else 0.0,
            "using_tool": 1.0 if any(any(kw in lbl for kw in ['weld', 'saw', 'torch', 'circular']) for lbl in yolo_labels) else 0.0,
            "using_hot_work": 1.0 if any('weld' in lbl or 'torch' in lbl for lbl in yolo_labels) else 0.0,
            "fire_safety":1.0 if any(any(kw in lbl for kw in ['fire_extinguisher', 'fire_prevention', 'extinguisher']) for lbl in yolo_labels) else 0.0,
        }

        # ── Step 2: CLIP scans ALL PPE items independently ────────────────────
        clip_result = {}
        for ppe_key, (pos, neg) in CLIP_PPE_PROMPTS.items():
            pos_prob, _ = self._score(cropped_pil_image, pos, neg)
            clip_result[ppe_key] = 1.0 if pos_prob > 0.55 else 0.0

        # ── Step 3: Merge ──────────────────────────────────────────────────────
        # An item is present if EITHER source confirms it.
        # YOLO cannot remove something CLIP confirmed, and vice versa.
        merged = {**yolo_result}
        for ppe_key in clip_result:
            if yolo_result[ppe_key] == 0.0 and clip_result[ppe_key] == 1.0:
                print(f"[CLIP GAP FILL] '{ppe_key}' missed by YOLO, confirmed by CLIP")
                merged[ppe_key] = 1.0

        # ── Step 4: Classify activity ──────────────────────────────────────────
        activity, confidence = self.classify_activity(cropped_pil_image)
        
        # --- YOLO Context Override ---
        # YOLO's physical tool detection is more reliable than CLIP's visual guess.
        # Priority: YOLO tool seen → use it. No tool seen → cap at light_labor.
        _lbl_set = [lbl.lower().replace(' ', '_') for lbl in yolo_labels]
        has_welding = any('weld' in lbl or 'torch' in lbl for lbl in _lbl_set)
        has_saw     = any('saw' in lbl or 'circular' in lbl or 'cutting' in lbl for lbl in _lbl_set)
        #yolo_saw_any_tool = has_welding or has_saw or has_grinder

        if has_welding:
            activity = 'welding'
            confidence = 1.0
            print(f"[FUSION] YOLO saw welding equipment → activity = 'welding'.")
        elif has_saw:
            activity = 'cutting'
            confidence = 1.0
            print(f"[FUSION] YOLO saw a saw → activity = 'cutting'.")
        elif activity != 'light_labor':
            # CLIP guessed a hazardous activity but YOLO saw no tool — don't trust it.
            print(f"[FUSION] YOLO saw no tool, but CLIP confidently read '{activity}' ({confidence:.2f}) → keeping CLIP's call.")
            #activity = 'light_labor'
            #confidence = 1.0

        merged["activity"] = activity
        merged["activity_confidence"] = round(confidence, 3)

        # Any active task counts as using a tool for the MLP
        if activity != 'light_labor':
            merged["using_tool"] = 1.0

        # ── Step 5: Determine missing required PPE ─────────────────────────────
        activity_required = ACTIVITY_PPE_RULES.get(activity, {}).get('required', [])
        all_required = list(set(UNIVERSAL_REQUIRED + activity_required))

        merged["missing_required"] = [
            ppe for ppe in all_required if merged.get(ppe, 0.0) == 0.0
        ]

        return merged