"""
data_generator.py
-----------------
Generates synthetic training data from OSHA rules.
Each row = one worker scenario → severity label.

Severity levels:
    0 = NONE        (fully compliant)
    1 = LOW         (minor violation)
    2 = MEDIUM      (moderate violation)
    3 = HIGH        (serious violation)
    4 = STOP_WORK   (immediate danger)

The New 9 Features (in order, all binary 0 or 1):
    [0] hard_hat 
    [1] vest
    [2] glass       (eye protection)
    [3] glove 
    [4] boots
    [5] ear_prot    (ear protection)
    [6] mask
    [7] using_tool  (1 if near circular_saw)
    [8] using_hot_work (for those tools which are required to have fire safety near by OSHA rules)
    [8] fire_safety (1 if near fire_extinguisher or fire_prevention_net)
    [9] legs_visible    (NEW — 0 or 1)
    [10] torso_visible  (NEW — 0 or 1)
"""

import numpy as np
import csv
import os

# ── Severity label map ──────────────────────────────────────────────────────
SEVERITY_MAP = {0: "NONE", 1: "LOW", 2: "MEDIUM", 3: "HIGH", 4: "STOP_WORK"}
SEVERITY_REVERSE = {v: k for k, v in SEVERITY_MAP.items()}


def determine_severity(hard_hat, vest, glass, glove, boots, ear_prot, mask, using_tool, using_hot_work, fire_safety, legs_visible, torso_visible):
    """
    Rule-based severity determination derived from OSHA standards.
    This is used to LABEL our synthetic training data for the custom MLP.
    """
    severity = 0  # start at NONE

    # ── STOP-WORK conditions (severity 4) ───────────────────────────────────
    
    #need fire safety for hot work i.e for which OSHA rules require it
    if using_hot_work and not fire_safety:
        return 4 # stop work-OSHA 1926.352(d)

    # Using grinder/welder without eye protection = Immediate Danger
    if using_tool and not glass:
        return 4  # STOP_WORK — OSHA 1926.102(a)(1)
        
    # Hot work (tools/welding) without fire safety equipment nearby
    if using_tool and not fire_safety:
        return 4  # STOP_WORK — OSHA 1926.352(d)

    # ── HIGH severity (3) ───────────────────────────────────────────────────
    
    # Power tools without heavy gloves
    if using_tool and not glove:
        severity = max(severity, 3) 
        
    # Power tools without ear protection
    if using_tool and not ear_prot:
        severity = max(severity, 3) # OSHA 1926.101(a)
        
    # Power tools/welding without mask (fumes/dust)
    if using_tool and not mask:
        severity = max(severity, 3)

    # ── MEDIUM severity (2) ─────────────────────────────────────────────────

    # Hard hat is a head/torso-region item. If torso isn't visible, "no
    # hard hat" is unconfirmed, not a violation — escalate only sometimes.
    if not hard_hat:
        if torso_visible:
            severity = max(severity, 2)  # OSHA 1926.100(a) — confirmed
        elif np.random.random() < 0.5:
            severity = max(severity, 1)  # unconfirmed — cautious read only

    # Boots are a leg-region item. Same logic.
    if not boots:
        if legs_visible:
            severity = max(severity, 2)  # OSHA 1926.96 — confirmed
        elif np.random.random() < 0.5:
            severity = max(severity, 1)  # unconfirmed — cautious read only

    # ── LOW severity (1) ────────────────────────────────────────────────────

    # Vest is torso-region too, same gating as hard hat.
    if not vest:
        if torso_visible:
            severity = max(severity, 1)  # OSHA 1926.201(a) — confirmed
        # if torso not visible: no escalation, genuinely can't tell

    return severity


def generate_dataset(n_samples=2000, random_seed=42):
    """
    Generates n_samples rows of synthetic training data.
    Returns X (features) and y (severity labels) as numpy arrays.
    """
    np.random.seed(random_seed)
    X = []
    y = []

    # ── Hardcoded OSHA rule scenarios (ensures core rules are represented heavily) ───
    # All scenarios below assume full visibility (legs_visible=1, torso_visible=1),
    # since these are meant to represent clean, fully-confirmed reference cases.
    # Visibility ambiguity gets introduced separately, in the random fill below.
    osha_scenarios = [
        # [hat, vest, glass, glove, boots, ear, mask, tool, hot_work, fire, legs_vis, torso_vis]

        # STOP_WORK
        [1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1],  # Saw, NO GLASSES → 4
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1],  # Welding, NO FIRE SAFETY → 4

        # HIGH
        [1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1],  # Saw, NO GLOVES → 3
        [1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1],  # Saw, NO EAR PROT → 3
        [1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1],  # Saw, NO MASK → 3

        # MEDIUM (confirmed)
        [0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1],  # Walking, NO HARD HAT → 2
        [1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1],  # Walking, NO BOOTS → 2

        # LOW (confirmed)
        [1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1],  # Walking, NO VEST → 1

        # NONE
        [1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1],  # Walking, compliant → 0
        [1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1],  # Sawing, compliant → 0

        # Ambiguous visibility cases
        [0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0],  # NO HARD HAT, torso not visible
        [1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1],  # NO BOOTS, legs not visible

        # New: a welding-specific STOP_WORK example with hot_work=1
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # Welding, compliant w/ fire safety → 0
    ]

    # Add all hardcoded scenarios first (multiple times for weight)
    for scenario in osha_scenarios:
        for _ in range(15):  # repeat each scenario 15x so they're well represented
            # Re-roll determine_severity each time for the ambiguous scenarios —
            # since those involve a random draw internally, repeating the same
            # input 15x should give a realistic MIX of labels, not one label
            # copied 15 times. Confirmed scenarios are deterministic, so the
            # re-roll is harmless for them.
            label = determine_severity(*scenario)
            X.append(scenario)
            y.append(label)

    # ── Random scenarios to fill up to n_samples ────────────────────────────
    while len(X) < n_samples:
        using_tool = np.random.randint(0, 2)
        using_hot_work = np.random.randint(0, 2) if using_tool else 0

        if using_hot_work:
            using_tool = 1

        
        ppe_features = [np.random.randint(0, 2) for _ in range(7)]
        fire_safety = np.random.randint(0, 2)

        # Visibility isn't 50/50 in real footage — legs get cut off by frame
        # edges more often than torsos do, so we skew accordingly rather than
        # using a uniform random draw like the PPE features above.
        legs_visible = 1 if np.random.random() < 0.85 else 0
        torso_visible = 1 if np.random.random() < 0.95 else 0

        features = ppe_features + [using_tool, using_hot_work,  fire_safety, legs_visible, torso_visible]
        label = determine_severity(*features)

        X.append(features)
        y.append(label)

    X = np.array(X[:n_samples], dtype=float)
    y = np.array(y[:n_samples], dtype=int)

    return X, y


def save_dataset(X, y, filepath="training_data.csv"):
    """Saves dataset to CSV for inspection."""
    headers = ["hard_hat", "vest", "glass", "glove", "boots", "ear_prot", "mask",
               "using_tool", "using_hot_work", "fire_safety", "legs_visible", "torso_visible", "severity"]
    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for features, label in zip(X, y):
            writer.writerow(list(features) + [label])
    print(f"[DataGenerator] Saved {len(X)} samples to {filepath}")


def print_distribution(y):
    """Prints class distribution of the dataset."""
    print("\n[DataGenerator] Class distribution:")
    for severity_int, severity_name in SEVERITY_MAP.items():
        count = np.sum(y == severity_int)
        bar = "█" * (count // 10)
        print(f"  {severity_name:<12} ({severity_int}): {count:>4} samples  {bar}")
    print()


# ── Run standalone to preview data ──────────────────────────────────────────
if __name__ == "__main__":
    X, y = generate_dataset(n_samples=2000)
    print_distribution(y)
    save_dataset(X, y, "training_data.csv")
    print(f"[DataGenerator] Sample row:")
    print(f"  Features: {X[0]}")
    print(f"  Severity: {SEVERITY_MAP[y[0]]}")