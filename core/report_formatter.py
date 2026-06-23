"""
report_formatter.py
---------------------
Turns raw detection lists and scene-analysis dicts into the Markdown
strings shown in the Gradio UI. Pure presentation — no detection or
severity logic lives here.
"""

RISK_EMOJI = {
    "NONE":      "🟢",
    "LOW":       "🟡",
    "MEDIUM":    "🟠",
    "HIGH":      "🔴",
    "STOP_WORK": "🚨",
    "unknown":   "⚪",
}

ACTIVITY_LABEL = {
    "welding":     "🔥 Welding",
    "cutting":     "🪚 Cutting",
    "light_labor": "🏗️ Light Labor",
    "unknown":     "❓ Unknown",
}


def format_detections(detections):
    if not detections:
        return "No objects detected."

    lines = []
    for i, d in enumerate(detections, 1):
        x0, y0, x1, y1 = d["box"]
        conf = d["confidence"]
        conf_str = conf if isinstance(conf, str) else f"{conf:.1%}"
        lines.append(
            f"**{i}. {d['label'].upper()}** — {conf_str}\n"
            f"   Box: ({int(x0)},{int(y0)}) → ({int(x1)},{int(y1)})"
        )
    return "\n\n".join(lines)


def _fmt(key: str) -> str:
    """'ear_prot' -> 'Ear Prot', 'hard_hat' -> 'Hard Hat'"""
    return key.replace("_", " ").title()


def format_scene_report(scene: dict) -> str:
    if not scene:
        return "No analysis available."

    overall_risk = scene.get("overall_risk", "unknown")
    scene_activity = scene.get("scene_activity", "unknown")
    scene_unified = scene.get("scene_unified_ppe", [])

    lines = [
        f"## 📷 {scene.get('camera_id', 'CAM-01')}  |  {scene.get('timestamp', '')}",
        f"{scene.get('scene_summary', '')}",
        "",
        "| Field | Value |",
        "|---|---|",
        f"| **Overall Risk** | {RISK_EMOJI.get(overall_risk, '⚪')} **{overall_risk}** |",
        f"| **Scene Activity** | {ACTIVITY_LABEL.get(scene_activity, scene_activity.title())} |",
        f"| **Scene Required PPE** | {', '.join(_fmt(p) for p in scene_unified) if scene_unified else 'None'} |",
    ]

    for w in scene.get("workers", []):
        activity = w.get("activity", "unknown")
        missing = w.get("missing_required", [])
        severity = w.get("severity", "NONE")
        required = w.get("required_ppe", [])
        is_viol = w.get("violation", False)

        status_icon = "❌" if is_viol else "✅"
        sev_icon = RISK_EMOJI.get(severity, "⚪")

        lines.append("\n---")
        lines.append(f"### {status_icon} {w['worker_id']}")
        lines.append("| | |")
        lines.append("|---|---|")
        lines.append(f"| **Activity** | {ACTIVITY_LABEL.get(activity, activity.title())} |")
        lines.append(f"| **Severity** | {sev_icon} {severity} |")

        if missing:
            missing_str = "  ".join(f"❌ {_fmt(m)}" for m in missing)
            lines.append(f"| **Missing Gear** | {missing_str} |")
        else:
            lines.append("| **Missing Gear** | ✅ None — fully compliant |")

        unconfirmed = w.get("unconfirmed_ppe", [])
        if unconfirmed:
            unconfirmed_str = "  ".join(f"❔ {_fmt(u)}" for u in unconfirmed)
            lines.append(f"| **Unconfirmed (out of frame)** | {unconfirmed_str} |")

        present = [item for item in required if item not in missing]
        if present:
            present_str = "  ".join(f"✅ {_fmt(p)}" for p in present)
            lines.append(f"| **Gear Present** | {present_str} |")

    violations = scene.get("violations", [])
    lines.append("\n---")
    if violations:
        lines.append("### ⚠️ Violation Summary")
        for v in violations:
            sev = v.get("severity", "MEDIUM")
            act = v.get("activity", "unknown").replace("_", " ").title()
            desc = v.get("description", "")
            missing_part = desc.split("] ", 1)[-1] if "]" in desc else desc
            lines.append(
                f"{RISK_EMOJI.get(sev, '🟠')} **{sev}** &nbsp;·&nbsp; "
                f"**{v.get('worker_id')}** &nbsp;·&nbsp; "
                f"Activity: **{act}** &nbsp;·&nbsp; {missing_part}"
            )
    else:
        lines.append("### ✅ No violations detected")

    if scene.get("notes"):
        lines.append(f"\n📝 {scene.get('notes')}")

    return "\n".join(lines)
