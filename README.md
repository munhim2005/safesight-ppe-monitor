# SafeSight — Construction Safety Monitor

A real-time construction site safety monitor that fuses a fine-tuned YOLOv8 PPE detector, CLIP-based scene understanding, pose-based visibility checking, and a from-scratch NumPy decision tree to flag PPE violations and classify their severity.

## What it does

- **Detects PPE and hazards** using a YOLOv8 model fine-tuned on a custom 19-class dataset (hard hats, vests, gloves, boots, eye/ear protection, masks, plus tools and fire-safety equipment).
- **Classifies worker activity** (welding, cutting, or general/light labor) using YOLO's tool detections as the primary signal, with CLIP as a fallback when no tool is detected by YOLO.
- **Checks body-region visibility** using YOLOv8-pose, so PPE that's simply out of frame (e.g. a worker's legs cut off by the camera edge) is reported as "unconfirmed" rather than a false violation. Requires ankle keypoints specifically (not just knees) to count legs as visible, since a visible knee doesn't guarantee a visible foot.
- **Classifies violation severity** (`NONE` → `STOP_WORK`) using a decision tree implemented from scratch in NumPy — no scikit-learn — trained on synthetic data derived from OSHA construction standards.
- Runs entirely **locally** — no external APIs at inference time, aside from the one-time model download from Hugging Face.

## Dashboard

A single Gradio app with five tabs:

| Tab | What it does |
|---|---|
| 📷 Image Inspector | Upload a single image, get annotated detections + full violation report |
| 🔴 Live Web-Stream | Browser-native webcam streaming (no OpenCV, no external camera setup) with live risk overlay and automatic violation screenshots on a 5-second cooldown |
| 🎥 Video Pipeline | Upload a video, sample frames at a configurable FPS, get an annotated output video + running summary + violation gallery |
| 📁 Video Archive | Browse, replay, and delete previously processed videos |
| 📸 Screenshot Archive | Browse, view, and delete auto-captured violation screenshots |

The live tab streams directly through the browser's webcam API — no IP camera, no separate hardware, no OpenCV window. Earlier versions of this project used a dedicated ESP32-CAM integration; that's been removed in favor of this simpler, more portable approach that works on any machine with a webcam.

## Architecture

```
Frame
  │
  ├─► YOLOv8 (fine-tuned)  ──► PPE + tool detections
  │
  ├─► YOLOv8-pose           ──► body-region visibility (torso/legs)
  │
  ▼
Match PPE to worker (IoA overlap) ──► CLIP (only if YOLO sees no tool) ──► activity
  │
  ▼
Fuse: confirmed PPE / missing PPE / unconfirmed PPE (visibility-gated)
  │
  ▼
Decision Tree ──► severity (NONE / LOW / MEDIUM / HIGH / STOP_WORK)
```

## Why a decision tree, not a hardcoded lookup table

The severity rules are derived from OSHA standards, which are naturally expressible as if/else logic — so it's fair to ask why this isn't just a hardcoded rule function instead of a trained model.

The decision tree was chosen specifically to make **future rule additions easy and auditable**. A tree's structure is itself a sequence of feature splits, the same shape as the OSHA logic it's modeling — so adding a new rule means adding a few lines to the synthetic data generator and retraining (seconds, since the dataset is small), rather than hand-editing a growing if/else chain and hoping nothing upstream breaks. `print_tree()` also lets you read the model's learned logic back as a flowchart, so you can directly verify it captured a rule correctly (e.g. confirm it checks `using_hot_work` before `fire_safety`, matching OSHA 1926.352) — a hardcoded function would need separate documentation to claim the same thing, while the tree's structure is its own documentation.

The real constraint, worth being upfront about: this only eases rule changes at the *logic* layer. A new rule that references a feature nothing currently detects (a new tool, a new PPE type) still requires fine-tuning YOLO on a new class first — the tree can't learn a feature that was never measured.

## Why the training data isn't fully deterministic

Once a decision tree is the chosen architecture, it should actually be learning something — not just memorizing a lookup table dressed up as a model. Early versions of this project trained the tree on perfectly deterministic synthetic data, where every input mapped to exactly one correct output. The tree hit ~99.5% accuracy, but that number was meaningless: it had simply reverse-engineered the rule generator with extra steps, since there was nothing genuinely ambiguous to get wrong.

To make this a real learning problem, the training data now includes visibility as a feature: when a worker's legs or torso aren't visible in frame, "PPE not detected" is genuinely ambiguous — it might be a real violation, or the PPE might just be out of shot. The data generator models this with a random draw rather than a fixed rule, so the same input can produce different valid labels. The tree has to learn a sensible default for this ambiguity rather than memorize a single answer, since no rule can perfectly separate it.

Fire-safety equipment is only required for hot work (welding) specifically, not general power-tool use like cutting — this follows OSHA 1926.352, which governs welding/cutting fire prevention, not power tools generally.

## Why CLIP doesn't get full authority over activity detection

The original design let CLIP guess the activity (welding/cutting/light labor) whenever YOLO detected no tool. Testing on real footage showed this fails specifically when a worker is simply standing still or inspecting a piece — CLIP confidently (>85%) misclassified these idle frames as "cutting," because it was being forced to choose between three options with no real visual evidence for any of them. CLIP's role is now restricted to genuinely ambiguous tool-shape cases, and a dedicated "idle/inspecting" prompt was added so CLIP has a real category to fall back to instead of forcing a hazardous guess. YOLO remains the primary authority on tool/activity detection.

There is no "grinding" activity in this system — the trained YOLO model has no grinder class, and letting CLIP solely invent an entire hazard category with zero detector backing produced unreliable results during testing. It was removed rather than patched around.

## Known limitations

- **Pose-based visibility is not used for facial orientation.** Construction PPE (helmets, welding masks, safety glasses) is designed to obscure facial keypoints, so pose models tend to underperform on faces in exactly this scene type. Visibility checking is restricted to torso/leg keypoints, where PPE doesn't interfere.
- **Activity detection depends on YOLO's recall on tool classes**, which varies — evaluation showed lower recall on welding equipment specifically under motion blur or bright arc-light conditions compared to clean reference images. When YOLO misses a real tool, CLIP's narrower fallback prompts may not always catch it.
- **No authentication or access control** — this is a local/demo system, not hardened for production deployment.
- **Pose matching to worker boxes uses IoU/IoA overlap** between YOLOv8-pose's independent person detections and the PPE model's worker boxes — in crowded scenes with overlapping workers, this matching can occasionally pick the wrong skeleton.

## PPE Model

The fine-tuned YOLOv8 PPE detector (19 classes, ~13,000 training images combined from multiple public datasets) is hosted separately on Hugging Face and downloaded automatically on first run:

**[huggingface.co/killuminati1/construction-ppe-yolov8](https://huggingface.co/killuminati1/construction-ppe-yolov8)**

Full training metrics, confusion matrix, and known per-class performance are documented there.

## Setup

```bash
pip install -r requirements.txt
```

The PPE model weights download automatically from Hugging Face on first run. YOLOv8-pose weights download automatically via Ultralytics on first run.

Generate the severity model before first launch:

```bash
cd severity
python3 decision_tree.py
cd ..
```

This trains the decision tree on synthetic OSHA-derived data and saves `dt_model.npz` (not committed to this repo — it's a build artifact, regenerated locally).

Run the app:

```bash
python3 app.py
```

This launches the Gradio dashboard. `annotated_vids/` and `screenshots/` folders are created automatically as you use the app — neither is committed to the repo.

## Project structure

```
app.py                  Gradio dashboard: all 5 tabs, archive helpers, live-stream handler
config.py               Constants (device, thresholds, label colors)
models.py               Loads all models once: YOLO, CLIP, pose, decision tree
video_pipeline.py       Video upload pipeline: frame sampling and summary building
core/
  scene_analyzer.py     Core per-frame analysis: PPE matching, visibility, fusion
  geometry.py           Bounding-box overlap math
  report_formatter.py   Markdown report generation (presentation only)
detectors/
  yolo_module.py        Fine-tuned YOLOv8 PPE detector wrapper
  clip_validator.py     CLIP activity classification + PPE rules
  pose_module.py        YOLOv8-pose body-region visibility checking
severity/
  decision_tree.py      From-scratch NumPy decision tree
  data_generator.py     Synthetic OSHA-derived training data generator
```

## License

MIT
