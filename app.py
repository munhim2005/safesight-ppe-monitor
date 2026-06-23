"""
SafeSight - Construction Safety Monitor
Fully integrated, browser-native dashboard.
"""

import gradio as gr
import os
import re, time
import glob
from datetime import datetime

from core.report_formatter import format_detections, format_scene_report
from core.scene_analyzer import analyze_scene
from models import vision_sys
from video_pipeline import process_video, _equipment_visible_for_activity

# ==========================================
# --- FILE MANAGEMENT & ARCHIVE HELPERS ---
# ==========================================

def parse_timestamp(filepath):
    basename = os.path.basename(filepath)
    match = re.search(r'(\d{8}_\d{6})', basename)
    if match:
        try:
            return datetime.strptime(match.group(1), "%Y%m%d_%H%M%S")
        except ValueError:
            pass
    return datetime.fromtimestamp(os.path.getctime(filepath))

def get_grouped_files(folder, extension):
    os.makedirs(folder, exist_ok=True)
    files = glob.glob(f"{folder}/*{extension}")
    files.sort(key=parse_timestamp, reverse=True)
    
    choices = []
    for f in files:
        dt = parse_timestamp(f)
        label = f"{dt.strftime('%Y-%m-%d %H:%M')} | {os.path.basename(f)}"
        choices.append((label, f))
    return choices

def refresh_videos():
    choices = get_grouped_files("annotated_vids", ".mp4")
    return gr.update(choices=choices, value=choices[0][1] if choices else None)

def refresh_screenshots():
    choices = get_grouped_files("screenshots", ".jpg")
    return gr.update(choices=choices, value=choices[0][1] if choices else None)

def delete_file(filepath, folder, extension):
    if filepath and os.path.exists(filepath):
        os.remove(filepath)
    choices = get_grouped_files(folder, extension)
    return gr.update(choices=choices, value=choices[0][1] if choices else None), None

# ==========================================
# --- PROCESSORS ---
# ==========================================

def process_image(pil_img, camera_id):
    if pil_img is None:
        return None, "No image provided.", "Upload an image first."

    annotated, detections = vision_sys.get_detections(pil_img)
    scene = analyze_scene(detections, pil_img, camera_id or "CAM-01")

    yolo_text = format_detections(detections)
    scene_text = format_scene_report(scene)

    return annotated, yolo_text, scene_text

def process_live_stream(frame, last_shot_time):
    """Handles browser-native live streaming with auto-screenshots"""
    if frame is None:
        return None, "Waiting for stream...", last_shot_time
        
    annotated, detections = vision_sys.get_detections(frame)
    scene = analyze_scene(detections, frame, "CAM-WEB")
    risk = scene.get("overall_risk", "NONE")
    
    status_msg = f"### Current Scene Risk: {risk}"
    if risk in ["HIGH", "STOP_WORK"]:
        status_msg = f"### 🚨 CRITICAL RISK: {risk} 🚨"
        
    # --- LIVE SCREENSHOT LOGIC (5-Second Cooldown) ---
    current_time = time.time()
    
    if (current_time - last_shot_time) >= 5.0: # 5.0 seconds cooldown
        detected_labels = {d["label"].lower() for d in detections}
        
        # Check if the dangerous tool is actually in the frame
        needs_screenshot = any(
            _equipment_visible_for_activity(v.get("activity", "light_labor"), detected_labels)
            for v in scene.get("violations", [])
        )
        
        if needs_screenshot:
            os.makedirs("screenshots", exist_ok=True)
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            screenshot_path = f"screenshots/violation_{timestamp_str}_LIVE.jpg"
            annotated.save(screenshot_path)
            print(f"[LIVE SCREENSHOT] Auto-captured at {timestamp_str}")
            
            last_shot_time = current_time # Reset the cooldown timer
            
    return annotated, status_msg, last_shot_time

# ==========================================
# --- UI LAYOUT ---
# ==========================================

# Using a built-in clean theme instead of messy CSS
theme = gr.themes.Soft(
    primary_hue="emerald",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Inter"), "sans-serif"]
)

with gr.Blocks(title="SafeSight Dashboard", theme=theme) as demo:
    gr.Markdown("# ⬡ SafeSight Safety Monitor\n*Local Processing · YOLOv8 Vision · Decision Tree Engine*")

    with gr.Tabs():
        
        # --- TAB 1: IMAGE ANALYSIS ---
        with gr.TabItem("📷 Image Inspector"):
            with gr.Row():
                with gr.Column(scale=1):
                    img_input = gr.Image(type="pil", label="Upload Site Image", height=350)
                    cam_id_img = gr.Textbox(value="CAM-01", label="Camera ID")
                    img_button = gr.Button("Analyze Image", variant="primary")
                with gr.Column(scale=1):
                    # Height explicitly constrained to prevent giant UI blow-ups
                    img_output = gr.Image(type="pil", label="Annotated Output", height=450)
            
            with gr.Row():
                with gr.Column():
                    img_dino = gr.Markdown("#### YOLOv8 Detections")
                with gr.Column():
                    img_scene = gr.Markdown("#### Decision Tree Report")

            img_button.click(fn=process_image, inputs=[img_input, cam_id_img], outputs=[img_output, img_dino, img_scene])

        # --- TAB 2: LIVE WEBCAM (IN BROWSER) ---
        # --- TAB 2: LIVE WEBCAM (IN BROWSER) ---
        with gr.TabItem("🔴 Live Web-Stream"):
            gr.Markdown("Streams directly from your local webcam into the browser for real-time risk assessment.")
            with gr.Row():
                with gr.Column(scale=1):
                    live_input = gr.Image(sources=["webcam"], streaming=True, type="pil", label="Live Camera Input", height=450)
                with gr.Column(scale=1):
                    live_output = gr.Image(type="pil", label="Live Annotations", height=450, interactive=False)
                    live_status = gr.Markdown("### Current Scene Risk: STANDBY")
            
            # ADD THIS: A hidden state variable to track the clock
            live_timer_state = gr.State(value=0.0)

            # Wires the continuous stream to the processor function
            live_input.stream(
                fn=process_live_stream,
                inputs=[live_input, live_timer_state],  # <-- Add the timer here
                outputs=[live_output, live_status, live_timer_state] # <-- And here
            )

        # --- TAB 3: VIDEO PIPELINE ---
        with gr.TabItem("🎥 Video Pipeline"):
            with gr.Row():
                with gr.Column(scale=1):
                    vid_input = gr.Video(label="Upload Site Video")
                    with gr.Row():
                        sample_fps = gr.Slider(minimum=0.5, maximum=5, value=2, step=0.5, label="FPS Rate")
                        summary_every = gr.Slider(minimum=1, maximum=30, value=5, step=1, label="UI Refresh (Frames)")
                    vid_button = gr.Button("Process Pipeline", variant="primary")
                with gr.Column(scale=1):
                    vid_output = gr.Video(label="Annotated Output", height=400, interactive=False)
                    vid_summary = gr.Markdown(value="*Upload a video and press Process...*")
            
            vid_gallery = gr.Gallery(label="Violation Snapshots", columns=4, height=250, object_fit="contain")

            vid_button.click(
                fn=process_video,
                inputs=[vid_input, sample_fps, gr.State("CAM-VID"), summary_every],
                outputs=[vid_output, vid_summary, vid_gallery],
            )

        # --- TAB 4: VIDEO ARCHIVE ---
        with gr.TabItem("📁 Video Archive"):
            with gr.Row():
                with gr.Column(scale=1):
                    vid_dropdown = gr.Dropdown(
                        choices=get_grouped_files("annotated_vids", ".mp4"), 
                        label="Select Processed Video"
                    )
                    with gr.Row():
                        refresh_vid_btn = gr.Button("🔄 Refresh", variant="secondary")
                        delete_vid_btn = gr.Button("🗑️ Delete", variant="stop")
                
                with gr.Column(scale=2):
                    archive_player = gr.Video(label="Playback", height=450, autoplay=True, interactive=False)

            refresh_vid_btn.click(fn=refresh_videos, inputs=[], outputs=[vid_dropdown])
            vid_dropdown.change(fn=lambda x: x, inputs=[vid_dropdown], outputs=[archive_player])
            delete_vid_btn.click(
                fn=lambda x: delete_file(x, "annotated_vids", ".mp4"),
                inputs=[vid_dropdown],
                outputs=[vid_dropdown, archive_player]
            )

        # --- TAB 5: SCREENSHOT ARCHIVE ---
        with gr.TabItem("📸 Screenshot Archive"):
            with gr.Row():
                with gr.Column(scale=1):
                    img_dropdown = gr.Dropdown(
                        choices=get_grouped_files("screenshots", ".jpg"), 
                        label="Select Violation Log"
                    )
                    with gr.Row():
                        refresh_img_btn = gr.Button("🔄 Refresh", variant="secondary")
                        delete_img_btn = gr.Button("🗑️ Delete", variant="stop")
                
                with gr.Column(scale=2):
                    # Explicit height and object_fit prevents the image from becoming huge
                    archive_viewer = gr.Image(type="filepath", label="Violation Viewer", height=500)

            refresh_img_btn.click(fn=refresh_screenshots, inputs=[], outputs=[img_dropdown])
            img_dropdown.change(fn=lambda x: x, inputs=[img_dropdown], outputs=[archive_viewer])
            delete_img_btn.click(
                fn=lambda x: delete_file(x, "screenshots", ".jpg"),
                inputs=[img_dropdown],
                outputs=[img_dropdown, archive_viewer]
            )

if __name__ == "__main__":
    demo.launch(share=False)