import hashlib
import os
import subprocess
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
import torch
from ultralytics import YOLO


st.set_page_config(
    page_title="Low-Light Traffic Detection",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)


def apply_styles() -> None:
    st.markdown(
        """
        <style>
            .block-container {
                padding-top: 1.6rem;
                padding-bottom: 3rem;
                max-width: 1120px;
            }

            .stApp {
                background: #0b1018;
            }

            [data-testid="stSidebar"] {
                background: #111827;
                border-right: 1px solid #243244;
            }

            [data-testid="stSidebar"] * {
                color: #f8fafc;
            }

            [data-testid="stSidebar"] .stButton > button {
                border: 1px solid #38bdf8;
                background: #0284c7;
                color: white;
            }

            .app-title {
                font-size: 2.15rem;
                line-height: 1.1;
                font-weight: 800;
                color: #f8fafc;
                margin: 0 0 .45rem 0;
            }

            .app-subtitle {
                color: #9fb0c7;
                font-size: 1rem;
                margin-bottom: 1rem;
                max-width: 760px;
            }

            .hero {
                border: 1px solid #26384f;
                border-radius: 8px;
                padding: 1.15rem 1.25rem;
                background: #111827;
                box-shadow: 0 14px 34px rgba(0, 0, 0, .24);
            }

            .hero h2 {
                margin: 0 0 .5rem 0;
                color: #f8fafc;
                font-size: 1.45rem;
                line-height: 1.2;
            }

            .hero p {
                margin: 0;
                color: #b6c4d6;
                max-width: 820px;
            }

            .home-grid {
                display: grid;
                grid-template-columns: repeat(3, minmax(0, 1fr));
                gap: .85rem;
                margin: .85rem 0 1rem;
            }

            .metric-card {
                border: 1px solid #26384f;
                border-radius: 8px;
                padding: .9rem 1rem;
                background: #111827;
                min-height: 96px;
            }

            .metric-card h3 {
                margin: 0 0 .35rem 0;
                font-size: .82rem;
                color: #8aa0ba;
                font-weight: 700;
                text-transform: uppercase;
                letter-spacing: .04em;
            }

            .metric-card p {
                margin: 0;
                color: #f8fafc;
                font-size: 1.18rem;
                font-weight: 800;
            }

            .section-title {
                margin: .5rem 0 .75rem;
                color: #f8fafc;
                font-size: 1.1rem;
                font-weight: 800;
            }

            .workflow-grid {
                display: grid;
                grid-template-columns: repeat(3, minmax(0, 1fr));
                gap: .85rem;
            }

            .step {
                border: 1px solid #26384f;
                border-top: 3px solid #38bdf8;
                border-radius: 8px;
                padding: .85rem;
                color: #b6c4d6;
                background: #101722;
                min-height: 116px;
            }

            .step strong {
                display: block;
                color: #f8fafc;
                margin-bottom: .35rem;
            }

            @media (max-width: 900px) {
                .home-grid,
                .workflow-grid {
                    grid-template-columns: 1fr;
                }
            }

            div[data-testid="stTabs"] button {
                font-weight: 700;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


apply_styles()


def make_gamma_lut(gamma: float) -> np.ndarray:
    inv_gamma = 1.0 / gamma
    return (((np.arange(256) / 255.0) ** inv_gamma) * 255).astype("uint8")


def make_clahe(clip_limit: float, tile_size: int):
    return cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))


def enhance_low_light(
    frame: np.ndarray,
    gamma_lut: np.ndarray,
    clahe,
    alpha: float,
    beta: int,
) -> np.ndarray:
    den = cv2.bilateralFilter(frame, d=5, sigmaColor=75, sigmaSpace=75)
    gamma_corrected = cv2.LUT(den, gamma_lut)

    lab = cv2.cvtColor(gamma_corrected, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    enhanced_l = clahe.apply(l_channel)
    merged = cv2.merge((enhanced_l, a_channel, b_channel))
    enhanced = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

    return cv2.convertScaleAbs(enhanced, alpha=alpha, beta=beta)


@st.cache_resource(show_spinner="Loading YOLOv8 model...")
def load_model(model_name: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return YOLO(model_name).to(device), device


@st.cache_data(show_spinner=False)
def read_first_frame(path: str):
    cap = cv2.VideoCapture(path)
    ok, frame = cap.read()
    cap.release()
    return frame if ok else None


def render_header(title: str, subtitle: str) -> None:
    st.markdown(f'<h1 class="app-title">{title}</h1>', unsafe_allow_html=True)
    st.markdown(f'<p class="app-subtitle">{subtitle}</p>', unsafe_allow_html=True)


def render_home() -> None:
    render_header(
        "Low-Light Traffic Detection System",
        "Enhance dark traffic footage and detect cars, motorbikes, buses, and trucks with YOLOv8.",
    )

    st.markdown(
        """
        <div class="hero">
            <h2>Night traffic analysis workspace</h2>
            <p>
                Upload a traffic video, preview the first-frame enhancement, tune visibility controls,
                and export an annotated MP4 with vehicle detections.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="home-grid">
            <div class="metric-card"><h3>Enhancement</h3><p>CLAHE + gamma</p></div>
            <div class="metric-card"><h3>Detection</h3><p>YOLOv8</p></div>
            <div class="metric-card"><h3>Output</h3><p>Annotated MP4</p></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-title">Workflow</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="workflow-grid">
            <div class="step"><strong>1. Open Detection</strong>Use the sidebar page selector and upload a video.</div>
            <div class="step"><strong>2. Tune Enhancement</strong>Adjust gamma, CLAHE, brightness, and contrast while checking the preview.</div>
            <div class="step"><strong>3. Run Detection</strong>Choose the model and vehicle classes, then download the processed result.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.button("Open Detection", type="primary"):
        st.session_state["pending_page"] = "Detection"
        st.rerun()


def _request_stop() -> None:
    st.session_state["stop_requested"] = True


def persist_upload(uploaded) -> str:
    file_sig = hashlib.md5(uploaded.getvalue()).hexdigest()
    if st.session_state.get("upload_sig") != file_sig:
        old_path = st.session_state.get("upload_path")
        if old_path and os.path.exists(old_path):
            try:
                os.unlink(old_path)
            except OSError:
                pass
        tmp_in = tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded.name).suffix)
        tmp_in.write(uploaded.getvalue())
        tmp_in.flush()
        tmp_in.close()
        st.session_state["upload_sig"] = file_sig
        st.session_state["upload_path"] = tmp_in.name
    return st.session_state["upload_path"]


def render_detection(
    uploaded,
    gamma: float,
    clip_limit: float,
    tile_size: int,
    alpha: float,
    beta: int,
    model_name: str,
    conf_threshold: float,
    iou_threshold: float,
    wanted_classes: list[int],
    process_btn: bool,
) -> None:
    render_header(
        "Detection Workspace",
        "Upload a video, inspect the enhancement preview, and run YOLOv8 vehicle detection.",
    )

    if uploaded is None:
        st.info("Upload a video from the sidebar to get started.")
        return

    max_upload_bytes = 100 * 1024 * 1024
    if uploaded.size > max_upload_bytes:
        st.error(f"File too large ({uploaded.size / 1024 / 1024:.1f} MB). Limit is 100 MB.")
        return

    upload_path = persist_upload(uploaded)
    tab_preview, tab_result = st.tabs(["Preview", "Results"])

    with tab_preview:
        st.subheader("Enhancement Preview")
        sample_frame = read_first_frame(upload_path)

        if sample_frame is None:
            st.warning("Could not read the first frame for preview.")
            return

        gamma_lut = make_gamma_lut(gamma)
        clahe = make_clahe(clip_limit, tile_size)
        enhanced_sample = enhance_low_light(sample_frame, gamma_lut, clahe, alpha, beta)

        col_a, col_b = st.columns(2)
        with col_a:
            st.caption("Original frame")
            st.image(cv2.cvtColor(sample_frame, cv2.COLOR_BGR2RGB), use_container_width=True)
        with col_b:
            st.caption("Enhanced frame")
            st.image(cv2.cvtColor(enhanced_sample, cv2.COLOR_BGR2RGB), use_container_width=True)

    with tab_result:
        if not process_btn:
            if st.session_state.pop("was_processing", False):
                st.warning("Processing was stopped. Click Process Video to run again.")
            else:
                st.info("Click Process Video in the sidebar to start detection.")
            return

        if not wanted_classes:
            st.error("Select at least one vehicle class in the sidebar.")
            return

        st.session_state["was_processing"] = True
        st.session_state["stop_requested"] = False

        for stale_path in st.session_state.pop("output_temp_files", []):
            if stale_path and os.path.exists(stale_path):
                try:
                    os.unlink(stale_path)
                except OSError:
                    pass

        model, device = load_model(model_name)
        st.caption(f"Running on: **{device.upper()}**")
        st.button(
            "Stop processing",
            key="stop_btn",
            help="Interrupt the running job",
            on_click=_request_stop,
        )

        cap = cv2.VideoCapture(upload_path)
        if not cap.isOpened():
            st.error("Could not open the uploaded video.")
            st.session_state["was_processing"] = False
            return

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

        if width <= 0 or height <= 0:
            st.error(
                "Could not read video dimensions — the file may be corrupted "
                "or use an unsupported codec."
            )
            cap.release()
            st.session_state["was_processing"] = False
            return

        st.write(
            f"Resolution: **{width} x {height}** | FPS: **{fps:.2f}** | Frames: **{frame_count}**"
        )

        tmp_raw = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tmp_raw.close()
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(tmp_raw.name, fourcc, fps, (width, height))

        if not writer.isOpened():
            st.error(
                "Could not initialize the video writer (mp4v codec unavailable). "
                "Install an OpenCV build with FFmpeg/codec support."
            )
            cap.release()
            if os.path.exists(tmp_raw.name):
                try:
                    os.unlink(tmp_raw.name)
                except OSError:
                    pass
            st.session_state["was_processing"] = False
            return

        gamma_lut = make_gamma_lut(gamma)
        clahe = make_clahe(clip_limit, tile_size)

        progress_bar = st.progress(0, text="Processing frames...")
        status_text = st.empty()
        preview_slot = st.empty()

        start_time = time.time()
        frame_index = 0
        stopped = False
        try:
            while True:
                if st.session_state.get("stop_requested"):
                    stopped = True
                    break

                ret, frame = cap.read()
                if not ret:
                    break

                enhanced = enhance_low_light(frame, gamma_lut, clahe, alpha, beta)
                results = model(
                    enhanced,
                    conf=conf_threshold,
                    iou=iou_threshold,
                    classes=wanted_classes,
                    verbose=False,
                )
                annotated = results[0].plot()

                if annotated.shape[1] != width or annotated.shape[0] != height:
                    annotated = cv2.resize(annotated, (width, height))

                writer.write(annotated)

                if frame_index % 10 == 0:
                    if frame_count > 0:
                        pct = min((frame_index + 1) / frame_count, 1.0)
                        progress_bar.progress(pct, text=f"Frame {frame_index + 1} / {frame_count}")
                    else:
                        progress_bar.progress(0.0, text=f"Frame {frame_index + 1}")
                    status_text.caption(f"Detections this frame: **{len(results[0].boxes)}**")
                    preview_slot.image(
                        cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                        caption="Live preview",
                        use_container_width=True,
                    )
                frame_index += 1
        except Exception:
            st.session_state["was_processing"] = False
            st.session_state["stop_requested"] = False
            raise
        finally:
            cap.release()
            writer.release()

        elapsed = time.time() - start_time
        proc_fps = frame_index / elapsed if elapsed > 0 else 0.0

        if stopped:
            progress_bar.empty()
            status_text.empty()
            preview_slot.empty()
            st.session_state["was_processing"] = False
            st.session_state["stop_requested"] = False
            st.warning(
                f"Processing stopped after {frame_index} frames "
                f"({elapsed:.1f}s, {proc_fps:.1f} fps)."
            )
            if os.path.exists(tmp_raw.name):
                try:
                    os.unlink(tmp_raw.name)
                except OSError:
                    pass
            return

        st.session_state["was_processing"] = False

        tmp_final = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tmp_final.close()
        ffmpeg_ok = False
        try:
            result = subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-i",
                    tmp_raw.name,
                    "-vcodec",
                    "libx264",
                    "-pix_fmt",
                    "yuv420p",
                    "-an",
                    tmp_final.name,
                ],
                capture_output=True,
                timeout=300,
            )
            ffmpeg_ok = result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            ffmpeg_ok = False

        output_path = tmp_final.name if ffmpeg_ok else tmp_raw.name

        progress_bar.progress(1.0, text="Done")
        status_text.empty()
        preview_slot.empty()

        st.success(
            f"Processing complete: {frame_index} frames in {elapsed:.1f}s ({proc_fps:.1f} fps)"
        )

        st.video(output_path)
        with open(output_path, "rb") as f:
            st.download_button(
                label="Download Processed Video",
                data=f,
                file_name=f"detected_{Path(uploaded.name).stem}.mp4",
                mime="video/mp4",
            )

        unused_path = tmp_raw.name if ffmpeg_ok else tmp_final.name
        if unused_path and os.path.exists(unused_path):
            try:
                os.unlink(unused_path)
            except OSError:
                pass
        st.session_state["output_temp_files"] = [output_path]


if "page" not in st.session_state:
    st.session_state["page"] = "Home"

if "pending_page" in st.session_state:
    st.session_state["page"] = st.session_state.pop("pending_page")


with st.sidebar:
    st.title("Traffic Vision")
    current_page = st.radio(
        "Page",
        ["Home", "Detection"],
        key="page",
        label_visibility="collapsed",
    )

    st.divider()
    uploaded_file = st.file_uploader(
        "Upload video",
        type=["mp4", "avi", "mov", "mkv", "webm"],
        disabled=current_page != "Detection",
    )

    st.divider()
    st.subheader("Enhancement")
    gamma_value = st.slider("Gamma", 0.3, 2.0, 0.9, 0.05, help="< 1 brightens, > 1 darkens")
    clip_limit_value = st.slider("CLAHE clip limit", 0.5, 10.0, 2.0, 0.5)
    tile_size_value = st.select_slider("CLAHE tile size", options=[4, 8, 16, 32], value=8)
    alpha_value = st.slider("Contrast", 0.5, 3.0, 1.2, 0.1)
    beta_value = st.slider("Brightness", -50, 80, 8, 1)

    st.divider()
    st.subheader("Detection")
    model_value = st.selectbox(
        "YOLOv8 model",
        ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt"],
        index=2,
        help="Larger models are more accurate but slower.",
    )
    conf_value = st.slider("Confidence threshold", 0.1, 0.9, 0.45, 0.05)
    iou_value = st.slider("IOU threshold", 0.1, 0.9, 0.45, 0.05)

    vehicle_classes = {
        "Car (2)": 2,
        "Motorbike (3)": 3,
        "Bus (5)": 5,
        "Truck (7)": 7,
    }
    selected_classes = st.multiselect(
        "Detect classes",
        list(vehicle_classes.keys()),
        default=list(vehicle_classes.keys()),
    )
    class_ids = [vehicle_classes[class_name] for class_name in selected_classes]

    st.divider()
    run_detection = st.button(
        "Process Video",
        type="primary",
        use_container_width=True,
        disabled=current_page != "Detection",
    )


if current_page == "Home":
    render_home()
else:
    render_detection(
        uploaded_file,
        gamma_value,
        clip_limit_value,
        tile_size_value,
        alpha_value,
        beta_value,
        model_value,
        conf_value,
        iou_value,
        class_ids,
        run_detection,
    )
