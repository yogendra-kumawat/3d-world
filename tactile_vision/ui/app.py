from __future__ import annotations
import time
import threading
import traceback
import numpy as np
import cv2
import gradio as gr
import matplotlib

matplotlib.use("Agg")

from tactile_vision.pipeline import process_frame, PipelineParams
from tactile_vision.depth.estimator import DepthEstimator
from tactile_vision.detection.detector import ObjectDetector
from tactile_vision.tactile.renderer import render_pin_matrix_bgr, render_pin_matrix_matplotlib_3d
from tactile_vision.geometry.viz import render_view, build_plotly_pointcloud, render_birds_eye
from tactile_vision.audio.narrator import Narrator, Pyttsx3Engine


_DEPTH_CACHE: dict[tuple[str, str], DepthEstimator] = {}
_DETECTOR_CACHE: dict[str, ObjectDetector] = {}


def _get_estimator(model_size: str, domain: str) -> DepthEstimator:
    key = (model_size, domain)
    if key not in _DEPTH_CACHE:
        _DEPTH_CACHE[key] = DepthEstimator(model_size=model_size, domain=domain)
    return _DEPTH_CACHE[key]


def _get_detector(yolo_model: str) -> ObjectDetector:
    if yolo_model not in _DETECTOR_CACHE:
        _DETECTOR_CACHE[yolo_model] = ObjectDetector(model_name=yolo_model)
    return _DETECTOR_CACHE[yolo_model]


def _draw_detections(frame_bgr: np.ndarray, detections, distances) -> np.ndarray:
    """Draw colored boxes + class + meters on the input frame."""
    out = frame_bgr.copy()
    palette = [
        (255, 99, 71), (60, 179, 113), (65, 105, 225), (255, 165, 0),
        (218, 112, 214), (32, 178, 170), (255, 215, 0), (220, 20, 60),
    ]
    for i, det in enumerate(detections):
        x1, y1, x2, y2 = det.box
        color = palette[i % len(palette)]
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        d = distances[i] if i < len(distances) else float("nan")
        if np.isfinite(d):
            label = f"{det.cls} {d:.1f}m {det.score:.0%}"
        else:
            label = f"{det.cls} {det.score:.0%}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        ty = max(0, y1 - 6)
        cv2.rectangle(out, (x1, ty - th - 4), (x1 + tw + 6, ty + 2), color, -1)
        cv2.putText(out, label, (x1 + 3, ty - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return out


def _depth_heatmap(depth: np.ndarray) -> np.ndarray:
    """Render depth as 'close=red, far=blue' (turbo colormap, inverted) plus min/max readout."""
    d = depth.copy().astype(np.float32)
    valid = np.isfinite(d) & (d > 0)
    if valid.any():
        d_min = float(np.percentile(d[valid], 2))
        d_max = float(np.percentile(d[valid], 98))
    else:
        d_min, d_max = 0.0, 1.0
    if d_max - d_min < 1e-6:
        d_norm = np.zeros_like(d, dtype=np.uint8)
    else:
        # Invert so closer = brighter end of colormap.
        t = (d - d_min) / (d_max - d_min)
        t = np.clip(1.0 - t, 0.0, 1.0)
        d_norm = (t * 255).astype(np.uint8)
    img = cv2.applyColorMap(d_norm, cv2.COLORMAP_TURBO)
    # Overlay min/max meters so the user can verify the model's metric scale.
    label = f"close={d_min:.1f}m  far={d_max:.1f}m  (red=close, blue=far)"
    cv2.rectangle(img, (0, 0), (img.shape[1], 28), (0, 0, 0), -1)
    cv2.putText(img, label, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def _png_to_array(png_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(png_bytes, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


class _NullEngine:
    def speak(self, text: str) -> None:
        pass


def _render_pin_panel(pin_matrix, viz_choice: str, labels=None):
    """Pick OpenCV or matplotlib 3D pin renderer based on UI choice."""
    if viz_choice.startswith("Matplotlib"):
        return render_pin_matrix_matplotlib_3d(pin_matrix, labels=labels)
    return render_pin_matrix_bgr(pin_matrix)


_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
_VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}


def _resolve_video_path(video_input) -> str | None:
    if video_input is None:
        return None
    if isinstance(video_input, str):
        return video_input
    if hasattr(video_input, "name"):
        return video_input.name
    return str(video_input)


def _classify_input(path: str | None) -> str:
    """Return 'image', 'video', or 'unknown' based on the file extension."""
    if not path:
        return "unknown"
    import os
    ext = os.path.splitext(path)[1].lower()
    if ext in _IMAGE_EXTS:
        return "image"
    if ext in _VIDEO_EXTS:
        return "video"
    return "unknown"


# Live shared state — read fresh each frame so slider changes apply mid-stream.
class LiveControls:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.values: dict = {
            "mode": "scene",
            "model_size": "small",
            "domain": "indoor",
            "rows": 8,
            "cols": 16,
            "levels": 8,
            "fx": 600.0,
            "fy": 600.0,
            "cx": 320.0,
            "cy": 240.0,
            "near_m": 0.5,
            "far_m": 15.0,
            "auto_range": True,
            "region_depth_m": 5.0,
            "region_width_m": 2.5,
            "tts_enabled": False,
            "frame_skip": 2,
            "infer_size": 392,
            "max_frames": 0,
            "enable_detection": True,
            "yolo_model": "yolov8n.pt",
            "yolo_conf": 0.35,
            "gate_walk_with_yolo": True,
            "subtract_ground": True,
            "scene_subtract_ground": True,
            "camera_pitch_deg": 0.0,
            "ground_clear_radius_m": 3.0,
            "ground_clear_falloff_m": 1.0,
            "view_name": "Depth heatmap",
            "show_3d_panel": True,
            "walk_auto_region": True,
            "loop_video": True,
            "pin_viz": "OpenCV heatmap (fast)",
            "identifier_min_dist_m": 0.5,
            "identifier_max_dist_m": 15.0,
            "identifier_region_width_m": 6.0,
        }
        self.running = False
        self.paused = False
        self.stop_requested = False

    def update(self, key: str, value) -> None:
        with self.lock:
            self.values[key] = value

    def snapshot(self) -> dict:
        with self.lock:
            return dict(self.values)


_LIVE = LiveControls()


def process_input_live(media_input):
    """Generator. Auto-detects image vs video and dispatches.
    Reads slider values from _LIVE on every frame so changes apply live."""
    path = _resolve_video_path(media_input)
    if path is None:
        yield None, None, None, None, gr.update(), "Please upload an image or video first."
        return
    kind = _classify_input(path)
    if kind == "image":
        yield from _process_image_once(path)
        return
    if kind == "unknown":
        yield None, None, None, None, gr.update(), f"Unsupported file type: {path}"
        return
    # Else: video.
    yield from _process_video_live(path)


def _build_panels_from_result(result, frame_bgr, snap, log_lines, last_plot):
    """Render the 5 output panels from a FrameResult. Returns (rgb, view, pin, bev, plot, last_plot)."""
    annotated = _draw_detections(frame_bgr, result.detections, result.detection_distances_m)
    rgb_panel = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    pin_panel = cv2.cvtColor(
        _render_pin_panel(result.pin_matrix, snap.get("pin_viz", "OpenCV heatmap (fast)"), labels=result.identifier_labels),
        cv2.COLOR_BGR2RGB,
    )
    try:
        view_bgr = render_view(
            snap["view_name"],
            frame_bgr=frame_bgr,
            depth=result.depth,
            points_xyz=result.points_xyz,
            height_above_floor=result.height_above_floor,
        )
        view_panel = cv2.cvtColor(view_bgr, cv2.COLOR_BGR2RGB)
    except Exception as e:
        log_lines.append(f"[view error] {type(e).__name__}: {e}")
        view_panel = np.zeros((360, 480, 3), dtype=np.uint8)
    try:
        bev_bgr = render_birds_eye(result.points_xyz, result.height_above_floor)
        bev_panel = cv2.cvtColor(bev_bgr, cv2.COLOR_BGR2RGB)
    except Exception as e:
        log_lines.append(f"[BEV error] {type(e).__name__}: {e}")
        bev_panel = np.zeros((360, 480, 3), dtype=np.uint8)
    plot_out = gr.update()
    if snap["show_3d_panel"]:
        try:
            last_plot = build_plotly_pointcloud(
                result.points_xyz, result.height_above_floor, frame_bgr
            )
            plot_out = last_plot
        except Exception as e:
            log_lines.append(f"[3D plot error] {type(e).__name__}: {e}")
    return rgb_panel, view_panel, pin_panel, bev_panel, plot_out, last_plot


def _process_image_once(image_path: str):
    """Single-image flow. Runs the same pipeline as one video frame, emits all 5 panels."""
    if _LIVE.running:
        yield None, None, None, None, gr.update(), "A run is already in progress. Stop it first."
        return

    _LIVE.running = True
    _LIVE.paused = False
    _LIVE.stop_requested = False
    log_lines: list[str] = []
    try:
        frame_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if frame_bgr is None:
            yield None, None, None, None, gr.update(), f"OpenCV could not read image: {image_path}"
            return

        snap = _LIVE.snapshot()
        estimator = _get_estimator(snap["model_size"], snap["domain"])
        estimator.infer_size = int(snap["infer_size"])

        detector = None
        if snap["enable_detection"]:
            detector = _get_detector(snap["yolo_model"])
            detector.conf_thresh = float(snap["yolo_conf"])

        params = PipelineParams(
            mode=snap["mode"],
            rows=int(snap["rows"]), cols=int(snap["cols"]), levels=int(snap["levels"]),
            fx=float(snap["fx"]), fy=float(snap["fy"]),
            cx=float(snap["cx"]), cy=float(snap["cy"]),
            near_m=float(snap["near_m"]), far_m=float(snap["far_m"]),
            auto_range=bool(snap["auto_range"]),
            region_depth_m=float(snap["region_depth_m"]),
            region_width_m=float(snap["region_width_m"]),
            enable_detection=bool(snap["enable_detection"]),
            gate_walk_with_yolo=bool(snap["gate_walk_with_yolo"]),
            subtract_ground=bool(snap["subtract_ground"]),
            scene_subtract_ground=bool(snap["scene_subtract_ground"]),
            camera_pitch_deg=float(snap["camera_pitch_deg"]),
            ground_clear_radius_m=float(snap["ground_clear_radius_m"]),
            ground_clear_falloff_m=float(snap["ground_clear_falloff_m"]),
            walk_auto_region=bool(snap["walk_auto_region"]),
            identifier_min_dist_m=float(snap["identifier_min_dist_m"]),
            identifier_max_dist_m=float(snap["identifier_max_dist_m"]),
            identifier_region_width_m=float(snap["identifier_region_width_m"]),
        )
        result = process_frame(
            frame_bgr, params=params, depth_estimator=estimator, timestamp=0.0, detector=detector,
        )
        if result.audio_text:
            log_lines.append(f"[image] {result.audio_text}")

        rgb_p, view_p, pin_p, bev_p, plot_out, _ = _build_panels_from_result(
            result, frame_bgr, snap, log_lines, last_plot=None
        )
        log_lines.append("-- single-image processing complete; change sliders and click Run again to re-render")
        yield rgb_p, view_p, pin_p, bev_p, plot_out, "\n".join(log_lines[-15:])
    finally:
        _LIVE.running = False
        _LIVE.paused = False
        _LIVE.stop_requested = False


def _process_video_live(video_path: str):
    """Original video streaming generator. Reads slider values from _LIVE per frame."""
    if _LIVE.running:
        yield None, None, None, None, gr.update(), "A run is already in progress. Stop it first."
        return

    _LIVE.running = True
    _LIVE.paused = False
    _LIVE.stop_requested = False

    try:
        # Initial estimator load.
        snap = _LIVE.snapshot()
        estimator = _get_estimator(snap["model_size"], snap["domain"])
        estimator.infer_size = int(snap["infer_size"])
        current_depth_key = (snap["model_size"], snap["domain"])

        detector: ObjectDetector | None = None
        current_yolo = None
        if snap["enable_detection"]:
            detector = _get_detector(snap["yolo_model"])
            detector.conf_thresh = float(snap["yolo_conf"])
            current_yolo = snap["yolo_model"]

        narrator_engine = Pyttsx3Engine() if snap["tts_enabled"] else _NullEngine()
        narrator = Narrator(engine=narrator_engine, min_interval_s=3.0, enabled=snap["tts_enabled"])

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            yield None, None, None, None, gr.update(), f"OpenCV could not open the file: {video_path}"
            return
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_idx = 0
        log_lines: list[str] = []
        t0 = time.monotonic()
        last_panels = None
        last_plot = None

        while not _LIVE.stop_requested:
            if _LIVE.paused:
                # Re-emit the last frame so the UI doesn't go stale, but don't advance.
                if last_panels is not None:
                    rgb_p, view_p, pin_p, bev_p, _ = last_panels
                    yield rgb_p, view_p, pin_p, bev_p, gr.update(), "\n".join(log_lines[-15:] + ["[paused]"])
                time.sleep(0.05)
                continue

            ok, frame_bgr = cap.read()
            if not ok:
                # End of video. Loop if requested, otherwise stop.
                if _LIVE.snapshot().get("loop_video", False):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    log_lines.append(f"-- looping video at frame {frame_idx}")
                    ok, frame_bgr = cap.read()
                    if not ok:
                        break
                else:
                    break

            snap = _LIVE.snapshot()

            # Hot-swap estimator if model size or domain changed.
            new_depth_key = (snap["model_size"], snap["domain"])
            if new_depth_key != current_depth_key:
                estimator = _get_estimator(*new_depth_key)
                current_depth_key = new_depth_key
            estimator.infer_size = int(snap["infer_size"])

            # Hot-swap detector.
            if snap["enable_detection"]:
                if detector is None or snap["yolo_model"] != current_yolo:
                    detector = _get_detector(snap["yolo_model"])
                    current_yolo = snap["yolo_model"]
                detector.conf_thresh = float(snap["yolo_conf"])
            else:
                detector = None

            # Hot-swap TTS state.
            if snap["tts_enabled"] != narrator.enabled:
                narrator.enabled = snap["tts_enabled"]
                if snap["tts_enabled"] and isinstance(narrator.engine, _NullEngine):
                    narrator.engine = Pyttsx3Engine()

            skip = max(1, int(snap["frame_skip"]))
            do_inference = (frame_idx % skip) == 0

            if do_inference:
                try:
                    params = PipelineParams(
                        mode=snap["mode"],
                        rows=int(snap["rows"]),
                        cols=int(snap["cols"]),
                        levels=int(snap["levels"]),
                        fx=float(snap["fx"]),
                        fy=float(snap["fy"]),
                        cx=float(snap["cx"]),
                        cy=float(snap["cy"]),
                        near_m=float(snap["near_m"]),
                        far_m=float(snap["far_m"]),
                        auto_range=bool(snap["auto_range"]),
                        region_depth_m=float(snap["region_depth_m"]),
                        region_width_m=float(snap["region_width_m"]),
                        enable_detection=bool(snap["enable_detection"]),
                        gate_walk_with_yolo=bool(snap["gate_walk_with_yolo"]),
                        subtract_ground=bool(snap["subtract_ground"]),
                        scene_subtract_ground=bool(snap["scene_subtract_ground"]),
                        camera_pitch_deg=float(snap["camera_pitch_deg"]),
                        ground_clear_radius_m=float(snap["ground_clear_radius_m"]),
                        ground_clear_falloff_m=float(snap["ground_clear_falloff_m"]),
                        walk_auto_region=bool(snap["walk_auto_region"]),
                        identifier_min_dist_m=float(snap["identifier_min_dist_m"]),
                        identifier_max_dist_m=float(snap["identifier_max_dist_m"]),
                        identifier_region_width_m=float(snap["identifier_region_width_m"]),
                    )
                    result = process_frame(
                        frame_bgr,
                        params=params,
                        depth_estimator=estimator,
                        timestamp=frame_idx / fps,
                        detector=detector,
                    )
                    if result.audio_text:
                        narrator.maybe_speak(result.audio_text)
                        log_lines.append(f"[{result.timestamp:5.2f}s] {result.audio_text}")

                    annotated = _draw_detections(frame_bgr, result.detections, result.detection_distances_m)
                    rgb_panel = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                    pin_panel = cv2.cvtColor(
        _render_pin_panel(result.pin_matrix, snap.get("pin_viz", "OpenCV heatmap (fast)"), labels=result.identifier_labels),
        cv2.COLOR_BGR2RGB,
    )
                    # Dispatched view: depth / BEV / side / obstacle map.
                    try:
                        view_bgr = render_view(
                            snap["view_name"],
                            frame_bgr=frame_bgr,
                            depth=result.depth,
                            points_xyz=result.points_xyz,
                            height_above_floor=result.height_above_floor,
                        )
                        view_panel = cv2.cvtColor(view_bgr, cv2.COLOR_BGR2RGB)
                    except Exception as e:
                        log_lines.append(f"[view error] {type(e).__name__}: {e}")
                        view_panel = np.zeros((360, 480, 3), dtype=np.uint8)
                    # 2D BEV for the bottom-right panel.
                    try:
                        bev_bgr = render_birds_eye(result.points_xyz, result.height_above_floor)
                        bev_panel = cv2.cvtColor(bev_bgr, cv2.COLOR_BGR2RGB)
                    except Exception as e:
                        log_lines.append(f"[BEV error] {type(e).__name__}: {e}")
                        bev_panel = np.zeros((360, 480, 3), dtype=np.uint8)
                    # 3D plot is heavier and optional; only rebuild every 6th inference frame.
                    if snap["show_3d_panel"] and frame_idx % (skip * 6) == 0:
                        try:
                            last_plot = build_plotly_pointcloud(
                                result.points_xyz, result.height_above_floor, frame_bgr
                            )
                        except Exception as e:
                            log_lines.append(f"[3D plot error] {type(e).__name__}: {e}")
                        plot_out = last_plot
                    else:
                        plot_out = gr.update()
                    last_panels = (rgb_panel, view_panel, pin_panel, bev_panel, last_plot)
                except Exception as e:
                    tb = traceback.format_exc(limit=3)
                    log_lines.append(f"[frame {frame_idx} error] {type(e).__name__}: {e}\n{tb}")
                    # Reuse the last good panels so the stream stays alive.
                    if last_panels is None:
                        frame_idx += 1
                        continue
                    rgb_panel, view_panel, pin_panel, bev_panel, _ = last_panels
                    plot_out = gr.update()
            else:
                if last_panels is None:
                    frame_idx += 1
                    continue
                rgb_panel = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                _, view_panel, pin_panel, bev_panel, _ = last_panels
                plot_out = gr.update()
                last_panels = (rgb_panel, view_panel, pin_panel, bev_panel, last_plot)

            yield rgb_panel, view_panel, pin_panel, bev_panel, plot_out, "\n".join(log_lines[-15:])

            frame_idx += 1
            cap_max = int(snap["max_frames"])
            if cap_max > 0 and frame_idx >= cap_max:
                break

        cap.release()
        elapsed = time.monotonic() - t0
        log_lines.append(
            f"-- finished {frame_idx} frames in {elapsed:.1f}s "
            f"({frame_idx / max(elapsed, 1e-3):.1f} fps)"
        )
        yield None, None, None, None, gr.update(), "\n".join(log_lines[-15:])
    finally:
        _LIVE.running = False
        _LIVE.paused = False
        _LIVE.stop_requested = False


def _bind_live(component, key: str):
    """Wire a Gradio component's change event to update the shared live state."""
    component.change(fn=lambda v, k=key: _LIVE.update(k, v), inputs=[component], outputs=None)


def _toggle_pause():
    _LIVE.paused = not _LIVE.paused
    return "Resume" if _LIVE.paused else "Pause"


def _stop():
    _LIVE.stop_requested = True
    return "stop requested"


def build_app() -> gr.Blocks:
    with gr.Blocks(title="Tactile Vision Aid - Phase 1 (live)") as app:
        gr.Markdown("# Tactile Vision Aid - Phase 1 (live tuning)")
        gr.Markdown("Sliders apply **live** during playback — no need to stop and re-run.")

        with gr.Row():
            with gr.Column(scale=1):
                video = gr.File(
                    label="Input image or video (auto-detected)",
                    file_types=[
                        ".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v",
                        ".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff",
                    ],
                    file_count="single",
                    type="filepath",
                )
                view_name = gr.Dropdown(
                    choices=[
                        "Depth heatmap",
                        "Top-down BEV",
                        "Side view",
                        "Obstacle map",
                        "Depth contours",
                        "3D depth mesh (matplotlib)",
                    ],
                    value="Depth heatmap",
                    label="Display view (top-right panel) - swap live",
                )
                show_3d_panel = gr.Checkbox(value=True, label="Show interactive 3D point cloud (slower)")
                pin_viz = gr.Dropdown(
                    choices=["OpenCV heatmap (fast)", "Matplotlib 3D bars (slow but rich)"],
                    value="OpenCV heatmap (fast)",
                    label="Pin matrix renderer",
                )
                mode = gr.Radio(
                    ["walk", "scene", "identifier"],
                    value="scene",
                    label="Mode (walk=BEV, scene=depth feel, identifier=YOLO objects placed in 3D)",
                )
                model_size = gr.Radio(["small", "base", "large"], value="small", label="Depth model size")
                domain = gr.Radio(["indoor", "outdoor"], value="indoor", label="Domain")
                with gr.Accordion("Pin grid", open=False):
                    rows = gr.Slider(4, 32, value=8, step=1, label="Rows")
                    cols = gr.Slider(8, 64, value=16, step=1, label="Cols")
                    levels = gr.Radio([2, 8, 256], value=8, label="Height levels")
                with gr.Accordion("Camera intrinsics + viewpoint", open=False):
                    fx = gr.Slider(50, 1500, value=600, step=10, label="fx")
                    fy = gr.Slider(50, 1500, value=600, step=10, label="fy")
                    cx = gr.Slider(0, 2000, value=320, step=10, label="cx")
                    cy = gr.Slider(0, 2000, value=240, step=10, label="cy")
                    camera_pitch_deg = gr.Slider(
                        -45, 45, value=0, step=1,
                        label="Camera pitch deg (+ = look up like a person walking forward)",
                    )
                with gr.Accordion("Ground plane (ignore the road)", open=True):
                    subtract_ground = gr.Checkbox(value=True, label="Subtract ground plane in walk mode")
                    scene_subtract_ground = gr.Checkbox(value=True, label="Subtract ground plane in scene mode")
                    ground_clear_radius_m = gr.Slider(
                        0.5, 10.0, value=3.0, step=0.1,
                        label="Ground subtraction radius (m) — only ignore floor within this distance",
                    )
                    ground_clear_falloff_m = gr.Slider(
                        0.1, 5.0, value=1.0, step=0.1,
                        label="Falloff zone (m) — soft transition outside the radius",
                    )
                with gr.Accordion("Scene-mode clipping", open=True):
                    auto_range = gr.Checkbox(value=True, label="Auto range (5th-95th percentile per frame)")
                    near_m = gr.Slider(0.1, 5.0, value=0.5, step=0.05, label="Near (m) - manual mode")
                    far_m = gr.Slider(2.0, 50.0, value=15.0, step=0.5, label="Far (m) - manual mode")
                with gr.Accordion("Identifier-mode range", open=False):
                    identifier_min_dist_m = gr.Slider(0.1, 5.0, value=0.5, step=0.1, label="Min distance (m) — closer = pin max")
                    identifier_max_dist_m = gr.Slider(2.0, 50.0, value=15.0, step=0.5, label="Max distance (m) — beyond = pin 0")
                    identifier_region_width_m = gr.Slider(1.0, 20.0, value=6.0, step=0.5, label="Lateral region width (m)")
                with gr.Accordion("Walk-mode region", open=False):
                    walk_auto_region = gr.Checkbox(
                        value=True,
                        label="Auto-fit region (recommended; falls back to fixed sliders if off)",
                    )
                    region_depth_m = gr.Slider(1.0, 30.0, value=5.0, step=0.5, label="Region depth (m) [manual]")
                    region_width_m = gr.Slider(1.0, 20.0, value=2.5, step=0.5, label="Region width (m) [manual]")
                    gate_walk_with_yolo = gr.Checkbox(
                        value=True,
                        label="Gate with YOLO (only raise pins where detector sees an obstacle)",
                    )
                with gr.Accordion("Object detection (YOLOv8)", open=True):
                    enable_detection = gr.Checkbox(value=True, label="Enable YOLO detection")
                    yolo_model = gr.Radio(
                        ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt"],
                        value="yolov8n.pt",
                        label="YOLO model (n=fastest, l=most accurate)",
                    )
                    yolo_conf = gr.Slider(0.05, 0.9, value=0.35, step=0.05, label="Confidence threshold")
                with gr.Accordion("Performance", open=True):
                    frame_skip = gr.Slider(1, 10, value=2, step=1, label="Frame skip (1=every frame)")
                    infer_size = gr.Slider(196, 784, value=392, step=14, label="Inference size (long side)")
                    max_frames = gr.Slider(0, 1000, value=0, step=10, label="Max frames (0 = no cap)")
                loop_video = gr.Checkbox(value=True, label="Loop video automatically when it ends")
                tts = gr.Checkbox(value=False, label="Enable TTS narration")
                with gr.Row():
                    run_btn = gr.Button("Run", variant="primary")
                    pause_btn = gr.Button("Pause")
                    stop_btn = gr.Button("Stop", variant="stop")
            with gr.Column(scale=2):
                with gr.Row():
                    rgb_out = gr.Image(label="Input frame + YOLO")
                    depth_out = gr.Image(label="Selected view (use dropdown to switch)")
                with gr.Row():
                    pin_out = gr.Image(label="Tactile pin matrix")
                    bev_out = gr.Image(label="Top-down BEV (floor=gray, obstacles=color)")
                plot3d_out = gr.Plot(label="3D point cloud (drag to rotate, scroll to zoom)")
                log_out = gr.Textbox(label="Audio narration log", lines=6)

        # Bind every control's change event to the shared live state.
        _bind_live(mode, "mode")
        _bind_live(model_size, "model_size")
        _bind_live(domain, "domain")
        _bind_live(rows, "rows")
        _bind_live(cols, "cols")
        _bind_live(levels, "levels")
        _bind_live(fx, "fx")
        _bind_live(fy, "fy")
        _bind_live(cx, "cx")
        _bind_live(cy, "cy")
        _bind_live(auto_range, "auto_range")
        _bind_live(near_m, "near_m")
        _bind_live(far_m, "far_m")
        _bind_live(walk_auto_region, "walk_auto_region")
        _bind_live(loop_video, "loop_video")
        _bind_live(pin_viz, "pin_viz")
        _bind_live(identifier_min_dist_m, "identifier_min_dist_m")
        _bind_live(identifier_max_dist_m, "identifier_max_dist_m")
        _bind_live(identifier_region_width_m, "identifier_region_width_m")
        _bind_live(region_depth_m, "region_depth_m")
        _bind_live(region_width_m, "region_width_m")
        _bind_live(gate_walk_with_yolo, "gate_walk_with_yolo")
        _bind_live(subtract_ground, "subtract_ground")
        _bind_live(scene_subtract_ground, "scene_subtract_ground")
        _bind_live(camera_pitch_deg, "camera_pitch_deg")
        _bind_live(ground_clear_radius_m, "ground_clear_radius_m")
        _bind_live(ground_clear_falloff_m, "ground_clear_falloff_m")
        _bind_live(view_name, "view_name")
        _bind_live(show_3d_panel, "show_3d_panel")
        _bind_live(tts, "tts_enabled")
        _bind_live(frame_skip, "frame_skip")
        _bind_live(infer_size, "infer_size")
        _bind_live(max_frames, "max_frames")
        _bind_live(enable_detection, "enable_detection")
        _bind_live(yolo_model, "yolo_model")
        _bind_live(yolo_conf, "yolo_conf")

        run_btn.click(
            fn=process_input_live,
            inputs=[video],
            outputs=[rgb_out, depth_out, pin_out, bev_out, plot3d_out, log_out],
        )
        pause_btn.click(fn=_toggle_pause, inputs=None, outputs=pause_btn)
        stop_btn.click(fn=_stop, inputs=None, outputs=log_out)

    return app
