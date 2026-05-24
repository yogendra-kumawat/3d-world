from __future__ import annotations
from dataclasses import dataclass, field
from typing import Literal, Optional, Protocol, List, Tuple
import numpy as np

from tactile_vision.types import DepthResult, PinMatrix, Detection
from tactile_vision.geometry.pointcloud import depth_to_points
from tactile_vision.geometry.ground import fit_ground_plane, height_above_plane, GroundPlane
from tactile_vision.tactile.encoder import SceneEncoder, WalkEncoder, IdentifierEncoder
from tactile_vision.detection.detector import detection_distance_m

Mode = Literal["walk", "scene", "identifier"]


class _DepthLike(Protocol):
    def predict(self, frame_bgr: np.ndarray) -> DepthResult: ...


class _DetectorLike(Protocol):
    def detect(self, frame_bgr: np.ndarray) -> List[Detection]: ...


@dataclass
class PipelineParams:
    mode: Mode = "scene"
    rows: int = 8
    cols: int = 16
    levels: int = 8
    fx: float = 320.0
    fy: float = 320.0
    cx: float = 160.0
    cy: float = 120.0
    near_m: float = 0.3
    far_m: float = 4.0
    auto_range: bool = False
    region_depth_m: float = 5.0
    region_width_m: float = 2.5
    walk_auto_region: bool = True
    enable_detection: bool = False
    gate_walk_with_yolo: bool = True
    subtract_ground: bool = True
    scene_subtract_ground: bool = True
    camera_pitch_deg: float = 0.0  # +ve = simulate tilting the camera up
    ground_clear_radius_m: float = 3.0  # only mask floor within this radius of camera
    ground_clear_falloff_m: float = 1.0  # soft transition zone outside the radius
    identifier_min_dist_m: float = 0.5
    identifier_max_dist_m: float = 15.0
    identifier_region_width_m: float = 6.0


@dataclass
class FrameResult:
    frame_bgr: np.ndarray
    depth: np.ndarray
    conf: np.ndarray
    points_xyz: Optional[np.ndarray]
    pin_matrix: PinMatrix
    timestamp: float
    audio_text: str = ""
    detections: List[Detection] = field(default_factory=list)
    detection_distances_m: List[float] = field(default_factory=list)
    ground_plane: Optional[GroundPlane] = None
    height_above_floor: Optional[np.ndarray] = None
    identifier_labels: Optional[List[str]] = None


_NON_OBSTACLE_CLASSES = {
    # COCO doesn't include road/sidewalk; reserved here for custom detectors that do.
    "road",
    "sidewalk",
    "floor",
    "ground",
}


def _build_obstacle_mask(
    detections: List[Detection],
    shape: Tuple[int, int],
    pad: int = 4,
) -> np.ndarray:
    """Union of detection boxes (excluding non-obstacle classes), as an H x W bool mask."""
    h, w = shape
    mask = np.zeros((h, w), dtype=bool)
    for det in detections:
        if det.cls.lower() in _NON_OBSTACLE_CLASSES:
            continue
        x1, y1, x2, y2 = det.box
        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(w, x2 + pad)
        y2 = min(h, y2 + pad)
        if x2 > x1 and y2 > y1:
            mask[y1:y2, x1:x2] = True
    return mask


def _direction_word(box: Tuple[int, int, int, int], frame_w: int) -> str:
    cx = (box[0] + box[2]) / 2
    rel = cx / max(frame_w, 1)
    if rel < 0.35:
        return "left"
    if rel > 0.65:
        return "right"
    return "ahead"


def _build_audio_text(
    pm: PinMatrix,
    params: PipelineParams,
    detections: List[Detection],
    distances: List[float],
    frame_w: int,
) -> str:
    if detections:
        # Rank detections by proximity (smallest finite distance first).
        order = sorted(
            range(len(detections)),
            key=lambda i: distances[i] if np.isfinite(distances[i]) else float("inf"),
        )
        parts: list[str] = []
        for i in order[:3]:
            d = distances[i]
            det = detections[i]
            direction = _direction_word(det.box, frame_w)
            if np.isfinite(d):
                parts.append(f"{det.cls} {d:.1f} meters {direction}")
            else:
                parts.append(f"{det.cls} {direction}")
        return ". ".join(parts) + "."
    # Fallback to pin-only narration.
    if pm.mode == "walk":
        if pm.data.max() == 0:
            return "Path clear ahead."
        rows_with = np.where(pm.data.max(axis=1) > 0)[0]
        nearest_row = int(rows_with.min())
        approx_dist_m = (nearest_row + 0.5) / pm.rows * params.region_depth_m
        return f"Obstacle {approx_dist_m:.1f} meters ahead."
    if pm.data.max() == 0:
        return ""
    return f"Closest pin level {int(pm.data.max())}."


def process_frame(
    frame_bgr: np.ndarray,
    params: PipelineParams,
    depth_estimator: _DepthLike,
    timestamp: float,
    detector: Optional[_DetectorLike] = None,
) -> FrameResult:
    dr = depth_estimator.predict(frame_bgr)

    detections: List[Detection] = []
    distances: List[float] = []
    if params.enable_detection and detector is not None:
        detections = detector.detect(frame_bgr)
        distances = [detection_distance_m(d, dr.depth_m) for d in detections]

    # Always project to 3D so we can fit the ground plane and pass points to the UI.
    points = depth_to_points(dr.depth_m, params.fx, params.fy, params.cx, params.cy)

    # Fit the floor BEFORE applying any user rotation, so the geometric assumption
    # ("floor is below the camera") holds.
    plane = None
    h_above = None
    if params.subtract_ground or params.scene_subtract_ground:
        plane = fit_ground_plane(points)
        if plane is not None:
            h_above = height_above_plane(points, plane)

    # Apply camera pitch AFTER plane fit. Rotate points and the plane normal together
    # so height_above_plane stays valid. Positive = simulate tilting the camera up.
    if abs(params.camera_pitch_deg) > 1e-3:
        theta = np.deg2rad(params.camera_pitch_deg)
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        rot = np.array(
            [[1.0, 0.0, 0.0], [0.0, cos_t, -sin_t], [0.0, sin_t, cos_t]],
            dtype=np.float32,
        )
        points = points @ rot.T
        if plane is not None:
            new_normal = (rot @ plane.normal).astype(np.float32)
            plane = GroundPlane(normal=new_normal, offset=plane.offset, inlier_count=plane.inlier_count)

    # Build a "near" weight map: 1.0 within ground_clear_radius_m, 0.0 well beyond.
    # Only points inside that radius get the floor-removal treatment, so the road
    # extending into the distance is still felt as part of the scene.
    near_weight = None
    if h_above is not None:
        depth = dr.depth_m
        r = params.ground_clear_radius_m
        f = max(params.ground_clear_falloff_m, 1e-3)
        near_weight = np.clip((r + f - depth) / f, 0.0, 1.0).astype(np.float32)

    identifier_labels: Optional[List[str]] = None

    if params.mode == "scene":
        depth_for_scene = dr.depth_m
        if params.scene_subtract_ground and h_above is not None:
            mask_floor = (h_above < 0.3) & (near_weight > 0.5)
            depth_for_scene = dr.depth_m.copy()
            depth_for_scene[mask_floor] = np.inf
        enc = SceneEncoder(
            rows=params.rows,
            cols=params.cols,
            levels=params.levels,
            near_m=params.near_m,
            far_m=params.far_m,
            auto_range=params.auto_range,
        )
        pm = enc.encode(depth_for_scene, timestamp=timestamp)
    elif params.mode == "identifier":
        id_enc = IdentifierEncoder(
            rows=params.rows,
            cols=params.cols,
            levels=params.levels,
            min_dist_m=params.identifier_min_dist_m,
            max_dist_m=params.identifier_max_dist_m,
            region_width_m=params.identifier_region_width_m,
        )
        id_res = id_enc.encode(detections, distances, points, timestamp=timestamp)
        pm = id_res.pin_matrix
        identifier_labels = id_res.labels
    else:  # walk
        enc = WalkEncoder(
            rows=params.rows,
            cols=params.cols,
            levels=params.levels,
            region_depth_m=params.region_depth_m,
            region_width_m=params.region_width_m,
            auto_region=params.walk_auto_region,
        )
        obstacle_mask = None
        if params.gate_walk_with_yolo and detections:
            obstacle_mask = _build_obstacle_mask(detections, dr.depth_m.shape)
        h_for_walk = None
        if params.subtract_ground and h_above is not None and near_weight is not None:
            h_finite = h_above[np.isfinite(h_above)]
            if h_finite.size and (h_finite.max() - h_finite.min()) > 0.5:
                h_for_walk = h_above.copy()
                far_floor = (h_above < 0.3) & (near_weight < 0.5)
                h_for_walk[far_floor] = 0.5
        pm = enc.encode(
            points,
            timestamp=timestamp,
            obstacle_mask=obstacle_mask,
            height_above_floor_map=h_for_walk,
        )

    audio_text = _build_audio_text(pm, params, detections, distances, frame_bgr.shape[1])

    return FrameResult(
        frame_bgr=frame_bgr,
        depth=dr.depth_m,
        conf=dr.conf,
        points_xyz=points,
        pin_matrix=pm,
        timestamp=timestamp,
        audio_text=audio_text,
        detections=detections,
        detection_distances_m=distances,
        ground_plane=plane,
        height_above_floor=h_above,
        identifier_labels=identifier_labels,
    )
