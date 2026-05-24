import numpy as np
import pytest
from tactile_vision.pipeline import process_frame, PipelineParams
from tactile_vision.types import DepthResult


class _DepthFromArray:
    def __init__(self, depth):
        self.depth = depth.astype(np.float32)

    def predict(self, frame_bgr):
        d = self.depth
        return DepthResult(depth_m=d, conf=np.ones_like(d))


def _perspective_floor_depth(h=240, w=320, fx=320.0, fy=320.0, camera_height_m=1.5, max_depth_m=30.0):
    """Synthetic perspective view of a flat floor, camera_height_m above it,
    looking straight ahead. Returns (H, W) metric depth."""
    cx, cy = w / 2.0, h / 2.0
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    v = yy - cy  # +ve below the horizon
    # For a downward-facing pixel below the horizon, depth = camera_height * fy / v.
    # Above horizon: no floor visible -> use max_depth.
    depth = np.full((h, w), max_depth_m, dtype=np.float32)
    below = v > 1
    depth[below] = camera_height_m * fy / v[below]
    np.clip(depth, 0.1, max_depth_m, out=depth)
    return depth


def test_radial_gating_keeps_far_road_visible_in_scene_mode():
    depth = _perspective_floor_depth()
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    params = PipelineParams(
        mode="scene",
        fx=320.0, fy=320.0, cx=160.0, cy=120.0,
        scene_subtract_ground=True,
        ground_clear_radius_m=3.0,
        ground_clear_falloff_m=1.0,
        auto_range=True,
    )
    result = process_frame(frame, params=params, depth_estimator=_DepthFromArray(depth), timestamp=0.0)
    # Far road (depth > radius) should still show up somewhere — not pure zeros.
    assert result.pin_matrix.data.max() > 0


def test_radial_gating_silences_near_road_in_scene_mode():
    # Push the camera close so the entire visible floor is within the 3m radius.
    depth = _perspective_floor_depth(camera_height_m=0.3, max_depth_m=2.5)
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    params = PipelineParams(
        mode="scene",
        fx=320.0, fy=320.0, cx=160.0, cy=120.0,
        scene_subtract_ground=True,
        ground_clear_radius_m=5.0,  # bigger than max_depth so all pixels are "near"
        ground_clear_falloff_m=0.5,
        auto_range=False,
        near_m=0.1, far_m=5.0,
    )
    result = process_frame(frame, params=params, depth_estimator=_DepthFromArray(depth), timestamp=0.0)
    # Within the radius, pure-floor scene -> pin matrix should be empty (all zero or all flat).
    assert result.pin_matrix.data.max() == 0
