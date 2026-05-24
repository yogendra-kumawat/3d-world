import numpy as np
from tactile_vision.pipeline import process_frame, PipelineParams
from tactile_vision.types import DepthResult, PinMatrix
from tests.fixtures.synthetic import obstacle_depth, dummy_bgr_frame


class FakeDepthEstimator:
    def predict(self, frame_bgr):
        depth = obstacle_depth(h=frame_bgr.shape[0], w=frame_bgr.shape[1], far_m=8.0, obstacle_m=1.5)
        conf = np.ones_like(depth)
        return DepthResult(depth_m=depth, conf=conf)


def test_process_frame_returns_pin_matrix_in_scene_mode():
    frame = dummy_bgr_frame(h=240, w=320)
    params = PipelineParams(mode="scene", rows=8, cols=16, levels=8)
    result = process_frame(frame, params=params, depth_estimator=FakeDepthEstimator(), timestamp=0.0)
    assert isinstance(result.pin_matrix, PinMatrix)
    assert result.pin_matrix.mode == "scene"
    assert result.pin_matrix.data.shape == (8, 16)


def test_process_frame_returns_pin_matrix_in_walk_mode():
    frame = dummy_bgr_frame(h=240, w=320)
    params = PipelineParams(mode="walk", rows=8, cols=16, levels=8, fx=320, fy=320, cx=160, cy=120)
    result = process_frame(frame, params=params, depth_estimator=FakeDepthEstimator(), timestamp=0.0)
    assert result.pin_matrix.mode == "walk"


def test_process_frame_returns_intermediates():
    frame = dummy_bgr_frame(h=240, w=320)
    params = PipelineParams(mode="scene")
    result = process_frame(frame, params=params, depth_estimator=FakeDepthEstimator(), timestamp=1.0)
    assert result.depth.shape == (240, 320)
    assert result.frame_bgr is frame
    assert result.timestamp == 1.0
