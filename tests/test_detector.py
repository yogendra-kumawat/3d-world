import numpy as np
from tactile_vision.detection.detector import detection_distance_m
from tactile_vision.types import Detection


def test_median_depth_in_box():
    depth = np.full((100, 100), 5.0, dtype=np.float32)
    depth[40:60, 40:60] = 1.5
    det = Detection(cls="chair", box=(40, 40, 60, 60), score=0.9)
    assert detection_distance_m(det, depth) == 1.5


def test_box_with_invalid_depth_returns_nan():
    depth = np.zeros((50, 50), dtype=np.float32)
    det = Detection(cls="chair", box=(10, 10, 20, 20), score=0.9)
    assert np.isnan(detection_distance_m(det, depth))


def test_clamps_box_to_image_bounds():
    depth = np.full((50, 50), 2.0, dtype=np.float32)
    det = Detection(cls="chair", box=(-5, -5, 200, 200), score=0.9)
    assert detection_distance_m(det, depth) == 2.0


def test_inverted_or_zero_box_returns_nan():
    depth = np.full((50, 50), 2.0, dtype=np.float32)
    det = Detection(cls="chair", box=(30, 30, 30, 30), score=0.9)
    assert np.isnan(detection_distance_m(det, depth))
