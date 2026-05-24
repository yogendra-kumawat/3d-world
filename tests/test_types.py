import numpy as np
import pytest
from tactile_vision.types import PinMatrix, DepthResult, Detection


def test_pinmatrix_validates_shape():
    data = np.zeros((8, 16), dtype=np.uint8)
    pm = PinMatrix(rows=8, cols=16, levels=8, data=data, timestamp=0.0, mode="walk")
    assert pm.data.shape == (8, 16)


def test_pinmatrix_rejects_wrong_shape():
    bad = np.zeros((4, 4), dtype=np.uint8)
    with pytest.raises(ValueError):
        PinMatrix(rows=8, cols=16, levels=8, data=bad, timestamp=0.0, mode="walk")


def test_pinmatrix_rejects_out_of_range_values():
    bad = np.full((8, 16), 99, dtype=np.uint8)
    with pytest.raises(ValueError):
        PinMatrix(rows=8, cols=16, levels=8, data=bad, timestamp=0.0, mode="walk")


def test_pinmatrix_rejects_invalid_mode():
    data = np.zeros((8, 16), dtype=np.uint8)
    with pytest.raises(ValueError):
        PinMatrix(rows=8, cols=16, levels=8, data=data, timestamp=0.0, mode="bogus")


def test_depthresult_shapes_must_match():
    d = np.zeros((10, 20), dtype=np.float32)
    c = np.zeros((10, 20), dtype=np.float32)
    r = DepthResult(depth_m=d, conf=c)
    assert r.depth_m.shape == r.conf.shape


def test_depthresult_rejects_mismatched_shapes():
    d = np.zeros((10, 20), dtype=np.float32)
    c = np.zeros((5, 5), dtype=np.float32)
    with pytest.raises(ValueError):
        DepthResult(depth_m=d, conf=c)


def test_detection_box_format():
    d = Detection(cls="chair", box=(10, 20, 50, 80), score=0.9)
    assert d.box == (10, 20, 50, 80)
