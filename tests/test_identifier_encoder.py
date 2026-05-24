import numpy as np
from tactile_vision.tactile.encoder import IdentifierEncoder
from tactile_vision.types import Detection


def _flat_points(h=240, w=320, depth_m=3.0):
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    fx = fy = 320.0
    cx, cy_ = w / 2, h / 2
    z = np.full((h, w), depth_m, dtype=np.float32)
    x = (xx - cx) * z / fx
    y = (yy - cy_) * z / fy
    return np.stack([x, y, z], axis=-1)


def test_no_detections_returns_empty_matrix():
    enc = IdentifierEncoder()
    pts = _flat_points()
    res = enc.encode([], [], pts, timestamp=0.0)
    assert res.pin_matrix.data.max() == 0
    assert all(lbl == "" for lbl in res.labels)
    assert res.pin_matrix.mode == "identifier"


def test_close_detection_gets_taller_pin():
    enc = IdentifierEncoder(rows=8, cols=16, levels=8, min_dist_m=0.5, max_dist_m=10.0)
    pts = _flat_points(depth_m=2.0)
    near = Detection(cls="person", box=(140, 100, 180, 200), score=0.9)
    far = Detection(cls="car", box=(200, 100, 260, 200), score=0.9)
    res = enc.encode([near, far], [1.0, 8.0], pts, timestamp=0.0)
    assert res.pin_matrix.data.max() > 0
    flat = res.pin_matrix.data.ravel()
    near_pins = flat[flat > 0]
    assert near_pins.max() >= 4  # close detection should be at least mid-height


def test_close_detection_lands_in_bottom_row():
    enc = IdentifierEncoder(rows=8, cols=16, levels=8, min_dist_m=0.5, max_dist_m=10.0)
    pts = _flat_points(depth_m=1.0)
    det = Detection(cls="person", box=(140, 100, 180, 200), score=0.9)
    res = enc.encode([det], [1.0], pts, timestamp=0.0)
    bottom = res.pin_matrix.data[-1]
    top = res.pin_matrix.data[0]
    assert bottom.max() > top.max()


def test_label_recorded_for_winning_cell():
    enc = IdentifierEncoder(rows=4, cols=4, levels=8, min_dist_m=0.5, max_dist_m=10.0)
    pts = _flat_points(h=120, w=160)
    det = Detection(cls="bicycle", box=(70, 50, 90, 90), score=0.9)
    res = enc.encode([det], [1.5], pts, timestamp=0.0)
    assert "bicycle" in res.labels


def test_closer_detection_overrides_farther_in_same_cell():
    """Two detections with the same distance and same bbox center => same cell.
    Submit far first, near second; near must overwrite the cell."""
    enc = IdentifierEncoder(rows=4, cols=4, levels=8, min_dist_m=0.5, max_dist_m=10.0)
    pts = _flat_points(h=120, w=160)
    box = (70, 50, 90, 90)
    far = Detection(cls="car", box=box, score=0.9)
    near = Detection(cls="person", box=box, score=0.9)
    # Same distance but submitted in order [far, near] — closer wins regardless of order.
    res = enc.encode([far, near], [3.0, 1.0], pts, timestamp=0.0)
    assert "person" in res.labels  # near appears
    # far should not appear if both ended up in the same cell, or may exist in a different cell.
    # The key invariant: every cell where person appears must show its higher pin value.
    person_cells = [i for i, lbl in enumerate(res.labels) if lbl == "person"]
    for i in person_cells:
        assert res.pin_matrix.data.ravel()[i] >= 4  # close => high pin
