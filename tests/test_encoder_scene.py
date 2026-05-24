import numpy as np
from tactile_vision.tactile.encoder import SceneEncoder
from tactile_vision.types import PinMatrix
from tests.fixtures.synthetic import flat_wall_depth, obstacle_depth, corridor_depth


def test_returns_pinmatrix_with_correct_shape():
    enc = SceneEncoder(rows=8, cols=16, levels=8, near_m=0.3, far_m=4.0)
    pm = enc.encode(flat_wall_depth(distance_m=2.0), timestamp=1.0)
    assert isinstance(pm, PinMatrix)
    assert pm.data.shape == (8, 16)
    assert pm.mode == "scene"
    assert pm.timestamp == 1.0


def test_uniform_wall_gives_uniform_pins():
    enc = SceneEncoder(rows=8, cols=16, levels=8, near_m=0.3, far_m=4.0)
    pm = enc.encode(flat_wall_depth(distance_m=2.0), timestamp=0.0)
    assert pm.data.min() == pm.data.max()


def test_closer_means_taller_pins():
    enc = SceneEncoder(rows=8, cols=16, levels=8, near_m=0.3, far_m=4.0)
    near = enc.encode(flat_wall_depth(distance_m=0.5), timestamp=0.0)
    far = enc.encode(flat_wall_depth(distance_m=3.5), timestamp=0.0)
    assert near.data.mean() > far.data.mean()


def test_obstacle_creates_local_peak():
    enc = SceneEncoder(rows=8, cols=16, levels=8, near_m=0.3, far_m=5.0)
    pm = enc.encode(obstacle_depth(far_m=5.0, obstacle_m=1.0), timestamp=0.0)
    cy, cx = pm.data.shape[0] // 2, pm.data.shape[1] // 2
    assert pm.data[cy, cx] >= pm.data.max() - 1


def test_values_within_levels():
    enc = SceneEncoder(rows=8, cols=16, levels=8, near_m=0.3, far_m=5.0)
    pm = enc.encode(corridor_depth(), timestamp=0.0)
    assert pm.data.min() >= 0
    assert pm.data.max() < 8


def test_zero_depth_treated_as_far():
    """Pixels with no depth info should map to pin level 0 (no obstacle), not max.
    This is the inverse of the old behaviour: it lets callers mask pixels by
    setting them to 0/inf without falsely raising pins."""
    enc = SceneEncoder(rows=4, cols=4, levels=8, near_m=0.3, far_m=5.0)
    d = np.zeros((40, 40), dtype=np.float32)
    pm = enc.encode(d, timestamp=0.0)
    assert (pm.data == 0).all()


def test_inf_depth_treated_as_far():
    enc = SceneEncoder(rows=4, cols=4, levels=8, near_m=0.3, far_m=5.0)
    d = np.full((40, 40), np.inf, dtype=np.float32)
    pm = enc.encode(d, timestamp=0.0)
    assert (pm.data == 0).all()
