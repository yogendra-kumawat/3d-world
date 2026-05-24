import numpy as np
from tactile_vision.tactile.encoder import WalkEncoder
from tactile_vision.geometry.pointcloud import depth_to_points
from tactile_vision.types import PinMatrix
from tests.fixtures.synthetic import flat_wall_depth, obstacle_depth


def _points_from_depth(depth):
    h, w = depth.shape
    return depth_to_points(depth, fx=w, fy=w, cx=w / 2, cy=h / 2)


def test_returns_walk_pinmatrix():
    enc = WalkEncoder(rows=8, cols=16, levels=8)
    depth = flat_wall_depth(distance_m=2.0)
    pts = _points_from_depth(depth)
    pm = enc.encode(pts, timestamp=2.0)
    assert isinstance(pm, PinMatrix)
    assert pm.mode == "walk"
    assert pm.data.shape == (8, 16)
    assert pm.timestamp == 2.0


def test_empty_scene_pins_all_low():
    enc = WalkEncoder(rows=8, cols=16, levels=8, region_depth_m=5.0, region_width_m=2.5)
    depth = flat_wall_depth(distance_m=20.0)
    pts = _points_from_depth(depth)
    pm = enc.encode(pts, timestamp=0.0)
    assert pm.data.max() == 0


def test_close_obstacle_raises_pins():
    enc = WalkEncoder(rows=8, cols=16, levels=8, region_depth_m=5.0, region_width_m=2.5)
    pts = _points_from_depth(obstacle_depth(far_m=10.0, obstacle_m=1.5))
    pm = enc.encode(pts, timestamp=0.0)
    assert pm.data.max() > 0


def test_values_within_levels():
    enc = WalkEncoder(rows=8, cols=16, levels=8)
    pts = _points_from_depth(obstacle_depth(far_m=8.0, obstacle_m=1.0))
    pm = enc.encode(pts, timestamp=0.0)
    assert pm.data.min() >= 0
    assert pm.data.max() < 8
