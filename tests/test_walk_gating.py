import numpy as np
from tactile_vision.pipeline import _build_obstacle_mask
from tactile_vision.tactile.encoder import WalkEncoder
from tactile_vision.geometry.pointcloud import depth_to_points
from tactile_vision.types import Detection
from tests.fixtures.synthetic import obstacle_depth, flat_wall_depth


def test_obstacle_mask_unions_boxes():
    dets = [
        Detection(cls="person", box=(10, 20, 40, 60), score=0.9),
        Detection(cls="car", box=(100, 50, 150, 100), score=0.9),
    ]
    mask = _build_obstacle_mask(dets, (200, 300), pad=0)
    assert mask[30, 25]      # inside person box
    assert mask[75, 125]     # inside car box
    assert not mask[80, 250]  # outside both


def test_obstacle_mask_skips_non_obstacle_classes():
    dets = [
        Detection(cls="road", box=(0, 0, 100, 100), score=0.9),
    ]
    mask = _build_obstacle_mask(dets, (200, 300), pad=0)
    assert not mask.any()


def test_walk_encoder_without_mask_raises_pins_for_close_obstacle():
    depth = obstacle_depth(h=240, w=320, far_m=10.0, obstacle_m=1.5)
    pts = depth_to_points(depth, fx=320, fy=320, cx=160, cy=120)
    enc = WalkEncoder(rows=8, cols=16, levels=8)
    pm = enc.encode(pts, timestamp=0.0)
    assert pm.data.max() > 0


def test_walk_encoder_with_empty_mask_keeps_pins_flat():
    depth = obstacle_depth(h=240, w=320, far_m=10.0, obstacle_m=1.5)
    pts = depth_to_points(depth, fx=320, fy=320, cx=160, cy=120)
    enc = WalkEncoder(rows=8, cols=16, levels=8)
    empty_mask = np.zeros(depth.shape, dtype=bool)
    pm = enc.encode(pts, timestamp=0.0, obstacle_mask=empty_mask)
    assert pm.data.max() == 0


def test_walk_encoder_with_mask_only_uses_masked_pixels():
    """Two close obstacles in the depth map; mask only covers one of them.
    The encoder should only raise pins corresponding to the masked one."""
    h, w = 240, 320
    depth = np.full((h, w), 10.0, dtype=np.float32)
    # Left obstacle at (~80, 120), right obstacle at (~240, 120)
    depth[100:140, 60:100] = 1.5  # left
    depth[100:140, 220:260] = 1.5  # right
    pts = depth_to_points(depth, fx=w, fy=w, cx=w / 2, cy=h / 2)

    mask_only_right = np.zeros((h, w), dtype=bool)
    mask_only_right[100:140, 220:260] = True

    enc = WalkEncoder(rows=8, cols=16, levels=8, region_width_m=5.0, region_depth_m=5.0)
    pm = enc.encode(pts, timestamp=0.0, obstacle_mask=mask_only_right)
    # The right half of the BEV grid should have raised pins; the left should be flat.
    left_max = pm.data[:, : pm.cols // 2].max()
    right_max = pm.data[:, pm.cols // 2 :].max()
    assert right_max > 0
    assert left_max == 0
