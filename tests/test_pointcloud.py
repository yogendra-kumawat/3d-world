import numpy as np
from tactile_vision.geometry.pointcloud import depth_to_points
from tests.fixtures.synthetic import flat_wall_depth


def test_shape_matches_input():
    depth = flat_wall_depth(h=10, w=20, distance_m=3.0)
    pts = depth_to_points(depth, fx=100, fy=100, cx=10, cy=5)
    assert pts.shape == (10, 20, 3)
    assert pts.dtype == np.float32


def test_principal_point_has_zero_xy():
    depth = flat_wall_depth(h=11, w=21, distance_m=4.0)
    pts = depth_to_points(depth, fx=100, fy=100, cx=10, cy=5)
    assert np.isclose(pts[5, 10, 0], 0.0)
    assert np.isclose(pts[5, 10, 1], 0.0)
    assert np.isclose(pts[5, 10, 2], 4.0)


def test_z_equals_depth_everywhere():
    depth = flat_wall_depth(h=8, w=8, distance_m=2.5)
    pts = depth_to_points(depth, fx=50, fy=50, cx=4, cy=4)
    assert np.allclose(pts[..., 2], 2.5)


def test_x_grows_to_the_right():
    depth = flat_wall_depth(h=4, w=4, distance_m=1.0)
    pts = depth_to_points(depth, fx=10, fy=10, cx=2, cy=2)
    assert pts[2, 3, 0] > pts[2, 0, 0]
