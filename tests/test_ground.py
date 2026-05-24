import numpy as np
from tactile_vision.geometry.ground import fit_ground_plane, height_above_plane


def _synth_floor_and_pillar(h=240, w=320):
    """Synthesize a point cloud: floor at y=1.5m below camera, plus a 1.7m-tall pillar
    centered at z=3m. Returns (points_xyz, expected pillar pixel mask)."""
    # Floor: y = 1.5 (camera is 1.5m above the floor in OpenCV +y-down).
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    fx = fy = 320.0
    cx, cy_ = w / 2, h / 2
    # Make z range from 1m at top to 8m at bottom for a forward-tilted floor view.
    z_floor = 1.0 + (yy / h) * 7.0
    pts_floor = np.stack(
        [(xx - cx) * z_floor / fx, (yy - cy_) * z_floor / fy, z_floor],
        axis=-1,
    ).astype(np.float32)

    # Pillar: vertical column at z=3m, from y=-0.2m to y=1.5m (1.7m tall).
    pillar_mask = np.zeros((h, w), dtype=bool)
    pillar_mask[60:200, 140:180] = True
    z_pillar = np.full((h, w), 3.0, dtype=np.float32)
    pts = pts_floor.copy()
    pts[pillar_mask] = np.stack(
        [
            (xx[pillar_mask] - cx) * z_pillar[pillar_mask] / fx,
            (yy[pillar_mask] - cy_) * z_pillar[pillar_mask] / fy,
            z_pillar[pillar_mask],
        ],
        axis=-1,
    )
    return pts, pillar_mask


def test_floor_normal_points_up():
    pts, _ = _synth_floor_and_pillar()
    plane = fit_ground_plane(pts)
    assert plane is not None
    # In OpenCV, "up" = -y. Normal should have a strongly negative y component.
    assert plane.normal[1] < -0.6


def test_height_above_plane_zero_on_floor_positive_on_pillar():
    pts, pillar_mask = _synth_floor_and_pillar()
    plane = fit_ground_plane(pts)
    assert plane is not None
    h = height_above_plane(pts, plane)
    # Floor band (very bottom rows that aren't the pillar) should be near zero.
    floor_band = h[-30:, :100]
    assert np.median(np.abs(floor_band)) < 0.2
    # Pillar pixels should be clearly above the floor.
    assert np.median(h[pillar_mask]) > 0.5


def test_returns_none_when_too_few_points():
    pts = np.full((4, 4, 3), np.nan, dtype=np.float32)
    plane = fit_ground_plane(pts)
    assert plane is None
