from __future__ import annotations
import numpy as np


def depth_to_points(
    depth_m: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
) -> np.ndarray:
    """Back-project a metric depth map to a per-pixel 3D point cloud.

    Camera convention: +x right, +y down, +z forward (OpenCV).
    Returns (H, W, 3) float32.
    """
    h, w = depth_m.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    z = depth_m.astype(np.float32)
    x = (xx - cx) * z / fx
    y = (yy - cy) * z / fy
    return np.stack([x, y, z], axis=-1)
