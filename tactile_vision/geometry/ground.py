"""Ground-plane fit + height-above-floor extraction.

Given a point cloud (H, W, 3) in camera frame (OpenCV: +y down, +z forward),
fit the dominant floor plane using points from the lower half of the image and
return per-pixel height above that plane. Vectorized, no RANSAC, ~1 ms per frame.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class GroundPlane:
    normal: np.ndarray  # shape (3,), unit
    offset: float       # plane: dot(normal, X) + offset = 0
    inlier_count: int


def fit_ground_plane(
    points_xyz: np.ndarray,
    sample_lower_fraction: float = 0.20,
    iterations: int = 3,
    inlier_thresh_m: float = 0.10,
) -> GroundPlane | None:
    """Robust least-squares plane fit on the lower portion of the point cloud.

    Strategy: take points from the bottom `sample_lower_fraction` of rows (those
    are most likely floor in a forward-facing camera), do an iterated reweighted
    LS fit. Each iteration trims outliers by `inlier_thresh_m`.

    Multiple sanity checks reject fits that grabbed buildings/walls/sky:
      - normal must be mostly vertical (|n_y| > 0.6)
      - the inlier band's centroid must lie below the camera (centroid_y > 0)
      - over the whole image, the lower half must mostly be at h >= 0
    Returns None if any check fails — caller should treat that as "no ground info".
    """
    h, w, _ = points_xyz.shape
    start_row = int(h * (1.0 - sample_lower_fraction))
    band = points_xyz[start_row:].reshape(-1, 3)
    band = band[np.isfinite(band).all(axis=1)]
    band = band[band[:, 2] > 0.1]
    if band.shape[0] < 100:
        return None

    inliers = band
    for _ in range(iterations):
        if inliers.shape[0] < 50:
            break
        centroid = inliers.mean(axis=0)
        centered = inliers - centroid
        if centered.shape[0] > 8000:
            idx = np.random.default_rng(0).choice(centered.shape[0], 8000, replace=False)
            sub = centered[idx]
        else:
            sub = centered
        _, _, vh = np.linalg.svd(sub, full_matrices=False)
        normal = vh[-1]
        normal = normal / (np.linalg.norm(normal) + 1e-12)
        if normal[1] > 0:
            normal = -normal
        offset = -normal @ centroid
        dist = np.abs(band @ normal + offset)
        inliers = band[dist < inlier_thresh_m]

    if inliers.shape[0] < 50:
        return None

    # Sanity check #1: normal must be mostly along -y (vertical).
    if abs(normal[1]) < 0.6:
        return None

    # Sanity check #2: centroid of inliers must be below the camera (y > 0 in OpenCV).
    inlier_centroid = inliers.mean(axis=0)
    if inlier_centroid[1] < 0.1:  # too close to camera height — probably a wall
        return None

    # Sanity check #3: project the whole image and require the lower half to mostly
    # have h >= 0 (the plane should pass through the visible floor, not above it).
    pts_full = points_xyz.reshape(-1, 3)
    h_full = -(pts_full @ normal + offset)
    h_full = h_full.reshape(h, w)
    lower_half = h_full[h // 2 :]
    lower_finite = lower_half[np.isfinite(lower_half)]
    if lower_finite.size == 0:
        return None
    frac_above = float((lower_finite >= -inlier_thresh_m).sum()) / lower_finite.size
    if frac_above < 0.6:
        # Plane is sitting above the floor (looking at red-blob-of-road symptom).
        return None

    return GroundPlane(normal=normal.astype(np.float32), offset=float(offset), inlier_count=int(inliers.shape[0]))


def height_above_plane(points_xyz: np.ndarray, plane: GroundPlane) -> np.ndarray:
    """Per-pixel signed height above the plane (meters). Positive = above floor."""
    pts = points_xyz.astype(np.float32)
    # Signed distance with the up-pointing convention.
    return -(pts @ plane.normal + plane.offset)
