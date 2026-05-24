"""Synthetic depth maps and frames for deterministic tests."""
import numpy as np


def flat_wall_depth(h: int = 240, w: int = 320, distance_m: float = 2.0) -> np.ndarray:
    return np.full((h, w), distance_m, dtype=np.float32)


def empty_room_depth(h: int = 240, w: int = 320, far_m: float = 5.0) -> np.ndarray:
    rows = np.linspace(far_m, 1.0, h, dtype=np.float32)
    return np.tile(rows[:, None], (1, w))


def corridor_depth(h: int = 240, w: int = 320, near_m: float = 1.0, far_m: float = 8.0) -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w]
    cy, cx = h / 2, w / 2
    r = np.hypot(yy - cy, xx - cx)
    r_norm = r / r.max()
    return (far_m - (far_m - near_m) * r_norm).astype(np.float32)


def obstacle_depth(h: int = 240, w: int = 320, far_m: float = 5.0, obstacle_m: float = 1.5) -> np.ndarray:
    d = np.full((h, w), far_m, dtype=np.float32)
    cy, cx = h // 2, w // 2
    d[cy - 30 : cy + 30, cx - 30 : cx + 30] = obstacle_m
    return d


def dummy_bgr_frame(h: int = 240, w: int = 320) -> np.ndarray:
    rng = np.random.default_rng(42)
    return rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)
