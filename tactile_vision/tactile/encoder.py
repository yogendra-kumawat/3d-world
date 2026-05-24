from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
from tactile_vision.types import PinMatrix, Detection


def _min_pool_2d(arr: np.ndarray, rows: int, cols: int) -> np.ndarray:
    """Min-pool a 2D array to shape (rows, cols). 'Closer wins' semantics."""
    h, w = arr.shape
    row_edges = np.linspace(0, h, rows + 1, dtype=int)
    col_edges = np.linspace(0, w, cols + 1, dtype=int)
    out = np.empty((rows, cols), dtype=arr.dtype)
    for i in range(rows):
        for j in range(cols):
            block = arr[row_edges[i]:row_edges[i + 1], col_edges[j]:col_edges[j + 1]]
            out[i, j] = block.min() if block.size else 0
    return out


class SceneEncoder:
    """Map a metric depth map to a tactile pin matrix where closer = taller.

    Two modes:
      - fixed: near_m / far_m are absolute meters (use when metric depth is reliable).
      - auto: clip to 5th/95th percentile of the depth in this frame (robust when the
        absolute metric scale is wrong — e.g., outdoor scene with indoor model).
    """

    def __init__(
        self,
        rows: int = 8,
        cols: int = 16,
        levels: int = 8,
        near_m: float = 0.3,
        far_m: float = 4.0,
        auto_range: bool = False,
    ) -> None:
        if near_m <= 0 or far_m <= near_m:
            raise ValueError("require 0 < near_m < far_m")
        self.rows = rows
        self.cols = cols
        self.levels = levels
        self.near_m = near_m
        self.far_m = far_m
        self.auto_range = auto_range

    def encode(self, depth_m: np.ndarray, timestamp: float) -> PinMatrix:
        # Non-finite or non-positive entries are treated as "infinitely far" so they
        # do not dominate the min-pool. (Callers can use np.inf to mask-out pixels,
        # e.g. floor pixels after ground-plane subtraction.)
        d = np.where(np.isfinite(depth_m) & (depth_m > 0), depth_m, np.float32(1e6))
        d_low = _min_pool_2d(d, self.rows, self.cols)

        if self.auto_range:
            valid = d_low[d_low > 0]
            if valid.size:
                near = float(np.percentile(valid, 5))
                far = float(np.percentile(valid, 95))
                if far - near < 1e-3:
                    far = near + 1e-3
            else:
                near, far = self.near_m, self.far_m
        else:
            near, far = self.near_m, self.far_m

        t = (far - d_low) / (far - near)
        t = np.clip(t, 0.0, 1.0)
        pins = np.round(t * (self.levels - 1)).astype(np.uint8)
        return PinMatrix(
            rows=self.rows,
            cols=self.cols,
            levels=self.levels,
            data=pins,
            timestamp=timestamp,
            mode="scene",
        )


class WalkEncoder:
    """Map a 3D point cloud to a top-down obstacle BEV pin matrix.

    Region: rectangle in front of the camera, region_width_m wide x region_depth_m deep.
    Pin height encodes obstacle height in that bin (above the floor).
    """

    GROUND_Y_M = 1.5
    MIN_OBSTACLE_HEIGHT_M = 0.3
    MAX_OBSTACLE_HEIGHT_M = 3.0

    def __init__(
        self,
        rows: int = 8,
        cols: int = 16,
        levels: int = 8,
        region_depth_m: float = 5.0,
        region_width_m: float = 2.5,
        auto_region: bool = False,
    ) -> None:
        self.rows = rows
        self.cols = cols
        self.levels = levels
        self.region_depth_m = region_depth_m
        self.region_width_m = region_width_m
        self.auto_region = auto_region

    def encode(
        self,
        points_xyz: np.ndarray,
        timestamp: float,
        obstacle_mask: np.ndarray | None = None,
        height_above_floor_map: np.ndarray | None = None,
    ) -> PinMatrix:
        """Encode point cloud to pin matrix.

        If obstacle_mask is given (H, W bool), only points whose source pixel is
        masked True contribute to pins. If height_above_floor_map is given (H, W
        float meters), it's used as the obstacle-height signal instead of the
        naive `GROUND_Y_M - y`. Pass it from a fitted ground plane to ignore the
        road properly.
        """
        x_all = points_xyz[..., 0].ravel()
        y_all = points_xyz[..., 1].ravel()
        z_all = points_xyz[..., 2].ravel()

        if obstacle_mask is not None:
            mask_flat = obstacle_mask.ravel().astype(bool)
        else:
            mask_flat = np.ones_like(z_all, dtype=bool)

        # Auto-fit the BEV region to the actual point distribution if requested.
        # This makes the encoder robust to scenes where the depth scale is wrong
        # (e.g., indoor model on outdoor scene) — fixed 2.5x5m boxes would otherwise
        # contain zero points.
        if self.auto_region:
            finite = np.isfinite(z_all) & (z_all > 0.1)
            if finite.any():
                z_pos = z_all[finite]
                x_pos = x_all[finite]
                region_depth_m = max(float(np.percentile(z_pos, 95)), 1.0)
                region_width_m = 2.0 * max(float(np.percentile(np.abs(x_pos), 95)), 0.5)
            else:
                region_depth_m = self.region_depth_m
                region_width_m = self.region_width_m
        else:
            region_depth_m = self.region_depth_m
            region_width_m = self.region_width_m

        in_z = (z_all > 0.1) & (z_all <= region_depth_m)
        in_x = np.abs(x_all) <= (region_width_m / 2)
        valid = in_z & in_x & np.isfinite(z_all) & mask_flat
        x, y, z = x_all[valid], y_all[valid], z_all[valid]
        # Stash effective region for downstream binning.
        self._effective_region_depth_m = region_depth_m
        self._effective_region_width_m = region_width_m

        if height_above_floor_map is not None:
            h_all = height_above_floor_map.ravel()
            height_above_floor = h_all[valid]
        else:
            height_above_floor = self.GROUND_Y_M - y
        keep = height_above_floor >= self.MIN_OBSTACLE_HEIGHT_M
        x, z, h = x[keep], z[keep], height_above_floor[keep]

        pins = np.zeros((self.rows, self.cols), dtype=np.float32)
        if x.size:
            col_idx = np.clip(
                ((x + self._effective_region_width_m / 2) / self._effective_region_width_m * self.cols).astype(int),
                0,
                self.cols - 1,
            )
            row_idx = np.clip(
                ((z / self._effective_region_depth_m) * self.rows).astype(int),
                0,
                self.rows - 1,
            )
            np.maximum.at(pins, (row_idx, col_idx), h)

        denom = self.MAX_OBSTACLE_HEIGHT_M - self.MIN_OBSTACLE_HEIGHT_M
        pins = (pins - self.MIN_OBSTACLE_HEIGHT_M) / denom
        pins = np.clip(pins, 0.0, 1.0)
        pin_data = np.round(pins * (self.levels - 1)).astype(np.uint8)

        return PinMatrix(
            rows=self.rows,
            cols=self.cols,
            levels=self.levels,
            data=pin_data,
            timestamp=timestamp,
            mode="walk",
        )


@dataclass
class IdentifierResult:
    """Tactile pin matrix annotated with which class label sits in which cell.
    Returned by IdentifierEncoder so the UI can label the 3D bar chart."""
    pin_matrix: PinMatrix
    labels: List[str]   # length rows*cols, row-major; "" for empty cells


class IdentifierEncoder:
    """Per-object tactile encoder.

    Each YOLO detection becomes a single pin in the matrix:
      - column = horizontal position of the detection's bbox center, mapped to [-W/2, W/2]
      - row = forward distance bin (closer detections at the bottom of the matrix)
      - pin height = (max_dist - distance) / (max_dist - min_dist), so closer = taller
    Multiple detections can land in the same cell; the closest wins, and the
    closest detection's class label is reported for that cell.
    """

    def __init__(
        self,
        rows: int = 8,
        cols: int = 16,
        levels: int = 8,
        min_dist_m: float = 0.5,
        max_dist_m: float = 15.0,
        region_width_m: float = 6.0,
    ) -> None:
        self.rows = rows
        self.cols = cols
        self.levels = levels
        self.min_dist_m = min_dist_m
        self.max_dist_m = max(max_dist_m, min_dist_m + 0.1)
        self.region_width_m = region_width_m

    def encode(
        self,
        detections: List[Detection],
        distances_m: List[float],
        points_xyz: np.ndarray,
        timestamp: float,
    ) -> IdentifierResult:
        rows, cols = self.rows, self.cols
        data = np.zeros((rows, cols), dtype=np.uint8)
        labels: List[str] = [""] * (rows * cols)
        # Track per-cell minimum distance so we know who "won" each cell.
        cell_dist = np.full((rows, cols), np.inf, dtype=np.float32)

        if not detections:
            return IdentifierResult(
                pin_matrix=PinMatrix(rows=rows, cols=cols, levels=self.levels, data=data, timestamp=timestamp, mode="identifier"),
                labels=labels,
            )

        h_img, w_img = points_xyz.shape[:2]

        for det, d in zip(detections, distances_m):
            if not np.isfinite(d) or d <= 0:
                continue
            x1, y1, x2, y2 = det.box
            cx_px = int((x1 + x2) / 2)
            cy_px = int((y1 + y2) / 2)
            cx_px = max(0, min(w_img - 1, cx_px))
            cy_px = max(0, min(h_img - 1, cy_px))
            world_xyz = points_xyz[cy_px, cx_px]
            if not np.isfinite(world_xyz).all():
                # Fall back to box-center direction in normalised image coords.
                rel = (cx_px / w_img) - 0.5
            else:
                rel = float(world_xyz[0]) / max(self.region_width_m / 2, 1e-3)
                rel = max(-1.0, min(1.0, rel))

            col = int(round((rel + 1.0) / 2.0 * (cols - 1)))
            col = max(0, min(cols - 1, col))

            # Distance -> row index. Closer detections go to the bottom row
            # (row index rows-1), farther go to the top (row index 0).
            t = (d - self.min_dist_m) / (self.max_dist_m - self.min_dist_m)
            t = max(0.0, min(1.0, t))
            row = int(round(t * (rows - 1)))   # t=0 (close) -> row 0; reverse below
            row = (rows - 1) - row              # close -> bottom
            row = max(0, min(rows - 1, row))

            # Closer wins for that cell.
            if d < cell_dist[row, col]:
                pin_value = int(round((1.0 - t) * (self.levels - 1)))
                pin_value = max(0, min(self.levels - 1, pin_value))
                data[row, col] = pin_value
                cell_dist[row, col] = d
                labels[row * cols + col] = det.cls

        return IdentifierResult(
            pin_matrix=PinMatrix(rows=rows, cols=cols, levels=self.levels, data=data, timestamp=timestamp, mode="identifier"),
            labels=labels,
        )
