from __future__ import annotations
from dataclasses import dataclass
from typing import Literal, Tuple
import numpy as np

Mode = Literal["walk", "scene", "identifier"]


@dataclass
class PinMatrix:
    rows: int
    cols: int
    levels: int
    data: np.ndarray
    timestamp: float
    mode: Mode

    def __post_init__(self) -> None:
        if self.data.shape != (self.rows, self.cols):
            raise ValueError(
                f"data shape {self.data.shape} does not match (rows={self.rows}, cols={self.cols})"
            )
        if self.data.dtype != np.uint8:
            raise ValueError(f"data dtype must be uint8, got {self.data.dtype}")
        if int(self.data.max(initial=0)) >= self.levels:
            raise ValueError(
                f"data contains values >= levels ({self.levels})"
            )
        if self.mode not in ("walk", "scene", "identifier"):
            raise ValueError(f"mode must be 'walk', 'scene', or 'identifier', got {self.mode!r}")


@dataclass
class DepthResult:
    depth_m: np.ndarray
    conf: np.ndarray

    def __post_init__(self) -> None:
        if self.depth_m.shape != self.conf.shape:
            raise ValueError(
                f"depth_m shape {self.depth_m.shape} != conf shape {self.conf.shape}"
            )


@dataclass
class Detection:
    cls: str
    box: Tuple[int, int, int, int]
    score: float
