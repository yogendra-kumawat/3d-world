from __future__ import annotations
from typing import Literal, Optional, List
import numpy as np

from tactile_vision.types import DepthResult

ModelSize = Literal["small", "base", "large"]
Domain = Literal["indoor", "outdoor"]

_HF_IDS = {
    ("small", "indoor"): "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf",
    ("base", "indoor"): "depth-anything/Depth-Anything-V2-Metric-Indoor-Base-hf",
    ("large", "indoor"): "depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf",
    ("small", "outdoor"): "depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf",
    ("base", "outdoor"): "depth-anything/Depth-Anything-V2-Metric-Outdoor-Base-hf",
    ("large", "outdoor"): "depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf",
}


class DepthEstimator:
    def __init__(
        self,
        model_size: ModelSize = "base",
        domain: Domain = "indoor",
        device: Optional[str] = None,
        infer_size: int = 392,
    ) -> None:
        import torch
        import cv2
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self._torch = torch
        self._cv2 = cv2
        self.infer_size = infer_size

        hf_id = _HF_IDS[(model_size, domain)]
        self.processor = AutoImageProcessor.from_pretrained(hf_id)
        self.model = AutoModelForDepthEstimation.from_pretrained(hf_id).to(device)
        if device == "cuda":
            self.model = self.model.half()
        self.model.eval()
        self._prev_depths: List[np.ndarray] = []

    def predict(self, frame_bgr: np.ndarray) -> DepthResult:
        torch = self._torch
        cv2 = self._cv2
        h_orig, w_orig = frame_bgr.shape[:2]
        # Downscale longest side to infer_size for ~Nx faster inference; depth is upsampled below.
        scale = self.infer_size / max(h_orig, w_orig)
        if scale < 1.0:
            new_w = max(1, int(round(w_orig * scale)))
            new_h = max(1, int(round(h_orig * scale)))
            small_bgr = cv2.resize(frame_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
        else:
            small_bgr = frame_bgr
        with torch.inference_mode():
            rgb = small_bgr[..., ::-1].copy()
            inputs = self.processor(images=rgb, return_tensors="pt").to(self.device)
            if self.device == "cuda":
                inputs = {
                    k: (v.half() if v.dtype == torch.float32 else v)
                    for k, v in inputs.items()
                }
            outputs = self.model(**inputs)
            depth = (
                torch.nn.functional.interpolate(
                    outputs.predicted_depth.unsqueeze(1).float(),
                    size=(h_orig, w_orig),
                    mode="bicubic",
                    align_corners=False,
                )
                .squeeze(1)
                .squeeze(0)
                .cpu()
                .numpy()
                .astype(np.float32)
            )

        conf = self._estimate_confidence(depth)
        return DepthResult(depth_m=depth, conf=conf)

    def _estimate_confidence(self, depth: np.ndarray) -> np.ndarray:
        gx = np.abs(np.gradient(depth, axis=1))
        gy = np.abs(np.gradient(depth, axis=0))
        grad = gx + gy
        grad_norm = grad / (grad.max() + 1e-6)
        gradient_conf = 1.0 - np.clip(grad_norm * 5.0, 0.0, 1.0)

        if self._prev_depths:
            temporal = np.mean(
                [np.abs(depth - p) / (depth + 1e-6) for p in self._prev_depths[-2:]],
                axis=0,
            )
            temporal_conf = 1.0 - np.clip(temporal, 0.0, 1.0)
        else:
            temporal_conf = np.ones_like(depth)

        self._prev_depths.append(depth.copy())
        if len(self._prev_depths) > 2:
            self._prev_depths.pop(0)

        return (gradient_conf * temporal_conf).astype(np.float32)
