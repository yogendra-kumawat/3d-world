from __future__ import annotations
from typing import List, Optional
import numpy as np

from tactile_vision.types import Detection


class ObjectDetector:
    """YOLOv8 object detector. Lazy-loads the model on first predict()."""

    def __init__(
        self,
        model_name: str = "yolov8n.pt",
        conf_thresh: float = 0.35,
        device: Optional[str] = None,
        imgsz: int = 640,
    ) -> None:
        self.model_name = model_name
        self.conf_thresh = conf_thresh
        self.device = device
        self.imgsz = imgsz
        self._model = None
        self._names: dict = {}

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        from ultralytics import YOLO
        import torch

        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model = YOLO(self.model_name)
        self._names = self._model.model.names if hasattr(self._model, "model") else {}

    def detect(self, frame_bgr: np.ndarray) -> List[Detection]:
        self._ensure_loaded()
        results = self._model.predict(
            frame_bgr,
            conf=self.conf_thresh,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,
        )
        out: List[Detection] = []
        if not results:
            return out
        r0 = results[0]
        if r0.boxes is None:
            return out
        boxes = r0.boxes
        xyxy = boxes.xyxy.cpu().numpy().astype(int) if hasattr(boxes.xyxy, "cpu") else np.asarray(boxes.xyxy, dtype=int)
        confs = boxes.conf.cpu().numpy() if hasattr(boxes.conf, "cpu") else np.asarray(boxes.conf)
        cls_ids = boxes.cls.cpu().numpy().astype(int) if hasattr(boxes.cls, "cpu") else np.asarray(boxes.cls, dtype=int)
        names = r0.names if hasattr(r0, "names") and r0.names else self._names
        for (x1, y1, x2, y2), score, cls_id in zip(xyxy, confs, cls_ids):
            cls_name = names.get(int(cls_id), str(cls_id)) if isinstance(names, dict) else str(cls_id)
            out.append(
                Detection(
                    cls=str(cls_name),
                    box=(int(x1), int(y1), int(x2), int(y2)),
                    score=float(score),
                )
            )
        return out


def detection_distance_m(det: Detection, depth_m: np.ndarray) -> float:
    """Median depth (meters) inside a detection box. Useful for narration / 3D placement."""
    h, w = depth_m.shape
    x1, y1, x2, y2 = det.box
    x1 = max(0, min(w - 1, x1))
    x2 = max(0, min(w, x2))
    y1 = max(0, min(h - 1, y1))
    y2 = max(0, min(h, y2))
    if x2 <= x1 or y2 <= y1:
        return float("nan")
    crop = depth_m[y1:y2, x1:x2]
    valid = crop[(crop > 0) & np.isfinite(crop)]
    if valid.size == 0:
        return float("nan")
    return float(np.median(valid))
