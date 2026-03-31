from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch

from app.logger import logger


@dataclass(frozen=True)
class YoloDetection:
    class_id: int
    class_name: str
    confidence: float
    xyxy: Tuple[int, int, int, int]  # coordinates in ORIGINAL frame space


class YoloDetector:
    def __init__(self, model_path: str, conf_threshold: float = 0.5, iou_threshold: float = 0.45):
        self._model_path = model_path
        self._conf_threshold = conf_threshold
        self._iou_threshold = iou_threshold
        self._model = None
        self._lock = threading.Lock()
        self._device = self._resolve_device()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_device() -> str:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
        logger.info(f"[YOLO] Device resolved | device={device}")
        return device

    def _ensure_model(self) -> None:
        """Thread-safe lazy model loader."""
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:  # double-checked locking
                return
            try:
                from ultralytics import YOLO
            except ImportError as e:
                raise RuntimeError(
                    "ultralytics is not installed. Run: pip install ultralytics"
                ) from e

            logger.info(f"[YOLO] Loading model | path={self._model_path} | device={self._device}")
            self._model = YOLO(self._model_path)
            self._model.to(self._device)
            logger.info(f"[YOLO] Model ready | path={self._model_path} | device={self._device}")

    @staticmethod
    def _letterbox(
        img: np.ndarray,
        target_size: int = 640,
    ) -> Tuple[np.ndarray, float, Tuple[int, int]]:
        """
        Resize image to target_size x target_size with letterboxing to preserve
        aspect ratio. Returns the resized image, the scale factor, and the
        (pad_left, pad_top) offsets so detections can be mapped back to the
        original frame coordinates.
        """
        h, w = img.shape[:2]
        scale = min(target_size / h, target_size / w)
        new_w, new_h = int(w * scale), int(h * scale)

        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        pad_top = (target_size - new_h) // 2
        pad_left = (target_size - new_w) // 2

        canvas = np.full((target_size, target_size, 3), 114, dtype=np.uint8)
        canvas[pad_top : pad_top + new_h, pad_left : pad_left + new_w] = resized

        return canvas, scale, (pad_left, pad_top)

    @staticmethod
    def _unscale_box(
        x1: int, y1: int, x2: int, y2: int,
        scale: float,
        pad_left: int,
        pad_top: int,
        orig_w: int,
        orig_h: int,
    ) -> Tuple[int, int, int, int]:
        """Map letterboxed coordinates back to original frame space."""
        x1 = int(max(0, (x1 - pad_left) / scale))
        y1 = int(max(0, (y1 - pad_top) / scale))
        x2 = int(min(orig_w, (x2 - pad_left) / scale))
        y2 = int(min(orig_h, (y2 - pad_top) / scale))
        return x1, y1, x2, y2

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(
        self,
        img_bgr: np.ndarray,
        *,
        conf_threshold: Optional[float] = None,
    ) -> List[YoloDetection]:
        """
        Run object detection on a BGR frame.

        Args:
            img_bgr: OpenCV BGR image (HxWx3 uint8).
            conf_threshold: Per-call override for confidence threshold.

        Returns:
            List of YoloDetection with bounding boxes in original frame coordinates.
        """
        self._ensure_model()

        if img_bgr is None or img_bgr.size == 0:
            logger.warning("[YOLO] Empty or null frame — skipping inference")
            return []

        threshold = conf_threshold if conf_threshold is not None else self._conf_threshold
        orig_h, orig_w = img_bgr.shape[:2]

        # Letterbox: preserves aspect ratio, avoids distortion
        img_lb, scale, (pad_left, pad_top) = self._letterbox(img_bgr, target_size=640)

        # BGR → RGB (ultralytics expects RGB numpy arrays)
        img_rgb = cv2.cvtColor(img_lb, cv2.COLOR_BGR2RGB)

        start_ts = time.monotonic()
        try:
            results = self._model.predict(
                img_rgb,
                conf=threshold,
                iou=self._iou_threshold,
                imgsz=640,
                device=self._device,
                verbose=False,
            )
        except Exception:
            logger.exception("[YOLO] Inference failed")
            return []

        elapsed_ms = (time.monotonic() - start_ts) * 1000.0

        detections: List[YoloDetection] = []

        for r in results:
            if r.boxes is None:
                continue

            names: dict = r.names

            for b in r.boxes:
                try:
                    x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
                    class_id = int(b.cls.item())
                    confidence = float(b.conf.item())
                    class_name = names.get(class_id, str(class_id))
                except Exception:
                    logger.exception("[YOLO] Failed to parse detection box — skipping")
                    continue

                # Map back to original frame coordinates
                x1, y1, x2, y2 = self._unscale_box(
                    x1, y1, x2, y2, scale, pad_left, pad_top, orig_w, orig_h
                )

                # Sanity check: skip degenerate boxes
                if x2 <= x1 or y2 <= y1:
                    logger.debug(f"[YOLO] Degenerate box skipped | ({x1},{y1},{x2},{y2})")
                    continue

                detections.append(
                    YoloDetection(
                        class_id=class_id,
                        class_name=class_name,
                        confidence=confidence,
                        xyxy=(x1, y1, x2, y2),
                    )
                )

        logger.info(
            f"[YOLO] Inference complete | detections={len(detections)} "
            f"| elapsed={elapsed_ms:.1f}ms | conf={threshold} | device={self._device}"
        )

        return detections