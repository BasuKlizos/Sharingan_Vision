from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

from app.logger import logger


@dataclass(frozen=True)
class YoloDetection:
    class_id: int
    class_name: str
    confidence: float
    xyxy: Tuple[int, int, int, int]


class YoloDetector:
    def __init__(self, model_path: str, conf_threshold: float = 0.25):
        self._model_path = model_path
        self._conf_threshold = conf_threshold
        self._model = None
        self._warmup_done = False

    def _ensure_model(self):
        if self._model is not None:
            return

        try:
            from ultralytics import YOLO  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "ultralytics is not installed. Add `ultralytics` to requirements and reinstall."
            ) from e

        logger.info(f"[YOLO] Loading model | path={self._model_path}")
        load_start = time.time()
        self._model = YOLO(self._model_path)
        load_time = time.time() - load_start
        logger.info(f"[YOLO] Model loaded | time={load_time:.2f}s")

        # Warm up the model with dummy inference to avoid first-run stall
        if not self._warmup_done:
            self._warmup_model()

    def _warmup_model(self):
        """Run dummy inference to warm up model and avoid first-call stalls"""
        try:
            import numpy as np

            logger.debug("[YOLO] Starting model warm-up...")
            warmup_start = time.time()

            # Small dummy image to warm up
            dummy = np.zeros((640, 640, 3), dtype=np.uint8)
            self._model.predict(dummy, conf=self._conf_threshold, verbose=False, device="cpu")

            warmup_time = time.time() - warmup_start
            logger.info(f"[YOLO] Model warm-up complete | time={warmup_time:.2f}s")
            self._warmup_done = True
        except Exception as e:
            logger.error(f"[YOLO] Warm-up failed | error={e}")
            self._warmup_done = False

    def detect(
        self, img_rgb: bytes, *, conf_threshold: Optional[float] = None
    ) -> List[YoloDetection]:
        """
        Args:
            img_rgb: OpenCV RGB image (numpy array HxWx3 uint8)
            conf_threshold: Optional override for confidence threshold
        """
        self._ensure_model()

        threshold = self._conf_threshold if conf_threshold is None else conf_threshold
        logger.debug(f"[YOLO] Running inference | conf_threshold={threshold}")

        inference_start = time.time()

        # ultralytics returns a list of Results objects
        results = self._model.predict(
            img_rgb, conf=threshold, verbose=False, device="cpu"  # Force CPU for stability
        )

        inference_time = time.time() - inference_start
        logger.debug(f"[YOLO] Inference time | duration={inference_time:.3f}s")

        detections: List[YoloDetection] = []
        for r in results:
            names = getattr(r, "names", {}) or {}
            boxes = getattr(r, "boxes", None)
            if boxes is None:
                continue

            # xyxy, cls, conf are torch tensors
            for b in boxes:
                xyxy = getattr(b, "xyxy", None)
                cls = getattr(b, "cls", None)
                conf = getattr(b, "conf", None)
                if xyxy is None or cls is None or conf is None:
                    continue

                x1, y1, x2, y2 = [int(v) for v in xyxy[0].tolist()]
                class_id = int(cls.item())
                confidence = float(conf.item())
                class_name = str(names.get(class_id, class_id))

                detections.append(
                    YoloDetection(
                        class_id=class_id,
                        class_name=class_name,
                        confidence=confidence,
                        xyxy=(x1, y1, x2, y2),
                    )
                )

        logger.debug(
            f"[YOLO] Inference complete | detections={len(detections)} time={inference_time:.3f}s"
        )
        return detections
