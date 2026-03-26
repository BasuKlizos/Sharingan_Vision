from __future__ import annotations

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
        self._model = YOLO(self._model_path)

    def detect(self, img_bgr, *, conf_threshold: Optional[float] = None) -> List[YoloDetection]:
        """
        Args:
            img_bgr: OpenCV BGR image (numpy array HxWx3 uint8)
            conf_threshold: Optional override for confidence threshold
        """
        self._ensure_model()

        threshold = self._conf_threshold if conf_threshold is None else conf_threshold
        logger.debug(f"[YOLO] Running inference | conf_threshold={threshold}")

        # ultralytics returns a list of Results objects
        results = self._model.predict(img_bgr, conf=threshold, verbose=False)

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

        logger.debug(f"[YOLO] Inference complete | detections={len(detections)}")
        return detections
