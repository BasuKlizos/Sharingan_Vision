from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple
import mediapipe as mp
from app.logger import logger


@dataclass(frozen=True)
class FaceDetection:
    score: float
    # pixel coords (x1, y1, x2, y2)
    bbox_xyxy: Tuple[int, int, int, int]


class MediaPipeFaceDetector:
    def __init__(
        self,
        *,
        model_selection: int = 0,
        min_detection_confidence: float = 0.5,
    ):
        self._model_selection = model_selection
        self._min_detection_confidence = min_detection_confidence
        self._detector = None

    def _ensure_detector(self):
        if self._detector is not None:
            return

        logger.info(
            "[MediaPipe] Initializing face detector | model_selection=%s | min_conf=%s",
            self._model_selection,
            self._min_detection_confidence,
        )

        self._detector = mp.solutions.face_detection.FaceDetection(
            model_selection=self._model_selection,
            min_detection_confidence=self._min_detection_confidence,
        )

    def detect(self, img_rgb) -> List[FaceDetection]:
        """
        Args:
            img_rgb: RGB image (numpy array HxWx3 uint8)
        """
        self._ensure_detector()

        h, w = img_rgb.shape[:2]
        result = self._detector.process(img_rgb)
        detections = getattr(result, "detections", None) or []

        faces: List[FaceDetection] = []
        for d in detections:
            try:
                score = float(d.score[0]) if getattr(d, "score", None) else 0.0
                rb = d.location_data.relative_bounding_box
                x1 = max(0, int(rb.xmin * w))
                y1 = max(0, int(rb.ymin * h))
                x2 = min(w, int((rb.xmin + rb.width) * w))
                y2 = min(h, int((rb.ymin + rb.height) * h))
                faces.append(FaceDetection(score=score, bbox_xyxy=(x1, y1, x2, y2)))
            except Exception:
                continue

        return faces

    def close(self):
        if self._detector is not None:
            try:
                self._detector.close()
            except Exception:
                pass
            self._detector = None
