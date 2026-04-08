from __future__ import annotations

from typing import Any

from app.modules.analytics.head_movement import HeadMovementAnalyzer
from app.logger import logger


class FaceAnalyzer:
    """
    Analysis layer that converts a single MediaPipe face result
    into stable behavioral signals for downstream decision making.
    """

    LEFT_EYE_IDX = 33
    RIGHT_EYE_IDX = 263
    NOSE_TIP_IDX = 1
    LEFT_IRIS_CENTER_IDX = 468
    RIGHT_IRIS_CENTER_IDX = 473
    MODERATE_CENTER_OFFSET_THRESHOLD = 0.16
    STRONG_CENTER_OFFSET_THRESHOLD = 0.22
    MODERATE_HEAD_YAW_THRESHOLD = 0.18
    STRONG_HEAD_YAW_THRESHOLD = 0.32
    MODERATE_EYE_DIRECTION_THRESHOLD = 0.08
    STRONG_EYE_DIRECTION_THRESHOLD = 0.16

    def __init__(
        self,
        no_face_buffer_frames: int = 3,
        off_center_threshold: float = 0.12,
        off_center_hold_frames: int = 3,
        center_recover_frames: int = 2,
    ):
        """
        Args:
            no_face_buffer_frames: Number of consecutive missed frames before emitting NO_FACE.
            off_center_threshold: Threshold for horizontal face-center deviation.
            off_center_hold_frames: Number of consecutive suspicious frames before alerting.
            center_recover_frames: Number of consecutive normal frames before clearing off-center state.
        """
        self.no_face_buffer_frames = max(1, int(no_face_buffer_frames))
        self.off_center_threshold = float(off_center_threshold)
        self.off_center_hold_frames = max(1, int(off_center_hold_frames))
        self.center_recover_frames = max(1, int(center_recover_frames))

        self.no_face_counter = 0
        self.off_center_counter = 0
        self.center_counter = 0
        self.off_center_active = False
        self.head_movement_analyzer = HeadMovementAnalyzer()

    def analyze(self, results: Any, img_shape: Any, session_id: str | None = None) -> dict:
        """
        Convert MediaPipe result into stable face analytics.

        Args:
            results: Raw MediaPipe FaceMesh result object.
            img_shape: Image shape tuple, expected like (H, W, C).
            session_id: Optional session identifier for logging.

        Returns:
            Dict with alerts, faces list, and face_count.
        """
        try:
            h, w = self._extract_image_size(img_shape)
        except ValueError as exc:
            logger.error(
                f"[Analyzer] Invalid image shape | session_id={session_id} "
                f"img_shape={img_shape} error={exc}"
            )
            return self._empty_response(alerts=["ANALYZER_ERROR"])

        if not self._has_face(results):
            return self._handle_no_face(session_id=session_id)

        self.no_face_counter = 0

        face_landmarks = results.multi_face_landmarks[0]

        left_eye = self._safe_landmark(face_landmarks, self.LEFT_EYE_IDX)
        right_eye = self._safe_landmark(face_landmarks, self.RIGHT_EYE_IDX)
        nose_tip = self._safe_landmark(face_landmarks, self.NOSE_TIP_IDX)
        left_iris = self._safe_landmark(face_landmarks, self.LEFT_IRIS_CENTER_IDX)
        right_iris = self._safe_landmark(face_landmarks, self.RIGHT_IRIS_CENTER_IDX)

        if left_eye is None or right_eye is None or nose_tip is None:
            logger.warning(
                f"[Analyzer] Required landmarks missing | session_id={session_id} "
                f"left_eye_missing={left_eye is None} right_eye_missing={right_eye is None} "
                f"nose_tip_missing={nose_tip is None}"
            )
            return {
                "alerts": ["LANDMARKS_INCOMPLETE"],
                "faces": [],
                "face_count": 1,
            }

        eye_center_x = self._clamp01((left_eye.x + right_eye.x) / 2.0)
        eye_center_y = self._clamp01((left_eye.y + right_eye.y) / 2.0)

        eye_x_px = self._to_pixel(eye_center_x, w)
        eye_y_px = self._to_pixel(eye_center_y, h)

        center_offset = abs(eye_center_x - 0.5)
        off_center_raw = center_offset > self.off_center_threshold
        off_center = self._smooth_off_center(off_center_raw)
        head_movement = self.head_movement_analyzer.analyze(
            left_eye=left_eye,
            right_eye=right_eye,
            nose_tip=nose_tip,
            left_iris=left_iris,
            right_iris=right_iris,
        )
        current_view = self._extract_face_bounds(face_landmarks, w, h)
        if current_view is not None:
            logger.debug(
                f"[Analyzer] Current view extracted | session_id={session_id} "
                f"min_x={current_view['minX']:.2f} max_x={current_view['maxX']:.2f} "
                f"min_y={current_view['minY']:.2f} max_y={current_view['maxY']:.2f}"
            )

        alerts: list[str] = []
        if off_center:
            alerts.append("HEAD_OFF_CENTER")
        alerts.extend(head_movement.alerts)
        alerts = list(set(alerts))
        attention_signal = self._build_attention_signal(
            alerts=alerts,
            center_offset=float(center_offset),
            head_yaw=float(head_movement.head_yaw),
            eye_direction=head_movement.eye_direction,
            eye_head_mismatch=bool(head_movement.eye_head_mismatch),
            head_turning=bool(head_movement.head_turning),
        )

        logger.debug(
            f"[Analyzer] Face analyzed | session_id={session_id} "
            f"off_center={off_center} raw={off_center_raw} offset={center_offset:.4f} "
            f"eye_norm=({eye_center_x:.4f},{eye_center_y:.4f}) "
            f"eye_px=({eye_x_px},{eye_y_px}) "
            f"head_yaw={head_movement.head_yaw:.4f} "
            f"head_velocity={head_movement.head_velocity:.4f} "
            f"head_turning={head_movement.head_turning} "
            f"eye_direction={head_movement.eye_direction} "
            f"eye_head_mismatch={head_movement.eye_head_mismatch} "
            f"no_face_counter={self.no_face_counter} "
            f"off_center_counter={self.off_center_counter} "
            f"center_counter={self.center_counter}"
        )

        return {
            "session_id": session_id,
            "alerts": alerts,
            "current_view": current_view,
            "faces": [
                {
                    "off_center": off_center,
                    "off_center_raw": off_center_raw,
                    "center_offset": round(float(center_offset), 4),
                    "eye_norm": (float(eye_center_x), float(eye_center_y)),
                    "eye_px": (eye_x_px, eye_y_px),
                    "head_yaw": head_movement.head_yaw,
                    "head_velocity": head_movement.head_velocity,
                    "head_turning": head_movement.head_turning,
                    "eye_direction": head_movement.eye_direction,
                    "eye_head_mismatch": head_movement.eye_head_mismatch,
                    "attention_state": attention_signal["state"],
                    "attention_score": attention_signal["score"],
                    "attention_reasons": attention_signal["reasons"],
                }
            ],
            "face_count": 1,
        }

    def reset(self) -> None:
        """
        Reset analyzer temporal state. Useful when a session ends.
        """
        self.no_face_counter = 0
        self.off_center_counter = 0
        self.center_counter = 0
        self.off_center_active = False
        self.head_movement_analyzer.reset()

    def _handle_no_face(self, session_id: str | None = None) -> dict:
        self.no_face_counter += 1

        if self.no_face_counter < self.no_face_buffer_frames:
            logger.debug(
                f"[Analyzer] Temporary face miss suppressed | session_id={session_id} "
                f"no_face_counter={self.no_face_counter}/{self.no_face_buffer_frames}"
            )
            return self._empty_response()

        self.off_center_counter = 0
        self.center_counter = 0
        self.off_center_active = False
        self.head_movement_analyzer.reset()

        logger.debug(
            f"[Analyzer] No face detected | session_id={session_id} "
            f"no_face_counter={self.no_face_counter}"
        )
        return self._empty_response(alerts=["NO_FACE"])

    def _smooth_off_center(self, off_center_raw: bool) -> bool:
        if off_center_raw:
            self.off_center_counter += 1
            self.center_counter = 0

            if self.off_center_counter >= self.off_center_hold_frames:
                self.off_center_active = True
        else:
            self.center_counter += 1
            self.off_center_counter = 0

            if self.center_counter >= self.center_recover_frames:
                self.off_center_active = False

        return self.off_center_active

    @staticmethod
    def _has_face(results: Any) -> bool:
        return bool(
            results is not None
            and hasattr(results, "multi_face_landmarks")
            and results.multi_face_landmarks
        )

    @staticmethod
    def _safe_landmark(face_landmarks: Any, index: int) -> Any | None:
        try:
            if face_landmarks is None or not hasattr(face_landmarks, "landmark"):
                return None
            if index < 0 or index >= len(face_landmarks.landmark):
                return None
            return face_landmarks.landmark[index]
        except Exception:
            return None

    @staticmethod
    def _extract_image_size(img_shape: Any) -> tuple[int, int]:
        if not isinstance(img_shape, tuple) or len(img_shape) < 2:
            raise ValueError("img_shape must be a tuple like (H, W, C)")

        h, w = img_shape[:2]

        if not isinstance(h, int) or not isinstance(w, int):
            raise ValueError("height and width must be integers")

        if h <= 0 or w <= 0:
            raise ValueError("height and width must be > 0")

        return h, w

    @staticmethod
    def _clamp01(value: float) -> float:
        return max(0.0, min(1.0, float(value)))

    @staticmethod
    def _to_pixel(norm_value: float, size: int) -> int:
        if size <= 0:
            return 0
        px = int(norm_value * size)
        return max(0, min(size - 1, px))

    @staticmethod
    def _empty_response(alerts: list[str] | None = None) -> dict:
        return {
            "session_id": None,
            "alerts": alerts or [],
            "current_view": None,
            "faces": [],
            "face_count": 0,
        }

    def _extract_face_bounds(self, face_landmarks: Any, width: int, height: int) -> dict | None:
        try:
            landmarks = getattr(face_landmarks, "landmark", None)
            if not landmarks:
                return None

            x_values = [self._to_pixel(self._clamp01(point.x), width) for point in landmarks]
            y_values = [self._to_pixel(self._clamp01(point.y), height) for point in landmarks]

            if not x_values or not y_values:
                return None

            return {
                "minX": float(min(x_values)),
                "maxX": float(max(x_values)),
                "minY": float(min(y_values)),
                "maxY": float(max(y_values)),
            }
        except Exception:
            return None

    def _build_attention_signal(
        self,
        *,
        alerts: list[str],
        center_offset: float,
        head_yaw: float,
        eye_direction: float | None,
        eye_head_mismatch: bool,
        head_turning: bool,
    ) -> dict[str, Any]:
        score = 0
        reasons: list[str] = []
        categories: set[str] = set()

        abs_head_yaw = abs(float(head_yaw))
        abs_eye_direction = abs(float(eye_direction or 0.0))

        if center_offset >= self.STRONG_CENTER_OFFSET_THRESHOLD:
            score += 2
            categories.add("position")
            reasons.append("strong_off_center")
        elif center_offset >= self.MODERATE_CENTER_OFFSET_THRESHOLD:
            score += 1
            categories.add("position")
            reasons.append("moderate_off_center")

        if abs_head_yaw >= self.STRONG_HEAD_YAW_THRESHOLD:
            score += 2
            categories.add("head")
            reasons.append("strong_head_turn")
        elif abs_head_yaw >= self.MODERATE_HEAD_YAW_THRESHOLD:
            score += 1
            categories.add("head")
            reasons.append("moderate_head_turn")

        if abs_eye_direction >= self.STRONG_EYE_DIRECTION_THRESHOLD:
            score += 2
            categories.add("eyes")
            reasons.append("strong_eye_shift")
        elif abs_eye_direction >= self.MODERATE_EYE_DIRECTION_THRESHOLD:
            score += 1
            categories.add("eyes")
            reasons.append("moderate_eye_shift")

        if eye_head_mismatch:
            score += 1
            categories.add("coordination")
            reasons.append("eye_head_mismatch")

        if head_turning or "HEAD_TURNING_PERSISTENT" in alerts:
            score += 1
            categories.add("motion")
            reasons.append("persistent_turn")

        if "HEAD_SUDDEN_JERK" in alerts:
            score += 1
            categories.add("motion")
            reasons.append("sudden_jerk")

        category_count = len(categories)
        if score >= 5 and category_count >= 2:
            state = "far_away"
        elif score >= 3 and category_count >= 2:
            state = "looking_away"
        else:
            state = "good"

        return {
            "state": state,
            "score": score,
            "reasons": reasons,
        }
