from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any


@dataclass
class HeadMovementResult:
    alerts: list[str]
    head_yaw: float
    head_velocity: float
    head_turning: bool
    eye_direction: float | None
    eye_head_mismatch: bool


class HeadMovementAnalyzer:
    """
    Tracks temporal head behavior to identify potentially unnatural movement patterns.
    """

    def __init__(
        self,
        turning_threshold: float = 0.35,
        turning_hold_frames: int = 8,
        frequent_turn_window_frames: int = 30,
        frequent_turn_min_switches: int = 4,
        jerk_delta_threshold: float = 0.35,
        static_motion_threshold: float = 0.01,
        static_hold_frames: int = 45,
        mismatch_head_threshold: float = 0.25,
        mismatch_eye_threshold: float = 0.12,
    ):
        self.turning_threshold = float(turning_threshold)
        self.turning_hold_frames = max(1, int(turning_hold_frames))
        self.frequent_turn_window_frames = max(5, int(frequent_turn_window_frames))
        self.frequent_turn_min_switches = max(1, int(frequent_turn_min_switches))
        self.jerk_delta_threshold = float(jerk_delta_threshold)
        self.static_motion_threshold = float(static_motion_threshold)
        self.static_hold_frames = max(1, int(static_hold_frames))
        self.mismatch_head_threshold = float(mismatch_head_threshold)
        self.mismatch_eye_threshold = float(mismatch_eye_threshold)

        self.prev_head_yaw: float | None = None
        self.prev_eye_mid: tuple[float, float] | None = None
        self.turning_counter = 0
        self.static_counter = 0
        self._turn_sign_history: deque[int] = deque(maxlen=self.frequent_turn_window_frames)

    def reset(self) -> None:
        self.prev_head_yaw = None
        self.prev_eye_mid = None
        self.turning_counter = 0
        self.static_counter = 0
        self._turn_sign_history.clear()

    def analyze(
        self,
        *,
        left_eye: Any,
        right_eye: Any,
        nose_tip: Any,
        left_iris: Any | None = None,
        right_iris: Any | None = None,
    ) -> HeadMovementResult:
        eye_mid_x = (left_eye.x + right_eye.x) / 2.0
        eye_mid_y = (left_eye.y + right_eye.y) / 2.0
        eye_distance = max(abs(right_eye.x - left_eye.x), 1e-6)

        # Positive yaw means face turned to subject's right.
        head_yaw = float((nose_tip.x - eye_mid_x) / eye_distance)

        if self.prev_head_yaw is None:
            head_velocity = 0.0
        else:
            head_velocity = float(abs(head_yaw - self.prev_head_yaw))

        alerts: list[str] = []
        head_turning = abs(head_yaw) >= self.turning_threshold
        if head_turning:
            self.turning_counter += 1
            self._turn_sign_history.append(1 if head_yaw > 0 else -1)
        else:
            self.turning_counter = 0
            self._turn_sign_history.append(0)

        if self.turning_counter >= self.turning_hold_frames:
            alerts.append("HEAD_TURNING_PERSISTENT")

        if self._count_turn_switches() >= self.frequent_turn_min_switches:
            alerts.append("HEAD_TURNING_FREQUENT")

        if head_velocity >= self.jerk_delta_threshold:
            alerts.append("HEAD_SUDDEN_JERK")

        eye_mid = (float(eye_mid_x), float(eye_mid_y))
        if self.prev_eye_mid is None:
            motion = 0.0
        else:
            motion = max(
                abs(eye_mid[0] - self.prev_eye_mid[0]),
                abs(eye_mid[1] - self.prev_eye_mid[1]),
            )

        if motion <= self.static_motion_threshold:
            self.static_counter += 1
        else:
            self.static_counter = 0

        if self.static_counter >= self.static_hold_frames:
            alerts.append("HEAD_STATIC_POSE")

        eye_direction = self._estimate_eye_direction(left_eye, right_eye, left_iris, right_iris)
        eye_head_mismatch = self._is_eye_head_mismatch(head_yaw, eye_direction)
        if eye_head_mismatch:
            alerts.append("HEAD_EYE_DIRECTION_MISMATCH")

        self.prev_head_yaw = head_yaw
        self.prev_eye_mid = eye_mid

        return HeadMovementResult(
            alerts=alerts,
            head_yaw=round(head_yaw, 4),
            head_velocity=round(head_velocity, 4),
            head_turning=head_turning,
            eye_direction=None if eye_direction is None else round(eye_direction, 4),
            eye_head_mismatch=eye_head_mismatch,
        )

    def _count_turn_switches(self) -> int:
        switches = 0
        prev = 0
        for sign in self._turn_sign_history:
            if sign == 0:
                continue
            if prev != 0 and sign != prev:
                switches += 1
            prev = sign
        return switches

    def _estimate_eye_direction(
        self,
        left_eye: Any,
        right_eye: Any,
        left_iris: Any | None,
        right_iris: Any | None,
    ) -> float | None:
        if left_iris is None or right_iris is None:
            return None

        eye_mid_x = (left_eye.x + right_eye.x) / 2.0
        iris_mid_x = (left_iris.x + right_iris.x) / 2.0
        eye_distance = max(abs(right_eye.x - left_eye.x), 1e-6)
        return float((iris_mid_x - eye_mid_x) / eye_distance)

    def _is_eye_head_mismatch(self, head_yaw: float, eye_direction: float | None) -> bool:
        if eye_direction is None:
            return False
        if abs(head_yaw) < self.mismatch_head_threshold:
            return False
        if abs(eye_direction) < self.mismatch_eye_threshold:
            return False
        return (head_yaw * eye_direction) < 0
