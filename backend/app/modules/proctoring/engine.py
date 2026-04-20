from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Optional

from app.core.config import settings
from app.modules.proctoring.types import ProctoringAlert, ProctoringInputs


@dataclass
class _ActiveRuleState:
    started_at: float
    last_seen_at: float
    last_emitted_at: float = 0.0


class ProctoringEngine:
    """
    Session-scoped rule engine.

    Design goals:
    - Minimal false positives via hold windows and emit cooldowns.
    - Deterministic rule evaluation.
    - Stores only the minimal session state required for temporal rules.
    """

    def __init__(self, session_id: str):
        self.session_id = session_id
        self._active: dict[str, _ActiveRuleState] = {}
        self._eye_closure_repeats = 0
        self._eye_closure_repeat_window_started_at = 0.0

    def update(self, inputs: ProctoringInputs) -> list[ProctoringAlert]:
        now = inputs.timestamp or time.time()
        alerts: list[ProctoringAlert] = []

        alerts.extend(self._rule_multiple_faces(inputs, now))
        alerts.extend(self._rule_face_missing(inputs, now))
        alerts.extend(self._rule_person_present_no_face(inputs, now))
        alerts.extend(self._rule_phone_detected(inputs, now))
        alerts.extend(self._rule_external_device_detected(inputs, now))
        alerts.extend(self._rule_unauthorized_materials(inputs, now))
        alerts.extend(self._rule_excessive_head_movement(inputs, now))
        alerts.extend(self._rule_gaze_away(inputs, now))
        alerts.extend(self._rule_suspicious_hand_movement(inputs, now))
        alerts.extend(self._rule_abnormal_blink(inputs, now))
        alerts.extend(self._rule_repeated_eye_closure(inputs, now))
        return alerts

    @staticmethod
    def _extra_list(inputs: ProctoringInputs, key: str) -> list[str]:
        extra = inputs.extra or {}
        values = extra.get(key) or []
        return [str(value) for value in values]

    def _activate_if_true(
        self, rule_id: str, condition: bool, now: float
    ) -> Optional[_ActiveRuleState]:
        if condition:
            existing = self._active.get(rule_id)
            if existing is None:
                existing = _ActiveRuleState(started_at=now, last_seen_at=now)
                self._active[rule_id] = existing
                return existing
            existing.last_seen_at = now
            return existing

        if rule_id in self._active:
            del self._active[rule_id]
        return None

    def _emit_once_with_cooldown(
        self,
        state: _ActiveRuleState,
        now: float,
        cooldown_s: float,
        build_alert: Callable[[], ProctoringAlert],
    ) -> list[ProctoringAlert]:
        if (now - state.last_emitted_at) < cooldown_s:
            return []
        state.last_emitted_at = now
        return [build_alert()]

    def _rule_multiple_faces(self, inputs: ProctoringInputs, now: float) -> list[ProctoringAlert]:
        state = self._activate_if_true(
            "THIRD_PARTY_PRESENCE",
            inputs.multiple_persons_detected or inputs.face_count > 1,
            now,
        )
        if state is None:
            return []
        if (now - state.started_at) < float(settings.PROCTOR_MULTI_FACE_HOLD_SECONDS):
            return []
        return self._emit_once_with_cooldown(
            state,
            now,
            float(settings.PROCTOR_ALERT_COOLDOWN_SECONDS),
            lambda: ProctoringAlert(
                session_id=inputs.session_id,
                frame_id=inputs.frame_id,
                rule_id="THIRD_PARTY_PRESENCE",
                label="Multiple faces detected",
                severity="critical",
                risk_score=0.98,
                started_at=state.started_at,
                last_seen_at=state.last_seen_at,
                evidence={
                    "face_count": inputs.face_count,
                    "confidence": round(float(inputs.multiple_persons_confidence), 3),
                },
            ),
        )

    def _rule_face_missing(self, inputs: ProctoringInputs, now: float) -> list[ProctoringAlert]:
        state = self._activate_if_true("CANDIDATE_LEFT_FRAME", not inputs.face_detected, now)
        if state is None:
            return []
        duration_s = now - state.started_at
        if duration_s < float(settings.PROCTOR_NO_FACE_SECONDS):
            return []
        return self._emit_once_with_cooldown(
            state,
            now,
            float(settings.PROCTOR_ALERT_COOLDOWN_SECONDS),
            lambda: ProctoringAlert(
                session_id=inputs.session_id,
                frame_id=inputs.frame_id,
                rule_id="CANDIDATE_LEFT_FRAME",
                label="Face missing from frame",
                severity="high",
                risk_score=0.84,
                started_at=state.started_at,
                last_seen_at=state.last_seen_at,
                evidence={"missing_for_s": round(duration_s, 2)},
            ),
        )

    def _rule_person_present_no_face(
        self, inputs: ProctoringInputs, now: float
    ) -> list[ProctoringAlert]:
        raw_alerts = {value.upper() for value in self._extra_list(inputs, "raw_alerts")}
        state = self._activate_if_true(
            "PERSON_PRESENT_NO_FACE",
            "PERSON_PRESENT_NO_FACE" in raw_alerts,
            now,
        )
        if state is None:
            return []
        return self._emit_once_with_cooldown(
            state,
            now,
            float(settings.PROCTOR_ALERT_COOLDOWN_SECONDS),
            lambda: ProctoringAlert(
                session_id=inputs.session_id,
                frame_id=inputs.frame_id,
                rule_id="PERSON_PRESENT_NO_FACE",
                label="Person visible but face not detected",
                severity="high",
                risk_score=0.9,
                started_at=state.started_at,
                last_seen_at=state.last_seen_at,
                evidence={
                    "face_count": inputs.face_count,
                    "labels": self._extra_list(inputs, "labels"),
                    "device_count": int((inputs.extra or {}).get("device_count", 0)),
                },
            ),
        )

    def _rule_phone_detected(self, inputs: ProctoringInputs, now: float) -> list[ProctoringAlert]:
        state = self._activate_if_true(
            "PHONE_DETECTED",
            inputs.phone_detected,
            now,
        )
        if state is None:
            return []
        return self._emit_once_with_cooldown(
            state,
            now,
            float(settings.PROCTOR_ALERT_COOLDOWN_SECONDS),
            lambda: ProctoringAlert(
                session_id=inputs.session_id,
                frame_id=inputs.frame_id,
                rule_id="PHONE_DETECTED",
                label="Cell phone detected",
                severity="high",
                risk_score=0.92,
                started_at=state.started_at,
                last_seen_at=state.last_seen_at,
                evidence={
                    "phone_confidence": round(float(inputs.phone_confidence), 3),
                    "labels": [
                        label
                        for label in self._extra_list(inputs, "labels")
                        if label in {"cell phone", "phone"}
                    ],
                },
            ),
        )

    def _rule_external_device_detected(
        self, inputs: ProctoringInputs, now: float
    ) -> list[ProctoringAlert]:
        state = self._activate_if_true(
            "EXTERNAL_DEVICE_DETECTED",
            inputs.other_device_detected,
            now,
        )
        if state is None:
            return []
        return self._emit_once_with_cooldown(
            state,
            now,
            float(settings.PROCTOR_ALERT_COOLDOWN_SECONDS),
            lambda: ProctoringAlert(
                session_id=inputs.session_id,
                frame_id=inputs.frame_id,
                rule_id="EXTERNAL_DEVICE_DETECTED",
                label="External device detected",
                severity="high",
                risk_score=0.72,
                started_at=state.started_at,
                last_seen_at=state.last_seen_at,
                evidence={
                    "other_device_confidence": round(float(inputs.other_device_confidence), 3),
                    "labels": [
                        label
                        for label in self._extra_list(inputs, "labels")
                        if label in {"laptop", "tablet", "keyboard", "mouse", "remote"}
                    ],
                },
            ),
        )

    def _rule_unauthorized_materials(
        self, inputs: ProctoringInputs, now: float
    ) -> list[ProctoringAlert]:
        state = self._activate_if_true(
            "UNAUTHORIZED_MATERIALS",
            inputs.unauthorized_materials_detected,
            now,
        )
        if state is None:
            return []
        if (now - state.started_at) < float(settings.PROCTOR_MATERIAL_HOLD_SECONDS):
            return []
        return self._emit_once_with_cooldown(
            state,
            now,
            float(settings.PROCTOR_ALERT_COOLDOWN_SECONDS),
            lambda: ProctoringAlert(
                session_id=inputs.session_id,
                frame_id=inputs.frame_id,
                rule_id="UNAUTHORIZED_MATERIALS",
                label="Unauthorized materials detected",
                severity="high",
                risk_score=0.78,
                started_at=state.started_at,
                last_seen_at=state.last_seen_at,
                evidence={
                    "confidence": round(float(inputs.unauthorized_materials_confidence), 3),
                },
            ),
        )

    def _rule_excessive_head_movement(
        self, inputs: ProctoringInputs, now: float
    ) -> list[ProctoringAlert]:
        yaw_threshold = float(settings.PROCTOR_HEAD_YAW_THRESHOLD)
        state = self._activate_if_true(
            "EXCESSIVE_HEAD_MOVEMENT",
            abs(float(inputs.head_yaw)) >= yaw_threshold,
            now,
        )
        if state is None:
            return []
        duration_s = now - state.started_at
        if duration_s < float(settings.PROCTOR_HEAD_AWAY_SECONDS):
            return []
        return self._emit_once_with_cooldown(
            state,
            now,
            float(settings.PROCTOR_ALERT_COOLDOWN_SECONDS),
            lambda: ProctoringAlert(
                session_id=inputs.session_id,
                frame_id=inputs.frame_id,
                rule_id="EXCESSIVE_HEAD_MOVEMENT",
                label="Excessive head movement / looking away",
                severity="medium",
                risk_score=0.62,
                started_at=state.started_at,
                last_seen_at=state.last_seen_at,
                evidence={
                    "head_yaw": round(float(inputs.head_yaw), 4),
                    "yaw_threshold": yaw_threshold,
                    "duration_s": round(duration_s, 2),
                },
            ),
        )

    def _rule_gaze_away(self, inputs: ProctoringInputs, now: float) -> list[ProctoringAlert]:
        gaze = (inputs.gaze_direction or "unknown").lower()
        state = self._activate_if_true(
            "GAZE_AWAY",
            gaze not in {"center", "unknown", "none"},
            now,
        )
        if state is None:
            return []
        duration_s = now - state.started_at
        if duration_s < float(settings.PROCTOR_GAZE_AWAY_SECONDS):
            return []
        return self._emit_once_with_cooldown(
            state,
            now,
            float(settings.PROCTOR_ALERT_COOLDOWN_SECONDS),
            lambda: ProctoringAlert(
                session_id=inputs.session_id,
                frame_id=inputs.frame_id,
                rule_id="GAZE_AWAY",
                label="Prolonged gaze away from screen",
                severity="medium",
                risk_score=0.68,
                started_at=state.started_at,
                last_seen_at=state.last_seen_at,
                evidence={
                    "gaze_direction": gaze,
                    "duration_s": round(duration_s, 2),
                },
            ),
        )

    def _rule_suspicious_hand_movement(
        self, inputs: ProctoringInputs, now: float
    ) -> list[ProctoringAlert]:
        hands_hidden = inputs.hands_visible_count is not None and inputs.hands_visible_count <= 0
        state = self._activate_if_true(
            "SUSPICIOUS_HAND_MOVEMENT",
            hands_hidden or inputs.rapid_hand_movement,
            now,
        )
        if state is None:
            return []
        duration_s = now - state.started_at
        if duration_s < float(settings.PROCTOR_HANDS_HIDDEN_SECONDS):
            return []
        return self._emit_once_with_cooldown(
            state,
            now,
            float(settings.PROCTOR_ALERT_COOLDOWN_SECONDS),
            lambda: ProctoringAlert(
                session_id=inputs.session_id,
                frame_id=inputs.frame_id,
                rule_id="SUSPICIOUS_HAND_MOVEMENT",
                label="Suspicious hand movement / hands hidden",
                severity="medium",
                risk_score=0.55,
                started_at=state.started_at,
                last_seen_at=state.last_seen_at,
                evidence={
                    "hands_visible_count": inputs.hands_visible_count,
                    "rapid_hand_movement": inputs.rapid_hand_movement,
                    "duration_s": round(duration_s, 2),
                },
            ),
        )

    def _rule_abnormal_blink(self, inputs: ProctoringInputs, now: float) -> list[ProctoringAlert]:
        state = self._activate_if_true(
            "ABNORMAL_BLINK_PATTERN",
            inputs.blink_rate_per_min >= float(settings.PROCTOR_BLINK_RATE_THRESHOLD)
            or inputs.sustained_eye_closure_s >= float(settings.PROCTOR_EYE_CLOSURE_SECONDS),
            now,
        )
        if state is None:
            return []
        return self._emit_once_with_cooldown(
            state,
            now,
            float(settings.PROCTOR_ALERT_COOLDOWN_SECONDS),
            lambda: ProctoringAlert(
                session_id=inputs.session_id,
                frame_id=inputs.frame_id,
                rule_id="ABNORMAL_BLINK_PATTERN",
                label="Abnormal blink pattern",
                severity="medium",
                risk_score=0.46,
                started_at=state.started_at,
                last_seen_at=state.last_seen_at,
                evidence={
                    "blink_rate_per_min": round(float(inputs.blink_rate_per_min), 2),
                    "sustained_eye_closure_s": round(float(inputs.sustained_eye_closure_s), 2),
                },
            ),
        )

    def _rule_repeated_eye_closure(
        self, inputs: ProctoringInputs, now: float
    ) -> list[ProctoringAlert]:
        left_closed = bool(inputs.left_eye_closed)
        right_closed = bool(inputs.right_eye_closed)
        closure_detected = left_closed and right_closed
        window_s = float(settings.PROCTOR_EYE_CLOSURE_REPEAT_WINDOW_SECONDS)
        repeat_threshold = int(settings.PROCTOR_EYE_CLOSURE_REPEAT_COUNT)

        if not closure_detected:
            self._activate_if_true("REPEATED_EYE_CLOSURE", False, now)
            return []

        if (
            self._eye_closure_repeat_window_started_at <= 0
            or (now - self._eye_closure_repeat_window_started_at) > window_s
        ):
            self._eye_closure_repeat_window_started_at = now
            self._eye_closure_repeats = 0

        state = self._activate_if_true("REPEATED_EYE_CLOSURE", True, now)
        if state is None:
            return []

        if (now - state.last_emitted_at) >= 1.0:
            self._eye_closure_repeats += 1

        if self._eye_closure_repeats < repeat_threshold:
            return []

        return self._emit_once_with_cooldown(
            state,
            now,
            float(settings.PROCTOR_ALERT_COOLDOWN_SECONDS),
            lambda: ProctoringAlert(
                session_id=inputs.session_id,
                frame_id=inputs.frame_id,
                rule_id="REPEATED_EYE_CLOSURE",
                label="Repeated eye closure",
                severity="medium",
                risk_score=0.52,
                started_at=state.started_at,
                last_seen_at=state.last_seen_at,
                evidence={
                    "repeat_count": self._eye_closure_repeats,
                    "window_s": window_s,
                },
            ),
        )
