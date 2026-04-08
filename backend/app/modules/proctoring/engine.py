from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

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
    - Minimal false positives: time-window thresholds, debounce, and emit cooldown.
    - Deterministic: purely rule-based (no ML).
    - Stores only aggregated, necessary metrics.
    """

    def __init__(self, session_id: str):
        self.session_id = session_id
        self._active: Dict[str, _ActiveRuleState] = {}
        self._eye_closure_repeats: int = 0
        self._eye_closure_repeat_window_started_at: Optional[float] = None

    def update(self, inputs: ProctoringInputs) -> list[ProctoringAlert]:
        now = float(inputs.timestamp or time.time())
        alerts: list[ProctoringAlert] = []

        # --- CRITICAL VIOLATIONS ---
        alerts.extend(self._rule_multiple_faces(inputs, now))
        alerts.extend(self._rule_face_missing(inputs, now))
        alerts.extend(self._rule_device_detected(inputs, now))
        alerts.extend(self._rule_unauthorized_materials(inputs, now))

        # --- HIGH VIOLATIONS ---
        alerts.extend(self._rule_excessive_head_movement(inputs, now))
        alerts.extend(self._rule_gaze_away(inputs, now))
        alerts.extend(self._rule_suspicious_hand_movement(inputs, now))

        # --- MEDIUM VIOLATIONS ---
        alerts.extend(self._rule_abnormal_blink(inputs, now))
        alerts.extend(self._rule_repeated_eye_closure(inputs, now))

        return alerts

    # -----------------------------
    # Helpers
    # -----------------------------
    def _activate_if_true(
        self,
        *,
        rule_id: str,
        condition: bool,
        now: float,
    ) -> Optional[_ActiveRuleState]:
        if condition:
            existing = self._active.get(rule_id)
            if existing is None:
                existing = _ActiveRuleState(started_at=now, last_seen_at=now)
                self._active[rule_id] = existing
            else:
                existing.last_seen_at = now
            return existing

        if rule_id in self._active:
            del self._active[rule_id]
        return None

    def _emit_once_with_cooldown(
        self,
        *,
        rule_id: str,
        state: _ActiveRuleState,
        now: float,
        cooldown_s: float,
        build_alert: callable,
    ) -> list[ProctoringAlert]:
        if (now - state.last_emitted_at) < cooldown_s:
            return []
        state.last_emitted_at = now
        return [build_alert()]

    # -----------------------------
    # Rules
    # -----------------------------
    def _rule_multiple_faces(self, inputs: ProctoringInputs, now: float) -> list[ProctoringAlert]:
        rule_id = "THIRD_PARTY_PRESENT"
        confidence = inputs.multiple_persons_confidence or 0.0
        condition = bool(inputs.face_count > 1 and confidence >= 0.8)
        state = self._activate_if_true(rule_id=rule_id, condition=condition, now=now)
        if not state:
            return []

        return self._emit_once_with_cooldown(
            rule_id=rule_id,
            state=state,
            now=now,
            cooldown_s=5.0,
            build_alert=lambda: ProctoringAlert(
                session_id=inputs.session_id,
                rule_id=rule_id,
                label="Multiple faces detected",
                severity="critical",
                risk_score=98,
                started_at=state.started_at,
                last_seen_at=state.last_seen_at,
                frame_id=inputs.frame_id,
                evidence={"face_count": inputs.face_count, "confidence": confidence},
            ),
        )

    def _rule_face_missing(self, inputs: ProctoringInputs, now: float) -> list[ProctoringAlert]:
        rule_id = "CANDIDATE_LEFT_FRAME"
        duration_s = float(getattr(settings, "PROCTOR_NO_FACE_SECONDS", 3.0))
        condition = not bool(inputs.face_detected)
        state = self._activate_if_true(rule_id=rule_id, condition=condition, now=now)
        if not state:
            return []
        if (now - state.started_at) < duration_s:
            return []

        return self._emit_once_with_cooldown(
            rule_id=rule_id,
            state=state,
            now=now,
            cooldown_s=5.0,
            build_alert=lambda: ProctoringAlert(
                session_id=inputs.session_id,
                rule_id=rule_id,
                label="Face missing from frame",
                severity="critical",
                risk_score=90,
                started_at=state.started_at,
                last_seen_at=state.last_seen_at,
                frame_id=inputs.frame_id,
                evidence={"missing_for_s": round(now - state.started_at, 2)},
            ),
        )

    def _rule_device_detected(self, inputs: ProctoringInputs, now: float) -> list[ProctoringAlert]:
        rule_id = "DEVICE_DETECTED"
        phone_conf = float(inputs.phone_confidence or 0.0)
        other_conf = float(inputs.other_device_confidence or 0.0)
        condition = bool(inputs.phone_detected or other_conf >= 0.75)
        state = self._activate_if_true(rule_id=rule_id, condition=condition, now=now)
        if not state:
            return []

        risk = 95 if inputs.phone_detected else 92
        return self._emit_once_with_cooldown(
            rule_id=rule_id,
            state=state,
            now=now,
            cooldown_s=3.0,
            build_alert=lambda: ProctoringAlert(
                session_id=inputs.session_id,
                rule_id=rule_id,
                label="Phone/external device detected",
                severity="critical",
                risk_score=risk,
                started_at=state.started_at,
                last_seen_at=state.last_seen_at,
                frame_id=inputs.frame_id,
                evidence={
                    "phone_detected": inputs.phone_detected,
                    "phone_confidence": phone_conf,
                    "other_device_confidence": other_conf,
                },
            ),
        )

    def _rule_unauthorized_materials(self, inputs: ProctoringInputs, now: float) -> list[ProctoringAlert]:
        rule_id = "UNAUTHORIZED_MATERIALS"
        confidence = float(inputs.unauthorized_materials_confidence or 0.0)
        condition = bool(inputs.unauthorized_materials_detected)
        state = self._activate_if_true(rule_id=rule_id, condition=condition, now=now)
        if not state:
            return []

        return self._emit_once_with_cooldown(
            rule_id=rule_id,
            state=state,
            now=now,
            cooldown_s=5.0,
            build_alert=lambda: ProctoringAlert(
                session_id=inputs.session_id,
                rule_id=rule_id,
                label="Unauthorized materials detected",
                severity="high",
                risk_score=85,
                started_at=state.started_at,
                last_seen_at=state.last_seen_at,
                frame_id=inputs.frame_id,
                evidence={"confidence": confidence},
            ),
        )

    def _rule_excessive_head_movement(self, inputs: ProctoringInputs, now: float) -> list[ProctoringAlert]:
        rule_id = "EXCESSIVE_HEAD_MOVEMENT"
        duration_s = float(getattr(settings, "PROCTOR_HEAD_AWAY_SECONDS", 5.0))
        yaw_threshold = float(getattr(settings, "PROCTOR_HEAD_YAW_THRESHOLD", 45.0))
        yaw = inputs.head_yaw
        condition = yaw is not None and (yaw > yaw_threshold or yaw < -yaw_threshold)
        state = self._activate_if_true(rule_id=rule_id, condition=bool(condition), now=now)
        if not state:
            return []
        if (now - state.started_at) < duration_s:
            return []

        return self._emit_once_with_cooldown(
            rule_id=rule_id,
            state=state,
            now=now,
            cooldown_s=5.0,
            build_alert=lambda: ProctoringAlert(
                session_id=inputs.session_id,
                rule_id=rule_id,
                label="Excessive head movement / looking away",
                severity="high",
                risk_score=78,
                started_at=state.started_at,
                last_seen_at=state.last_seen_at,
                frame_id=inputs.frame_id,
                evidence={"head_yaw": yaw, "duration_s": round(now - state.started_at, 2)},
            ),
        )

    def _rule_gaze_away(self, inputs: ProctoringInputs, now: float) -> list[ProctoringAlert]:
        rule_id = "GAZE_AWAY"
        duration_s = float(getattr(settings, "PROCTOR_GAZE_AWAY_SECONDS", 8.0))
        gaze = (inputs.gaze_direction or "unknown").lower()
        condition = gaze not in ("center", "unknown", "none", "")
        state = self._activate_if_true(rule_id=rule_id, condition=bool(condition), now=now)
        if not state:
            return []
        if (now - state.started_at) < duration_s:
            return []

        return self._emit_once_with_cooldown(
            rule_id=rule_id,
            state=state,
            now=now,
            cooldown_s=5.0,
            build_alert=lambda: ProctoringAlert(
                session_id=inputs.session_id,
                rule_id=rule_id,
                label="Prolonged gaze away from screen",
                severity="high",
                risk_score=70,
                started_at=state.started_at,
                last_seen_at=state.last_seen_at,
                frame_id=inputs.frame_id,
                evidence={"gaze_direction": gaze, "duration_s": round(now - state.started_at, 2)},
            ),
        )

    def _rule_suspicious_hand_movement(self, inputs: ProctoringInputs, now: float) -> list[ProctoringAlert]:
        rule_id = "SUSPICIOUS_HAND_MOVEMENT"
        duration_s = float(getattr(settings, "PROCTOR_HANDS_HIDDEN_SECONDS", 10.0))
        hands_visible = inputs.hands_visible_count
        condition = (hands_visible is not None and hands_visible == 0) or bool(inputs.rapid_hand_movement)
        state = self._activate_if_true(rule_id=rule_id, condition=bool(condition), now=now)
        if not state:
            return []
        if hands_visible == 0 and (now - state.started_at) < duration_s and not inputs.rapid_hand_movement:
            return []

        return self._emit_once_with_cooldown(
            rule_id=rule_id,
            state=state,
            now=now,
            cooldown_s=8.0,
            build_alert=lambda: ProctoringAlert(
                session_id=inputs.session_id,
                rule_id=rule_id,
                label="Suspicious hand movement / hands hidden",
                severity="high",
                risk_score=60,
                started_at=state.started_at,
                last_seen_at=state.last_seen_at,
                frame_id=inputs.frame_id,
                evidence={
                    "hands_visible_count": hands_visible,
                    "rapid_hand_movement": bool(inputs.rapid_hand_movement),
                    "duration_s": round(now - state.started_at, 2),
                },
            ),
        )

    def _rule_abnormal_blink(self, inputs: ProctoringInputs, now: float) -> list[ProctoringAlert]:
        rule_id = "ABNORMAL_BLINK_PATTERN"
        blink_rate = inputs.blink_rate_per_min
        sustained = inputs.sustained_eye_closure_s
        condition = bool(
            (blink_rate is not None and blink_rate > 30.0)
            or (sustained is not None and sustained > 2.0)
        )
        state = self._activate_if_true(rule_id=rule_id, condition=condition, now=now)
        if not state:
            return []

        return self._emit_once_with_cooldown(
            rule_id=rule_id,
            state=state,
            now=now,
            cooldown_s=15.0,
            build_alert=lambda: ProctoringAlert(
                session_id=inputs.session_id,
                rule_id=rule_id,
                label="Abnormal blink pattern",
                severity="medium",
                risk_score=50,
                started_at=state.started_at,
                last_seen_at=state.last_seen_at,
                frame_id=inputs.frame_id,
                evidence={"blink_rate_per_min": blink_rate, "sustained_eye_closure_s": sustained},
            ),
        )

    def _rule_repeated_eye_closure(self, inputs: ProctoringInputs, now: float) -> list[ProctoringAlert]:
        rule_id = "REPEATED_EYE_CLOSURE"

        left_closed = bool(inputs.left_eye_closed) if inputs.left_eye_closed is not None else False
        right_closed = bool(inputs.right_eye_closed) if inputs.right_eye_closed is not None else False
        closure = left_closed or right_closed

        window_s = float(getattr(settings, "PROCTOR_EYE_CLOSURE_REPEAT_WINDOW_SECONDS", 30.0))
        if self._eye_closure_repeat_window_started_at is None or (now - self._eye_closure_repeat_window_started_at) > window_s:
            self._eye_closure_repeat_window_started_at = now
            self._eye_closure_repeats = 0

        if closure and (inputs.sustained_eye_closure_s or 0.0) > 1.0:
            self._eye_closure_repeats += 1

        condition = self._eye_closure_repeats >= 3
        state = self._activate_if_true(rule_id=rule_id, condition=condition, now=now)
        if not state:
            return []

        return self._emit_once_with_cooldown(
            rule_id=rule_id,
            state=state,
            now=now,
            cooldown_s=20.0,
            build_alert=lambda: ProctoringAlert(
                session_id=inputs.session_id,
                rule_id=rule_id,
                label="Repeated eye closure",
                severity="medium",
                risk_score=45,
                started_at=state.started_at,
                last_seen_at=state.last_seen_at,
                frame_id=inputs.frame_id,
                evidence={"repeat_count": self._eye_closure_repeats, "window_s": window_s},
            ),
        )

