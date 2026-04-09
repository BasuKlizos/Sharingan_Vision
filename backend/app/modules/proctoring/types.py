from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass(frozen=True, slots=True)
class ProctoringInputs:
    session_id: str
    frame_id: int
    timestamp: float
    face_count: int
    face_detected: bool
    head_yaw: float = 0.0
    pitch: float = 0.0
    roll: float = 0.0
    gaze_direction: Optional[str] = None
    blink_rate_per_min: float = 0.0
    sustained_eye_closure_s: float = 0.0
    left_eye_closed: Optional[bool] = None
    right_eye_closed: Optional[bool] = None
    hands_visible_count: Optional[int] = None
    rapid_hand_movement: bool = False
    phone_detected: bool = False
    phone_confidence: float = 0.0
    other_device_detected: bool = False
    other_device_confidence: float = 0.0
    unauthorized_materials_detected: bool = False
    unauthorized_materials_confidence: float = 0.0
    multiple_persons_detected: bool = False
    multiple_persons_confidence: float = 0.0
    risk_score: float = 0.0
    suspicious: bool = False
    extra: Optional[dict[str, Any]] = None


@dataclass(frozen=True, slots=True)
class ProctoringAlert:
    session_id: str
    frame_id: int
    rule_id: str
    label: str
    severity: str
    risk_score: float
    started_at: float
    last_seen_at: float
    occurrence_count: int = 1
    evidence: dict[str, Any] = field(default_factory=dict)
