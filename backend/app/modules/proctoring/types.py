from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


@dataclass(frozen=True)
class ProctoringInputs:
    session_id: str
    frame_id: int
    timestamp: float

    # Face
    face_count: int
    face_detected: bool
    face_confidence: Optional[float] = None

    # Head / eye
    head_yaw: Optional[float] = None  # degrees preferred; engine treats as generic float with configurable thresholds
    pitch: Optional[float] = None
    roll: Optional[float] = None
    gaze_direction: Optional[str] = None  # center|left|right|up|down|unknown
    blink_rate_per_min: Optional[float] = None
    sustained_eye_closure_s: Optional[float] = None
    left_eye_closed: Optional[bool] = None
    right_eye_closed: Optional[bool] = None

    # Hands
    hands_visible_count: Optional[int] = None
    rapid_hand_movement: Optional[bool] = None

    # Objects (YOLO-derived)
    phone_detected: bool = False
    phone_confidence: Optional[float] = None
    other_device_detected: bool = False
    other_device_confidence: Optional[float] = None
    unauthorized_materials_detected: bool = False
    unauthorized_materials_confidence: Optional[float] = None
    multiple_persons_detected: bool = False
    multiple_persons_confidence: Optional[float] = None

    # Raw (optional) for debugging; do not persist by default
    extra: Optional[dict[str, Any]] = None


@dataclass(frozen=True)
class ProctoringAlert:
    session_id: str
    rule_id: str
    label: str
    severity: str  # critical|high|medium
    risk_score: int  # 0-100
    started_at: float
    last_seen_at: float
    frame_id: int
    evidence: dict[str, Any]

