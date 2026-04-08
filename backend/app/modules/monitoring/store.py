from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from threading import RLock
from typing import Dict, List, Optional
from uuid import uuid4

from app.modules.monitoring.schemas import (
    CalibrationData,
    CurrentViewData,
    CurrentViewUpdateRequest,
    ViolationEventRequest,
)
from app.modules.monitoring.zones import assess_current_view_against_calibration, build_zone_definition


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class ViolationEventRecord:
    id: str
    session_id: str
    event_type: str
    gaze_x: float
    gaze_y: float
    min_x: float
    max_x: float
    min_y: float
    max_y: float
    client_timestamp: Optional[datetime]
    point_timestamp_ms: Optional[int]
    received_at: datetime


@dataclass
class SessionMonitoringRecord:
    session_id: str
    created_at: datetime
    webrtc_status: str = "connected"
    calibration: Optional[CalibrationData] = None
    calibration_zone_definition: Optional[dict] = None
    latest_current_view: Optional[CurrentViewData] = None
    latest_current_view_client_at: Optional[datetime] = None
    latest_current_view_received_at: Optional[datetime] = None
    latest_zone_assessment: Optional[dict] = None
    violation_count: int = 0
    last_violation_at: Optional[datetime] = None
    violation_events: List[ViolationEventRecord] = field(default_factory=list)


class SessionMonitoringStore:
    def __init__(self) -> None:
        self._sessions: Dict[str, SessionMonitoringRecord] = {}
        self._lock = RLock()

    def register_session(self, session_id: str, webrtc_status: str = "connected") -> SessionMonitoringRecord:
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                session = SessionMonitoringRecord(
                    session_id=session_id,
                    created_at=utc_now(),
                    webrtc_status=webrtc_status,
                )
                self._sessions[session_id] = session
                return session

            session.webrtc_status = webrtc_status
            return session

    def get_session(self, session_id: str) -> Optional[SessionMonitoringRecord]:
        with self._lock:
            return self._sessions.get(session_id)

    def mark_session_status(self, session_id: str, webrtc_status: str) -> Optional[SessionMonitoringRecord]:
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return None

            session.webrtc_status = webrtc_status
            return session

    def save_calibration(self, session_id: str, calibration: CalibrationData) -> Optional[SessionMonitoringRecord]:
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return None

            session.calibration = calibration
            session.calibration_zone_definition = build_zone_definition(calibration)
            # Save frontend calibration exactly as received. Zone assessment is
            # computed only when a later current-view update arrives.
            session.latest_zone_assessment = None
            return session

    def create_violation_event(self, payload: ViolationEventRequest) -> Optional[ViolationEventRecord]:
        with self._lock:
            session = self._sessions.get(payload.session_id)
            if session is None:
                return None

            event = ViolationEventRecord(
                id=f"violation-{uuid4()}",
                session_id=payload.session_id,
                event_type=payload.event_type,
                gaze_x=payload.point.x,
                gaze_y=payload.point.y,
                min_x=payload.boundaries.min_x,
                max_x=payload.boundaries.max_x,
                min_y=payload.boundaries.min_y,
                max_y=payload.boundaries.max_y,
                client_timestamp=payload.timestamp,
                point_timestamp_ms=payload.point.timestamp,
                received_at=utc_now(),
            )

            session.violation_events.append(event)
            session.violation_count += 1
            session.last_violation_at = event.received_at
            return event

    def update_current_view(self, payload: CurrentViewUpdateRequest) -> Optional[SessionMonitoringRecord]:
        with self._lock:
            session = self._sessions.get(payload.session_id)
            if session is None:
                return None

            session.latest_current_view = payload.current_view
            session.latest_current_view_client_at = payload.timestamp
            session.latest_current_view_received_at = utc_now()
            if session.calibration is not None:
                session.latest_zone_assessment = assess_current_view_against_calibration(
                    calibration=session.calibration,
                    current_view=payload.current_view,
                )
            else:
                session.latest_zone_assessment = None
            return session

    def update_current_view_for_session(
        self,
        session_id: str,
        current_view: CurrentViewData,
        client_timestamp: Optional[datetime] = None,
    ) -> Optional[SessionMonitoringRecord]:
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return None

            session.latest_current_view = current_view
            session.latest_current_view_client_at = client_timestamp
            session.latest_current_view_received_at = utc_now()
            if session.calibration is not None:
                session.latest_zone_assessment = assess_current_view_against_calibration(
                    calibration=session.calibration,
                    current_view=current_view,
                )
            else:
                session.latest_zone_assessment = None
            return session


session_monitoring_store = SessionMonitoringStore()
