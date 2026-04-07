from __future__ import annotations

from fastapi import APIRouter, Body, status
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from app.logger import logger
from app.modules.monitoring.schemas import (
    CalibrationData,
    CalibrationSaveRequest,
    CurrentViewUpdateRequest,
    ViolationEventRequest,
)
from app.modules.monitoring.store import session_monitoring_store, utc_now

router = APIRouter()


def _error_response(status_code: int, message: str) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={
            "success": False,
            "error": message,
        },
    )


def _validation_error_response(default_message: str, exc: ValidationError) -> JSONResponse:
    errors = exc.errors()
    message = errors[0]["msg"] if errors else default_message
    return _error_response(status.HTTP_400_BAD_REQUEST, message or default_message)


@router.post("/calibration/save")
async def save_calibration(payload: dict = Body(...)):
    try:
        request = CalibrationSaveRequest.model_validate(payload)
    except ValidationError as exc:
        return _validation_error_response("Invalid calibration boundaries", exc)

    session = session_monitoring_store.get_session(request.session_id)
    if session is None:
        return _error_response(status.HTTP_404_NOT_FOUND, "Session not found")
    logger.info(
        "[Calibration] Received save request | session_id=%s min_x=%.2f max_x=%.2f min_y=%.2f max_y=%.2f",
        request.session_id,
        request.min_x,
        request.max_x,
        request.min_y,
        request.max_y,
    )
    calibration = CalibrationData(
        minX=request.min_x,
        maxX=request.max_x,
        minY=request.min_y,
        maxY=request.max_y,
        savedAt=utc_now(),
    )
    session_monitoring_store.save_calibration(request.session_id, calibration)

    logger.info(
        "[Calibration] Saved safe zone | session_id=%s min_x=%.2f max_x=%.2f min_y=%.2f max_y=%.2f",
        request.session_id,
        request.min_x,
        request.max_x,
        request.min_y,
        request.max_y,
    )

    return {
        "success": True,
        "sessionId": request.session_id,
    }


@router.post("/violation-event")
async def create_violation_event(payload: dict = Body(...)):
    try:
        request = ViolationEventRequest.model_validate(payload)
    except ValidationError as exc:
        return _validation_error_response("Invalid violation payload", exc)

    session = session_monitoring_store.get_session(request.session_id)
    if session is None:
        return _error_response(status.HTTP_404_NOT_FOUND, "Session not found")

    if session.calibration is None:
        return _error_response(status.HTTP_409_CONFLICT, "Calibration missing for session")

    session_monitoring_store.create_violation_event(request)

    logger.info(
        "[Violation] Stored event | session_id=%s count=%s gaze_x=%.2f gaze_y=%.2f",
        request.session_id,
        session.violation_count,
        request.point.x,
        request.point.y,
    )

    return {"success": True}


@router.post("/current-view")
async def update_current_view(payload: dict = Body(...)):
    logger.info("[CurrentView] Incoming update request")
    try:
        request = CurrentViewUpdateRequest.model_validate(payload)
    except ValidationError as exc:
        return _validation_error_response("Invalid current view payload", exc)

    session = session_monitoring_store.get_session(request.session_id)
    if session is None:
        return _error_response(status.HTTP_404_NOT_FOUND, "Session not found")

    session_monitoring_store.update_current_view(request)

    logger.info(
        "[CurrentView] Stored snapshot | session_id=%s min_x=%.2f max_x=%.2f min_y=%.2f max_y=%.2f",
        request.session_id,
        request.current_view.min_x,
        request.current_view.max_x,
        request.current_view.min_y,
        request.current_view.max_y,
    )

    return {"success": True}
