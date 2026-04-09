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
    updated_session = session_monitoring_store.save_calibration(request.session_id, calibration)

    logger.info(
        "[Calibration] Saved safe zone | session_id=%s min_x=%.2f max_x=%.2f min_y=%.2f max_y=%.2f",
        request.session_id,
        request.min_x,
        request.max_x,
        request.min_y,
        request.max_y,
    )

    if updated_session and updated_session.calibration_zone_definition:
        logger.info(
            "[Calibration] Zone thresholds created | session_id=%s good=20%% looking_away=30%% far_away=40%%",
            request.session_id,
        )

    return {
        "success": True,
        "sessionId": request.session_id,
        "zoneDefinition": updated_session.calibration_zone_definition if updated_session else None,
        "latestZoneAssessment": updated_session.latest_zone_assessment if updated_session else None,
    }

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

    updated_session = session_monitoring_store.update_current_view(request)

    logger.info(
        "[CurrentView] Stored snapshot | session_id=%s min_x=%.2f max_x=%.2f min_y=%.2f max_y=%.2f",
        request.session_id,
        request.current_view.min_x,
        request.current_view.max_x,
        request.current_view.min_y,
        request.current_view.max_y,
    )

    if updated_session and updated_session.latest_zone_assessment:
        zone_assessment = updated_session.latest_zone_assessment
        logger.info(
            "[CurrentView] Zone assessment | session_id=%s status=%s drift_percent=%.2f",
            request.session_id,
            zone_assessment["status"],
            zone_assessment["driftPercent"],
        )

    return {
        "success": True,
        "zoneAssessment": updated_session.latest_zone_assessment if updated_session else None,
    }
