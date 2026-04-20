from __future__ import annotations

from typing import Any

from app.modules.monitoring.schemas import CalibrationData, CurrentViewData

GOOD_ZONE_THRESHOLD = 0.30
LOOKING_AWAY_THRESHOLD = 0.55
FAR_AWAY_THRESHOLD = 0.80


def _build_square(min_x: float, max_x: float, min_y: float, max_y: float) -> dict[str, float]:
    width = float(max_x) - float(min_x)
    height = float(max_y) - float(min_y)
    side = max(width, height)

    center_x = (float(min_x) + float(max_x)) / 2.0
    center_y = (float(min_y) + float(max_y)) / 2.0
    half_side = side / 2.0

    return {
        "minX": center_x - half_side,
        "maxX": center_x + half_side,
        "minY": center_y - half_side,
        "maxY": center_y + half_side,
        "centerX": center_x,
        "centerY": center_y,
        "side": side,
    }


def _expand_square(square: dict[str, float], threshold_ratio: float) -> dict[str, float]:
    padding = square["side"] * float(threshold_ratio)
    return {
        "minX": square["minX"] - padding,
        "maxX": square["maxX"] + padding,
        "minY": square["minY"] - padding,
        "maxY": square["maxY"] + padding,
    }


def build_zone_definition(calibration: CalibrationData) -> dict[str, Any]:
    base_square = _build_square(
        min_x=calibration.min_x,
        max_x=calibration.max_x,
        min_y=calibration.min_y,
        max_y=calibration.max_y,
    )

    return {
        "baseSquare": base_square,
        "zones": {
            "good": {
                "thresholdPercent": 30,
                "bounds": _expand_square(base_square, GOOD_ZONE_THRESHOLD),
            },
            "lookingAway": {
                "thresholdPercent": 55,
                "bounds": _expand_square(base_square, LOOKING_AWAY_THRESHOLD),
            },
            "farAway": {
                "thresholdPercent": 80,
                "bounds": _expand_square(base_square, FAR_AWAY_THRESHOLD),
            },
        },
    }


def assess_current_view_against_calibration(
    calibration: CalibrationData,
    current_view: CurrentViewData,
) -> dict[str, Any]:
    zone_definition = build_zone_definition(calibration)
    base_square = zone_definition["baseSquare"]
    current_square = _build_square(
        min_x=current_view.min_x,
        max_x=current_view.max_x,
        min_y=current_view.min_y,
        max_y=current_view.max_y,
    )

    side = max(base_square["side"], 1.0)
    drift_x = abs(current_square["centerX"] - base_square["centerX"])
    drift_y = abs(current_square["centerY"] - base_square["centerY"])
    drift_ratio = max(drift_x, drift_y) / side

    if drift_ratio <= GOOD_ZONE_THRESHOLD:
        status = "good"
        message = "User is not looking away from the screen"
    elif drift_ratio <= LOOKING_AWAY_THRESHOLD:
        status = "looking_away"
        message = "User is looking away from the screen"
    else:
        status = "far_away"
        message = "Detected looking far away from the screen"

    return {
        "status": status,
        "message": message,
        "driftPercent": round(drift_ratio * 100.0, 2),
        "drift": {
            "x": round(drift_x, 2),
            "y": round(drift_y, 2),
        },
        "thresholds": {
            "good": 30,
            "lookingAway": 55,
            "farAway": 80,
        },
        "calibrationSquare": base_square,
        "currentViewSquare": current_square,
        "zoneDefinition": zone_definition["zones"],
    }
