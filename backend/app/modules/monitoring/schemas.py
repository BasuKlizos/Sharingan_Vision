from __future__ import annotations

from datetime import datetime
from numbers import Real
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


class CalibrationSaveRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    session_id: str = Field(alias="sessionId")
    min_x: float = Field(alias="minX")
    max_x: float = Field(alias="maxX")
    min_y: float = Field(alias="minY")
    max_y: float = Field(alias="maxY")

    @field_validator("session_id")
    @classmethod
    def validate_session_id(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("sessionId is required")
        return value.strip()

    @field_validator("min_x", "max_x", "min_y", "max_y", mode="before")
    @classmethod
    def validate_boundary_number(cls, value: object) -> float:
        if isinstance(value, bool) or not isinstance(value, Real):
            raise ValueError("Calibration boundaries must be numeric")
        return float(value)

    @field_validator("max_x")
    @classmethod
    def validate_x_bounds(cls, value: float, info) -> float:
        min_x = info.data.get("min_x")
        if min_x is not None and min_x > value:
            raise ValueError("minX must be less than or equal to maxX")
        return value

    @field_validator("max_y")
    @classmethod
    def validate_y_bounds(cls, value: float, info) -> float:
        min_y = info.data.get("min_y")
        if min_y is not None and min_y > value:
            raise ValueError("minY must be less than or equal to maxY")
        return value


class CalibrationData(BaseModel):
    min_x: float = Field(alias="minX")
    max_x: float = Field(alias="maxX")
    min_y: float = Field(alias="minY")
    max_y: float = Field(alias="maxY")
    saved_at: datetime = Field(alias="savedAt")

    model_config = ConfigDict(populate_by_name=True)


class ViolationPoint(BaseModel):
    x: float
    y: float
    timestamp: Optional[int] = None

    @field_validator("x", "y", mode="before")
    @classmethod
    def validate_coordinate(cls, value: object) -> float:
        if isinstance(value, bool) or not isinstance(value, Real):
            raise ValueError("Violation point coordinates must be numeric")
        return float(value)


class ViolationBoundaries(BaseModel):
    min_x: float = Field(alias="minX")
    max_x: float = Field(alias="maxX")
    min_y: float = Field(alias="minY")
    max_y: float = Field(alias="maxY")

    model_config = ConfigDict(populate_by_name=True)

    @field_validator("min_x", "max_x", "min_y", "max_y", mode="before")
    @classmethod
    def validate_boundary_number(cls, value: object) -> float:
        if isinstance(value, bool) or not isinstance(value, Real):
            raise ValueError("Violation boundaries must be numeric")
        return float(value)

    @field_validator("max_x")
    @classmethod
    def validate_x_bounds(cls, value: float, info) -> float:
        min_x = info.data.get("min_x")
        if min_x is not None and min_x > value:
            raise ValueError("minX must be less than or equal to maxX")
        return value

    @field_validator("max_y")
    @classmethod
    def validate_y_bounds(cls, value: float, info) -> float:
        min_y = info.data.get("min_y")
        if min_y is not None and min_y > value:
            raise ValueError("minY must be less than or equal to maxY")
        return value


class ViolationEventRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    session_id: str = Field(alias="sessionId")
    event_type: Literal["violation-event"] = Field(alias="type")
    point: ViolationPoint
    boundaries: ViolationBoundaries
    timestamp: Optional[datetime] = None

    @field_validator("session_id")
    @classmethod
    def validate_session_id(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("sessionId is required")
        return value.strip()
