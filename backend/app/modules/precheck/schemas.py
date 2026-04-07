from pydantic import BaseModel, Field


class LightingPrecheckRequest(BaseModel):
    frames: list[str] = Field(
        ...,
        min_length=1,
        description="Base64-encoded image frames. Data URLs are also accepted.",
    )


class LightingFrameResult(BaseModel):
    index: int
    valid: bool
    status: str
    message: str
    brightness_mean: float | None = None
    dark_pixel_ratio: float | None = None
    bright_pixel_ratio: float | None = None


class LightingSummary(BaseModel):
    brightness_mean: float | None = None
    dark_pixel_ratio: float | None = None
    bright_pixel_ratio: float | None = None
    min_brightness: float
    max_brightness: float
    max_dark_ratio: float
    max_bright_ratio: float


class LightingPrecheckResponse(BaseModel):
    ok: bool
    status: str
    message: str
    checked_frames: int
    valid_frames: int
    summary: LightingSummary
    frames: list[LightingFrameResult]
