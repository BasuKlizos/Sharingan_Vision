import base64
import statistics
from typing import TypedDict

import cv2
import numpy as np

from app.core.config import settings
from app.logger import logger
from app.modules.precheck.schemas import (
    LightingFrameResult,
    LightingPrecheckResponse,
    LightingSummary,
)


class LightingMetrics(TypedDict):
    status: str
    brightness_mean: float
    dark_pixel_ratio: float
    bright_pixel_ratio: float


class LightingPrecheckService:
    STATUS_OK = "ok"
    STATUS_TOO_DARK = "too_dark"
    STATUS_TOO_BRIGHT = "too_bright"
    STATUS_INVALID_FRAME = "invalid_frame"
    STATUS_INVALID_FRAMES = "invalid_frames"

    def __init__(self):
        self.min_brightness = float(getattr(settings, "PRECHECK_MIN_BRIGHTNESS", 70.0))
        self.max_brightness = float(getattr(settings, "PRECHECK_MAX_BRIGHTNESS", 190.0))
        self.max_dark_ratio = float(getattr(settings, "PRECHECK_MAX_DARK_RATIO", 0.35))
        self.max_bright_ratio = float(getattr(settings, "PRECHECK_MAX_BRIGHT_RATIO", 0.25))
        self.dark_pixel_threshold = int(getattr(settings, "PRECHECK_DARK_PIXEL_THRESHOLD", 45))
        self.bright_pixel_threshold = int(getattr(settings, "PRECHECK_BRIGHT_PIXEL_THRESHOLD", 225))

    def evaluate_frames(self, frames: list[str]) -> LightingPrecheckResponse:
        logger.info(f"[Precheck] Processing lighting precheck | frames={len(frames)}")
        frame_results: list[LightingFrameResult] = []
        valid_metrics: list[LightingMetrics] = []

        for index, encoded_frame in enumerate(frames):
            image = self._decode_frame(encoded_frame)
            if image is None:
                frame_results.append(
                    LightingFrameResult(
                        index=index,
                        valid=False,
                        status=self.STATUS_INVALID_FRAME,
                        message=self._status_message(self.STATUS_INVALID_FRAME),
                    )
                )
                continue

            metrics = self._measure_lighting(image)
            frame_results.append(
                LightingFrameResult(
                    index=index,
                    valid=True,
                    status=metrics["status"],
                    message=self._status_message(metrics["status"]),
                    brightness_mean=metrics["brightness_mean"],
                    dark_pixel_ratio=metrics["dark_pixel_ratio"],
                    bright_pixel_ratio=metrics["bright_pixel_ratio"],
                )
            )
            valid_metrics.append(metrics)

        summary = self._build_summary(valid_metrics)
        ok, status = self._decide(summary, valid_metrics)
        logger.info(
            f"[Precheck] Lighting precheck result | status={status} ok={ok} "
            f"valid_frames={len(valid_metrics)}/{len(frames)}"
        )

        return LightingPrecheckResponse(
            ok=ok,
            status=status,
            message=self._status_message(status),
            checked_frames=len(frames),
            valid_frames=len(valid_metrics),
            summary=summary,
            frames=frame_results,
        )

    def _decode_frame(self, encoded_frame: str) -> np.ndarray | None:
        if not encoded_frame:
            return None

        payload = encoded_frame.strip()
        payload = payload.split(",", 1)[1] if "," in payload else payload

        try:
            raw_bytes = base64.b64decode(payload, validate=True)
        except ValueError:
            return None

        if not raw_bytes:
            return None

        np_buffer = np.frombuffer(raw_bytes, dtype=np.uint8)
        image = cv2.imdecode(np_buffer, cv2.IMREAD_COLOR)
        return image

    def _measure_lighting(self, image: np.ndarray) -> LightingMetrics:
        grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        brightness_mean = float(np.mean(grayscale))
        dark_pixel_ratio = float(np.mean(grayscale <= self.dark_pixel_threshold))
        bright_pixel_ratio = float(np.mean(grayscale >= self.bright_pixel_threshold))

        if brightness_mean < self.min_brightness or dark_pixel_ratio > self.max_dark_ratio:
            status = self.STATUS_TOO_DARK
        elif brightness_mean > self.max_brightness or bright_pixel_ratio > self.max_bright_ratio:
            status = self.STATUS_TOO_BRIGHT
        else:
            status = self.STATUS_OK

        return {
            "status": status,
            "brightness_mean": round(brightness_mean, 2),
            "dark_pixel_ratio": round(dark_pixel_ratio, 4),
            "bright_pixel_ratio": round(bright_pixel_ratio, 4),
        }

    def _build_summary(self, valid_metrics: list[LightingMetrics]) -> LightingSummary:
        if not valid_metrics:
            return LightingSummary(
                brightness_mean=None,
                dark_pixel_ratio=None,
                bright_pixel_ratio=None,
                min_brightness=self.min_brightness,
                max_brightness=self.max_brightness,
                max_dark_ratio=self.max_dark_ratio,
                max_bright_ratio=self.max_bright_ratio,
            )

        return LightingSummary(
            brightness_mean=round(statistics.median(m["brightness_mean"] for m in valid_metrics), 2),
            dark_pixel_ratio=round(statistics.median(m["dark_pixel_ratio"] for m in valid_metrics), 4),
            bright_pixel_ratio=round(statistics.median(m["bright_pixel_ratio"] for m in valid_metrics), 4),
            min_brightness=self.min_brightness,
            max_brightness=self.max_brightness,
            max_dark_ratio=self.max_dark_ratio,
            max_bright_ratio=self.max_bright_ratio,
        )

    def _decide(
        self,
        summary: LightingSummary,
        valid_metrics: list[LightingMetrics],
    ) -> tuple[bool, str]:
        if not valid_metrics:
            return False, self.STATUS_INVALID_FRAMES

        if summary.brightness_mean is None:
            return False, self.STATUS_INVALID_FRAMES

        if (
            summary.brightness_mean < self.min_brightness
            or (summary.dark_pixel_ratio is not None and summary.dark_pixel_ratio > self.max_dark_ratio)
        ):
            return False, self.STATUS_TOO_DARK

        if (
            summary.brightness_mean > self.max_brightness
            or (summary.bright_pixel_ratio is not None and summary.bright_pixel_ratio > self.max_bright_ratio)
        ):
            return False, self.STATUS_TOO_BRIGHT

        return True, self.STATUS_OK

    @staticmethod
    def _status_message(status: str) -> str:
        messages = {
            LightingPrecheckService.STATUS_OK: "Lighting looks good. You can start WebRTC.",
            LightingPrecheckService.STATUS_TOO_DARK: "Lighting is too dark. Increase front lighting before starting WebRTC.",
            LightingPrecheckService.STATUS_TOO_BRIGHT: "Lighting is too bright. Reduce glare or strong backlight before starting WebRTC.",
            LightingPrecheckService.STATUS_INVALID_FRAME: "One of the frames could not be decoded.",
            LightingPrecheckService.STATUS_INVALID_FRAMES: "No valid frames were provided for lighting precheck.",
        }
        return messages.get(status, "Lighting precheck completed.")
