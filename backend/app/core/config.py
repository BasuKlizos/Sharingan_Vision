from pydantic import field_validator
from pydantic_settings import BaseSettings
from functools import lru_cache
from pathlib import Path
from starlette.config import Config

_DEFAULT_ENV_FILES = [
    Path(__file__).resolve().parents[1] / ".env",  # backend/app/.env
    Path(__file__).resolve().parents[3] / "env" / "backend" / ".env",  # env/backend/.env
]

_env_file = next((str(p) for p in _DEFAULT_ENV_FILES if p.exists()), None)
config = Config(env_file=_env_file)

def _parse_bool(value: str | None, *, default: bool) -> bool:
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on", "debug"}:
        return True
    if normalized in {"0", "false", "no", "n", "off", "release", "prod", "production"}:
        return False
    # Fall back to default instead of crashing at import-time.
    return default


class Settings(BaseSettings):
    """
    Application settings class with default values
    """

    APP_NAME: str = "AI Interview Monitoring System"
    PROJECT_NAME: str = "ai-interview-backend"
    DESCRIPTION: str = "Backend for AI Interview Monitoring System"
    VERSION: str = "1.0.0"

    # Environment
    DEBUG: bool = False

    @field_validator("DEBUG", mode="before")
    @classmethod
    def _coerce_debug(cls, value):
        if isinstance(value, bool):
            return value
        if value is None:
            # Prefer explicit env var, then app/.env, then default.
            return _parse_bool(config("DEBUG", cast=str, default=None), default=False)
        return _parse_bool(str(value), default=False)

    API_V1_STR: str = config("API_V1_STR", default="/api/v1")

    # Server settings
    HOST: str = config("HOST", default="127.0.0.1")
    PORT: int = config("PORT", default=8000, cast=int)

    # Redis Credentials
    REDIS_PORT: int = config("REDIS_PORT", cast=int, default=6379)
    REDIS_HOST: str = config("REDIS_HOST", cast=str, default="localhost")
    REDIS_DB: int = config("REDIS_DB", cast=int, default=0)
    REDIS_TIMEOUT: int = config("REDIS_TIMEOUT", cast=int, default=5)

    # MongoDB
    MONGODB_URI: str = config("MONGODB_URI", cast=str, default="mongodb://localhost:27017/")
    MONGODB_DB: str = config("MONGODB_DB", cast=str, default="sharingan_vision")

    # WebSocket / Signaling
    WS_HEARTBEAT_INTERVAL: int = 30
    WS_TIMEOUT: int = 60
    MAX_ROOM_USERS: int = 2  # interviewer + candidate

    # WebRTC Config
    STUN_SERVER: str = "stun:stun.l.google.com:19302"
    TURN_SERVER: str = "turn:free.expressturn.com:3478"
    TURN_USERNAME: str = "000000002090847582"
    TURN_PASSWORD: str = "5NTdASVWw8AreIFCZHrGBweFDTA="

    # AI / Camera Detection
    ENABLE_CAMERA_MONITORING: bool = True
    FACE_DETECTION_MODEL: str = "mediapipe"  # or opencv, dlib
    DETECTION_FPS: int = 10  # frames per second (optimization)
    MAX_NO_FACE_SECONDS: int = 10  # trigger alert
    ENABLE_CROP: bool = config("ENABLE_CROP", cast=bool, default=True)
    CROP_PERCENT: float = config("CROP_PERCENT", cast=float, default=0.8)

    # YOLO (video)
    ENABLE_YOLO: bool = config("ENABLE_YOLO", cast=bool, default=False)
    YOLO_MODEL_PATH: str = config("YOLO_MODEL_PATH", default="yolov8n.pt")
    YOLO_CONF_THRESHOLD: float = config("YOLO_CONF_THRESHOLD", cast=float, default=0.25)
    YOLO_MAX_WIDTH: int = config("YOLO_MAX_WIDTH", cast=int, default=640)

    # Proctoring
    PROCTORING_ENABLED: bool = config("PROCTORING_ENABLED", cast=bool, default=True)
    PROCTOR_STORE_BACKEND: str = config("PROCTOR_STORE_BACKEND", cast=str, default="dual")
    PROCTOR_METRICS_COLLECTION: str = config(
        "PROCTOR_METRICS_COLLECTION", cast=str, default="proctor_metrics"
    )
    PROCTOR_FLUSH_BATCH_SIZE: int = config("PROCTOR_FLUSH_BATCH_SIZE", cast=int, default=25)
    PROCTOR_FLUSH_INTERVAL_SECONDS: float = config(
        "PROCTOR_FLUSH_INTERVAL_SECONDS", cast=float, default=1.5
    )
    PROCTOR_FLUSH_QUEUE_SIZE: int = config("PROCTOR_FLUSH_QUEUE_SIZE", cast=int, default=2000)
    PROCTOR_REDIS_MAX_ITEMS: int = config("PROCTOR_REDIS_MAX_ITEMS", cast=int, default=500)
    PROCTOR_METRIC_SAMPLE_SECONDS: float = config(
        "PROCTOR_METRIC_SAMPLE_SECONDS", cast=float, default=2.0
    )
    PROCTOR_SUSPICIOUS_SAMPLE_SECONDS: float = config(
        "PROCTOR_SUSPICIOUS_SAMPLE_SECONDS", cast=float, default=0.5
    )
    PROCTOR_HIGH_RISK_THRESHOLD: float = config(
        "PROCTOR_HIGH_RISK_THRESHOLD", cast=float, default=0.65
    )
    PROCTOR_ALERT_COOLDOWN_SECONDS: float = config(
        "PROCTOR_ALERT_COOLDOWN_SECONDS", cast=float, default=10.0
    )
    PROCTOR_NO_FACE_SECONDS: float = config("PROCTOR_NO_FACE_SECONDS", cast=float, default=3.0)
    PROCTOR_HEAD_AWAY_SECONDS: float = config("PROCTOR_HEAD_AWAY_SECONDS", cast=float, default=2.5)
    PROCTOR_GAZE_AWAY_SECONDS: float = config("PROCTOR_GAZE_AWAY_SECONDS", cast=float, default=2.0)
    PROCTOR_HANDS_HIDDEN_SECONDS: float = config(
        "PROCTOR_HANDS_HIDDEN_SECONDS", cast=float, default=3.0
    )
    PROCTOR_DEVICE_HOLD_SECONDS: float = config(
        "PROCTOR_DEVICE_HOLD_SECONDS", cast=float, default=1.0
    )
    PROCTOR_MATERIAL_HOLD_SECONDS: float = config(
        "PROCTOR_MATERIAL_HOLD_SECONDS", cast=float, default=1.0
    )
    PROCTOR_MULTI_FACE_HOLD_SECONDS: float = config(
        "PROCTOR_MULTI_FACE_HOLD_SECONDS", cast=float, default=1.0
    )
    PROCTOR_HEAD_YAW_THRESHOLD: float = config(
        "PROCTOR_HEAD_YAW_THRESHOLD", cast=float, default=0.35
    )
    PROCTOR_BLINK_RATE_THRESHOLD: float = config(
        "PROCTOR_BLINK_RATE_THRESHOLD", cast=float, default=35.0
    )
    PROCTOR_EYE_CLOSURE_SECONDS: float = config(
        "PROCTOR_EYE_CLOSURE_SECONDS", cast=float, default=2.0
    )
    PROCTOR_EYE_CLOSURE_REPEAT_WINDOW_SECONDS: float = config(
        "PROCTOR_EYE_CLOSURE_REPEAT_WINDOW_SECONDS", cast=float, default=10.0
    )
    PROCTOR_EYE_CLOSURE_REPEAT_COUNT: int = config(
        "PROCTOR_EYE_CLOSURE_REPEAT_COUNT", cast=int, default=3
    )

    # Lighting Precheck
    PRECHECK_MIN_BRIGHTNESS: float = config("PRECHECK_MIN_BRIGHTNESS", cast=float, default=70.0)
    PRECHECK_MAX_BRIGHTNESS: float = config("PRECHECK_MAX_BRIGHTNESS", cast=float, default=190.0)
    PRECHECK_MAX_DARK_RATIO: float = config("PRECHECK_MAX_DARK_RATIO", cast=float, default=0.35)
    PRECHECK_MAX_BRIGHT_RATIO: float = config("PRECHECK_MAX_BRIGHT_RATIO", cast=float, default=0.25)
    PRECHECK_DARK_PIXEL_THRESHOLD: int = config(
        "PRECHECK_DARK_PIXEL_THRESHOLD", cast=int, default=45
    )
    PRECHECK_BRIGHT_PIXEL_THRESHOLD: int = config(
        "PRECHECK_BRIGHT_PIXEL_THRESHOLD", cast=int, default=225
    )

    # Video Recording & Storage
    ENABLE_VIDEO_RECORDING: bool = config("ENABLE_VIDEO_RECORDING", cast=bool, default=True)
    VIDEO_RECORDING_DIR: str = config("VIDEO_RECORDING_DIR", default="./recordings")
    VIDEO_RECORDING_FPS: int = config("VIDEO_RECORDING_FPS", cast=int, default=30)

    # S3 Upload Configuration
    ENABLE_S3_UPLOAD: bool = config("ENABLE_S3_UPLOAD", cast=bool, default=False)
    AWS_ACCESS_KEY_ID: str = config("AWS_ACCESS_KEY_ID", default="")
    AWS_SECRET_ACCESS_KEY: str = config("AWS_SECRET_ACCESS_KEY", default="")
    S3_BUCKET_NAME: str = config("S3_BUCKET_NAME", default="")
    S3_REGION: str = config("S3_REGION", default="us-east-1")
    S3_VIDEO_PREFIX: str = config("S3_VIDEO_PREFIX", default="videos")
    S3_UPLOAD_CLEANUP_LOCAL: bool = config("S3_UPLOAD_CLEANUP_LOCAL", cast=bool, default=True)


@lru_cache()
def get_settings():
    settings = Settings()
    return settings


settings = get_settings()
