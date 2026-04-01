from pydantic_settings import BaseSettings
from functools import lru_cache
from typing import Optional
from starlette.config import Config


config = Config(env_file="app/.env")

class Settings(BaseSettings):
    """
    Application settings class with default values
    """
    
    APP_NAME: str = "AI Interview Monitoring System"
    PROJECT_NAME: str = "ai-interview-backend"
    DESCRIPTION: str = "Backend for AI Interview Monitoring System"
    VERSION: str = "1.0.0"
    
    # Environment
    DEBUG: bool = config("DEBUG", cast=bool, default=False)
    
    API_V1_STR: str = config("API_V1_STR", default="/api/v1")

    # Server settings
    HOST: str = config("HOST", default="127.0.0.1")
    PORT: int = config("PORT", default=8000, cast=int)
    
    # Redis Credentials
    REDIS_PORT: int = config("REDIS_PORT", cast=int, default=6379)
    REDIS_HOST: str = config("REDIS_HOST", cast=str, default="localhost")
    REDIS_DB: int = config("REDIS_DB", cast=int, default=0)
    REDIS_TIMEOUT: int = config("REDIS_TIMEOUT", cast=int, default=5)
    
    
    # WebSocket / Signaling
    WS_HEARTBEAT_INTERVAL: int = 30
    WS_TIMEOUT: int = 60
    MAX_ROOM_USERS: int = 2  # interviewer + candidate

    # WebRTC Config
    STUN_SERVER: str = "stun:stun.l.google.com:19302"
    TURN_SERVER: str = ""
    TURN_USERNAME: str = ""
    TURN_PASSWORD: str = ""

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
