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
    VERSION: str = "1.0.0"
    
    # Environment
    DEBUG: bool = config("DEBUG", cast=bool, default=False)
    
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
    DETECTION_FPS: int = 5  # frames per second (optimization)
    MAX_NO_FACE_SECONDS: int = 10  # trigger alert



@lru_cache()
def get_settings():
    settings = Settings()
    return settings


settings = get_settings()

