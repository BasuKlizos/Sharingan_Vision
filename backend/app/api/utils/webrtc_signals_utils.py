from functools import lru_cache

from app.modules.signaling.service import WebRTCService


@lru_cache(maxsize=1)
def get_webrtc_service() -> WebRTCService:
    """Provide WebRTC service instance."""
    return WebRTCService()
