from app.modules.signaling.service import WebRTCService

def get_webrtc_service() -> WebRTCService:
    """Provide WebRTC service instance."""
    return WebRTCService()