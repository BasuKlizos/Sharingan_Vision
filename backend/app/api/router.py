from app.api.v1 import *

from fastapi import APIRouter

api_router = APIRouter()

api_router.include_router(webrtc_router, prefix="/webrtc", tags=["WebRTC Signaling API's Management"])
api_router.include_router(precheck_router, prefix="/precheck", tags=["Precheck API"])
