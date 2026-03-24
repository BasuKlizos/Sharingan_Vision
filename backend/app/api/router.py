from app.api.v1 import *

from fastapi import APIRouter

api_router = APIRouter()

api_router.include_router(websocket_router, prefix="/ws", tags=["WebSocket Signaling API's Management"])