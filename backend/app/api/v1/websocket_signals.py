from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from pydantic import ValidationError

from app.logger import logger
from app.modules.signaling.websocket_manager import InMemoryConnectionManager
from app.modules.signaling.service import SignalingService
from app.modules.signaling.schemas import SignalMessage
from app.common.exceptions import (
    SignalingError,
    RoomFullError,
    InvalidMessageError,
)

router = APIRouter()

manager = InMemoryConnectionManager()
service = SignalingService(manager)


@router.websocket("/ws/{room_id}")
async def websocket_endpoint(websocket: WebSocket, room_id: str):
    """
    WebSocket endpoint for WebRTC signaling.

    Handles:
    - Connection to a signaling room (interview session)
    - Exchange of signaling messages:
        - offer
        - answer
        - ice-candidate
        - join / leave
    - Message validation and relay between peers

    Args:
        websocket (WebSocket): Active WebSocket connection
        room_id (str): Unique identifier for interview session (call_id)

    Notes:
        - Only 2 participants allowed per room
        - Candidate sends offer
        - Interviewer sends answer
        - ICE candidates exchanged both ways

    Future Enhancements:
        - Token-based authentication
        - Role validation via query params
    """
    try:
        await manager.connect(room_id, websocket)

    except RoomFullError as e:
        await websocket.accept()
        await websocket.send_json({
            "type": "error",
            "data": {"message": str(e)}
        })
        await websocket.close(code=1008)
        return

    logger.info(f"Client connected room={room_id} total={manager.room_size(room_id)}")

    try:
        while True:
            try:
                raw_data = await websocket.receive_json()
            except Exception:
                logger.warning("Invalid JSON received")
                await websocket.send_json({
                    "type": "error",
                    "data": {"message": "Invalid JSON"}
                })
                continue

            try:
                message = SignalMessage(**raw_data)
            except ValidationError:
                await websocket.send_json({
                    "type": "error",
                    "data": {"message": "Invalid message format"}
                })
                continue

            logger.debug(
                f"Signal received room={room_id} type={message.type} role={message.role}"
            )

            try:
                await service.handle_message(
                    room_id,
                    websocket,
                    message.model_dump()
                )

            except InvalidMessageError as e:
                await websocket.send_json({
                    "type": "error",
                    "data": {"message": str(e)}
                })

            except SignalingError as e:
                await websocket.send_json({
                    "type": "error",
                    "data": {"message": str(e)}
                })

            except Exception:
                logger.exception(f"Unexpected signaling error room={room_id}")
                await websocket.send_json({
                    "type": "error",
                    "data": {"message": "Internal server error"}
                })

    except WebSocketDisconnect:
        logger.info(f"Client disconnected from room={room_id}")

    finally:
        await manager.disconnect(room_id, websocket)

        await manager.relay(
            room_id,
            websocket,
            {
                "type": "leave",
                "data": {"message": "Peer disconnected"}
            }
        )

        logger.info(f"Cleanup done for room={room_id}")