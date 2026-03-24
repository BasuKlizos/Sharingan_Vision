from typing import Dict
from fastapi import WebSocket

from app.modules.signaling.interfaces import BaseConnectionManager
from app.common.exceptions import InvalidMessageError


class SignalingService:

    def __init__(self, manager: BaseConnectionManager):
        self.manager = manager

    async def handle_message(self, room_id: str, sender: WebSocket, message: Dict):

        role = message.get("role")
        msg_type = message.get("type")

        if msg_type not in [
            "join",
            "ready",
            "offer",
            "answer",
            "ice-candidate",
            "leave",
        ]:
            raise InvalidMessageError(f"Invalid message type: {msg_type}")

        if msg_type == "offer" and role != "candidate":
            raise InvalidMessageError("Only candidate can send offer")

        if msg_type == "answer" and role != "interviewer":
            raise InvalidMessageError("Only interviewer can send answer")
        
        if msg_type == "join":
            return

        await self.manager.relay(room_id, sender, message)