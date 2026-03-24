from collections import defaultdict
from typing import Dict, List
from fastapi import WebSocket

from app.modules.signaling.interfaces import BaseConnectionManager
from app.common.exceptions import RoomFullError


class InMemoryConnectionManager(BaseConnectionManager):

    def __init__(self):
        self.rooms: Dict[str, List[WebSocket]] = defaultdict(list)

    async def connect(self, room_id: str, websocket: WebSocket):

        if len(self.rooms[room_id]) >= 2:
            raise RoomFullError("Room is full")

        await websocket.accept()
        self.rooms[room_id].append(websocket)

    async def disconnect(self, room_id: str, websocket: WebSocket):
        if room_id in self.rooms and websocket in self.rooms[room_id]:
            self.rooms[room_id].remove(websocket)
            if not self.rooms[room_id]:
                del self.rooms[room_id]

    async def relay(self, room_id: str, sender: WebSocket, message: Dict):
        for client in self.rooms.get(room_id, []):
            if client != sender:
                try:
                    await client.send_json(message)
                except Exception:
                    pass

    def room_size(self, room_id: str) -> int:
        return len(self.rooms.get(room_id, []))