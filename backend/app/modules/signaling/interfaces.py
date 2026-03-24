from abc import ABC, abstractmethod
from fastapi import WebSocket
from typing import Dict


class BaseConnectionManager(ABC):

    @abstractmethod
    async def connect(self, room_id: str, websocket: WebSocket):
        pass

    @abstractmethod
    async def disconnect(self, room_id: str, websocket: WebSocket):
        pass

    @abstractmethod
    async def relay(self, room_id: str, sender: WebSocket, message: Dict):
        pass

    @abstractmethod
    def room_size(self, room_id: str) -> int:
        pass