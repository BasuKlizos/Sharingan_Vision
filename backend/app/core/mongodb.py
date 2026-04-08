from __future__ import annotations

import asyncio
from typing import Optional

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase

from app.core.config import settings
from app.core.singleton import SingletonMeta


class MongoClient(metaclass=SingletonMeta):
    """
    Async MongoDB client manager (Motor) with lifecycle handling.
    """

    def __init__(self) -> None:
        self._client: Optional[AsyncIOMotorClient] = None
        self._db: Optional[AsyncIOMotorDatabase] = None

    def _create_client(self) -> AsyncIOMotorClient:
        return AsyncIOMotorClient(settings.MONGODB_URI)

    async def connect(self, retries: int = 5, delay: int = 2) -> None:
        for attempt in range(1, retries + 1):
            try:
                self._client = self._create_client()
                self._db = self._client.get_database(settings.MONGODB_DB)
                # quick connectivity check
                await self._client.admin.command("ping")
                return
            except Exception:
                if attempt == retries:
                    raise
                await asyncio.sleep(delay)

    async def disconnect(self) -> None:
        if self._client is not None:
            self._client.close()
        self._client = None
        self._db = None

    def get_db(self) -> AsyncIOMotorDatabase:
        if self._db is None:
            raise RuntimeError("MongoDB client is not initialized. Call connect() first.")
        return self._db


mongo_manager = MongoClient()

