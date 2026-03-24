from typing import Optional
import asyncio

from redis.asyncio import Redis
from app.core.config import settings
from app.core.singleton import SingletonMeta


class RedisClient(metaclass=SingletonMeta):
    """
    Async Redis client manager with lifecycle handling.
    """

    def __init__(self) -> None:
        self._client: Optional[Redis] = None

    # Internal creator
    def _create_client(self) -> Redis:
        return Redis(
            host=settings.REDIS_HOST,
            port=settings.REDIS_PORT,
            password=settings.REDIS_PASSWORD or None,
            db=settings.REDIS_DB,
            socket_timeout=settings.REDIS_TIMEOUT,
            decode_responses=True,
        )

    # Startup (with retry)
    async def connect(self, retries: int = 5, delay: int = 2) -> None:
        for attempt in range(1, retries + 1):
            try:
                self._client = self._create_client()
                await self._client.ping()
                return
            except Exception as e:
                if attempt == retries:
                    raise RuntimeError("Failed to connect to Redis") from e
                await asyncio.sleep(delay)

    # Shutdown
    async def disconnect(self) -> None:
        if self._client:
            await self._client.close()

    # Getter (recommended method)
    def get_client(self) -> Redis:
        if self._client is None:
            raise RuntimeError("Redis client is not initialized. Call connect() first.")
        return self._client


# Singleton instance
redis_manager = RedisClient()