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
        """
        Initialize a Redis client manager.

        This method initializes a Redis client manager with an optional client
        instance. If no client instance is provided, the client instance will
        be initialized with default settings.
        """
        self._client: Optional[Redis] = None

    # Internal creator
    def _create_client(self) -> Redis:
        """
        Create a Redis client instance.

        This method creates a Redis client instance with the connection settings
        defined in the application configuration.

        :return: the Redis client instance
        """
        return Redis(
            host=settings.REDIS_HOST,
            port=settings.REDIS_PORT,
            db=settings.REDIS_DB,
            socket_timeout=settings.REDIS_TIMEOUT,
            decode_responses=True,
        )

    # Startup (with retry)
    async def connect(self, retries: int = 5, delay: int = 2) -> None:
        """
        Establish a connection to Redis with retries and delay.

        This method will attempt to connect to Redis up to `retries` times with
        a delay of `delay` seconds between each attempt. If all attempts fail,
        it will raise a RuntimeError with a message indicating that the connection
        failed.

        :param retries: int: number of times to attempt to connect to Redis
        :param delay: int: delay in seconds between each attempt
        :raises RuntimeError: if the connection to Redis failed
        """
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
        """
        Close the Redis client and clean up resources.

        This method is idempotent: multiple calls will not result in multiple
        close operations. If the client was not initialized (i.e., connect() was
        not called), this method does nothing.

        :raises RuntimeError: if the client was not initialized
        """
        if self._client:
            await self._client.close()

    # Getter (recommended method)
    def get_client(self) -> Redis:
        """
        Get the Redis client instance.

        This method will return the Redis client instance if it is initialized.
        Otherwise, it will raise a RuntimeError with a message indicating that the
        client is not initialized.

        :raises RuntimeError: if the client is not initialized
        :return: the Redis client instance
        """
        if self._client is None:
            raise RuntimeError("Redis client is not initialized. Call connect() first.")
        return self._client


# Singleton instance
redis_manager = RedisClient()
