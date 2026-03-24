from redis.asyncio import Redis
from app.core.redis import redis_manager


async def get_redis() -> Redis:
    """
    Get the Redis client instance.

    This method will return the Redis client instance if it is initialized.
    Otherwise, it will raise a RuntimeError with a message indicating that the
    client is not initialized.

    :return: the Redis client instance
    :raises RuntimeError: if the client is not initialized
    """
    return redis_manager.get_client()