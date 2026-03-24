from redis.asyncio import Redis
from app.core.redis import redis_manager


async def get_redis() -> Redis:
    return redis_manager.get_client()