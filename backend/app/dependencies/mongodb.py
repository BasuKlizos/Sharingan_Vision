from motor.motor_asyncio import AsyncIOMotorDatabase

from app.core.mongodb import mongo_manager


async def get_mongo_db() -> AsyncIOMotorDatabase:
    return mongo_manager.get_db()
