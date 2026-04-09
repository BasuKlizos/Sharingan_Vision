from __future__ import annotations

import json
from functools import lru_cache
from dataclasses import asdict
from typing import Any, Protocol
from pymongo import UpdateOne

from app.core.config import settings
from app.core.mongodb import mongo_manager
from app.core.redis import redis_manager
from app.logger import logger
from app.modules.proctoring.types import ProctoringAlert, ProctoringInputs


class ProctoringStore(Protocol):
    async def ensure_ready(self) -> None: ...
    async def append_metrics(self, item: ProctoringInputs) -> bool: ...
    async def append_alert(self, item: ProctoringAlert) -> bool: ...
    async def append_metrics_batch(self, items: list[ProctoringInputs]) -> bool: ...
    async def append_alerts_batch(self, items: list[ProctoringAlert]) -> bool: ...
    async def get_latest_alerts(self, session_id: str, limit: int) -> list[dict[str, Any]]: ...
    async def get_latest_metrics(self, session_id: str, limit: int) -> list[dict[str, Any]]: ...


class ProctoringRedisStore:
    """
    Fast-access cache store.
    """

    async def ensure_ready(self) -> None:
        return None

    async def append_metrics(self, item: ProctoringInputs) -> bool:
        return await self.append_metrics_batch([item])

    async def append_alert(self, item: ProctoringAlert) -> bool:
        return await self.append_alerts_batch([item])

    async def append_metrics_batch(self, items: list[ProctoringInputs]) -> bool:
        if not items:
            return True
        client = redis_manager.get_client()
        cache_limit = int(settings.PROCTOR_REDIS_MAX_ITEMS)
        try:
            async with client.pipeline(transaction=False) as pipe:
                for item in items:
                    payload = json.dumps(asdict(item), default=str)
                    key = f"proctor:metrics:{item.session_id}"
                    pipe.lpush(key, payload)
                    pipe.ltrim(key, 0, cache_limit - 1)
                await pipe.execute()
            return True
        except Exception as exc:
            logger.warning("[ProctorStoreRedis] Failed to append metrics | error=%r", exc)
            return False

    async def append_alerts_batch(self, items: list[ProctoringAlert]) -> bool:
        if not items:
            return True
        client = redis_manager.get_client()
        cache_limit = int(settings.PROCTOR_REDIS_MAX_ITEMS)
        try:
            async with client.pipeline(transaction=False) as pipe:
                for item in items:
                    payload = json.dumps(asdict(item), default=str)
                    key = f"proctor:alerts:{item.session_id}"
                    pipe.lpush(key, payload)
                    pipe.ltrim(key, 0, cache_limit - 1)
                await pipe.execute()
            return True
        except Exception as exc:
            logger.warning("[ProctorStoreRedis] Failed to append alerts | error=%r", exc)
            return False

    async def get_latest_alerts(self, session_id: str, limit: int) -> list[dict[str, Any]]:
        client = redis_manager.get_client()
        raw = await client.lrange(f"proctor:alerts:{session_id}", 0, max(0, limit - 1))
        return [json.loads(item) for item in raw]

    async def get_latest_metrics(self, session_id: str, limit: int) -> list[dict[str, Any]]:
        client = redis_manager.get_client()
        raw = await client.lrange(f"proctor:metrics:{session_id}", 0, max(0, limit - 1))
        return [json.loads(item) for item in raw]


class ProctoringMongoStore:
    """
    Persistent audit store in MongoDB.

    Collections:
    - `proctor_metrics`
    - `proctor_alerts`
    """

    def __init__(self) -> None:
        self._metrics_col = settings.PROCTOR_METRICS_COLLECTION
        self._alerts_col = settings.PROCTOR_ALERTS_COLLECTION

    async def ensure_ready(self) -> None:
        db = mongo_manager.get_db()
        ttl_seconds = int(settings.PROCTOR_METRICS_TTL_SECONDS)
        await db[self._metrics_col].create_index(
            [("session_id", 1), ("timestamp", -1)],
            name="idx_proctor_metrics_session_timestamp",
        )
        if ttl_seconds > 0:
            await db[self._metrics_col].create_index(
                "timestamp",
                expireAfterSeconds=ttl_seconds,
                name="idx_proctor_metrics_ttl",
            )
        await db[self._alerts_col].create_index(
            [("session_id", 1), ("last_seen_at", -1)],
            name="idx_proctor_alerts_session_last_seen",
        )
        await db[self._alerts_col].create_index(
            [("session_id", 1), ("rule_id", 1), ("last_seen_at", -1)],
            name="idx_proctor_alerts_session_rule_last_seen",
        )
        await db[self._alerts_col].create_index(
            [("session_id", 1), ("rule_id", 1), ("started_at", 1)],
            unique=True,
            name="uniq_proctor_alert_event",
        )

    async def append_metrics(self, item: ProctoringInputs) -> bool:
        return await self.append_metrics_batch([item])

    async def append_alert(self, item: ProctoringAlert) -> bool:
        return await self.append_alerts_batch([item])

    async def append_metrics_batch(self, items: list[ProctoringInputs]) -> bool:
        if not items:
            return True
        try:
            db = mongo_manager.get_db()
            docs = [asdict(item) for item in items]
            result = await db[self._metrics_col].insert_many(docs, ordered=False)
            return bool(result.inserted_ids)
        except Exception as exc:
            logger.error("[ProctorStoreMongo] Failed to append metrics | error=%r", exc)
            return False

    async def append_alerts_batch(self, items: list[ProctoringAlert]) -> bool:
        if not items:
            return True
        try:
            db = mongo_manager.get_db()
            operations = []
            for item in items:
                doc = asdict(item)
                selector = {
                    "session_id": item.session_id,
                    "rule_id": item.rule_id,
                    "started_at": item.started_at,
                }
                operations.append(
                    UpdateOne(
                        selector,
                        {
                            "$setOnInsert": {
                                "session_id": item.session_id,
                                "rule_id": item.rule_id,
                                "label": item.label,
                                "severity": item.severity,
                                "started_at": item.started_at,
                            },
                            "$set": {
                                "frame_id": item.frame_id,
                                "last_seen_at": item.last_seen_at,
                                "risk_score": item.risk_score,
                                "evidence": doc["evidence"],
                            },
                            "$inc": {
                                "occurrence_count": int(item.occurrence_count),
                            },
                        },
                        upsert=True,
                    )
                )
            result = await db[self._alerts_col].bulk_write(operations, ordered=False)
            return bool(result.upserted_count or result.modified_count or result.matched_count)
        except Exception as exc:
            logger.error("[ProctorStoreMongo] Failed to append alerts | error=%r", exc)
            return False

    async def get_latest_alerts(self, session_id: str, limit: int) -> list[dict[str, Any]]:
        db = mongo_manager.get_db()
        cursor = (
            db[self._alerts_col]
            .find({"session_id": session_id}, {"_id": False})
            .sort("last_seen_at", -1)
            .limit(limit)
        )
        return [doc async for doc in cursor]

    async def get_latest_metrics(self, session_id: str, limit: int) -> list[dict[str, Any]]:
        db = mongo_manager.get_db()
        cursor = (
            db[self._metrics_col]
            .find({"session_id": session_id}, {"_id": False})
            .sort("timestamp", -1)
            .limit(limit)
        )
        return [doc async for doc in cursor]


class ProctoringDualStore:
    """
    Persist to both Redis and MongoDB.
    """

    def __init__(self) -> None:
        self.redis_store = ProctoringRedisStore()
        self.mongo_store = ProctoringMongoStore()

    async def ensure_ready(self) -> None:
        await self.redis_store.ensure_ready()
        await self.mongo_store.ensure_ready()

    async def append_metrics(self, item: ProctoringInputs) -> bool:
        return await self.append_metrics_batch([item])

    async def append_alert(self, item: ProctoringAlert) -> bool:
        return await self.append_alerts_batch([item])

    async def append_metrics_batch(self, items: list[ProctoringInputs]) -> bool:
        redis_ok = await self.redis_store.append_metrics_batch(items)
        mongo_ok = await self.mongo_store.append_metrics_batch(items)
        if not redis_ok and not mongo_ok:
            logger.error("[ProctorStoreDual] Both backends failed for metrics")
        return redis_ok or mongo_ok

    async def append_alerts_batch(self, items: list[ProctoringAlert]) -> bool:
        redis_ok = await self.redis_store.append_alerts_batch(items)
        mongo_ok = await self.mongo_store.append_alerts_batch(items)
        if not redis_ok and not mongo_ok:
            logger.error("[ProctorStoreDual] Both backends failed for alerts")
        return redis_ok or mongo_ok

    async def get_latest_alerts(self, session_id: str, limit: int) -> list[dict[str, Any]]:
        try:
            return await self.mongo_store.get_latest_alerts(session_id, limit)
        except Exception:
            return await self.redis_store.get_latest_alerts(session_id, limit)

    async def get_latest_metrics(self, session_id: str, limit: int) -> list[dict[str, Any]]:
        try:
            return await self.mongo_store.get_latest_metrics(session_id, limit)
        except Exception:
            return await self.redis_store.get_latest_metrics(session_id, limit)


@lru_cache(maxsize=1)
def get_proctoring_store() -> ProctoringStore:
    backend = settings.PROCTOR_STORE_BACKEND.strip().lower()
    if backend == "mongo":
        return ProctoringMongoStore()
    if backend == "redis":
        return ProctoringRedisStore()
    return ProctoringDualStore()
