from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any, Protocol

from app.core.config import settings
from app.core.mongodb import mongo_manager
from app.core.redis import redis_manager
from app.logger import logger
from app.modules.proctoring.types import ProctoringAlert, ProctoringInputs


class ProctoringStore(Protocol):
    async def append_metrics(self, inputs: ProctoringInputs) -> bool: ...
    async def append_alert(self, alert: ProctoringAlert) -> bool: ...
    async def get_latest_alerts(self, session_id: str, limit: int = 50) -> list[dict[str, Any]]: ...
    async def get_latest_metrics(self, session_id: str, limit: int = 200) -> list[dict[str, Any]]: ...


class ProctoringRedisStore:
    """
    Persist only necessary metrics + alerts for a session.

    Storage model (Redis):
    - `proctor:metrics:{session_id}`: capped list of metric snapshots (JSON)
    - `proctor:alerts:{session_id}`: capped list of emitted alerts (JSON)
    """

    def __init__(self, metrics_maxlen: int = 1000, alerts_maxlen: int = 500):
        self.metrics_maxlen = int(metrics_maxlen)
        self.alerts_maxlen = int(alerts_maxlen)

    async def append_metrics(self, inputs: ProctoringInputs) -> bool:
        # Only store aggregated/necessary fields (no raw landmarks / boxes).
        payload: dict[str, Any] = {
            "session_id": inputs.session_id,
            "frame_id": inputs.frame_id,
            "timestamp": float(inputs.timestamp),
            "face_detected": bool(inputs.face_detected),
            "face_count": int(inputs.face_count),
            "head_yaw": inputs.head_yaw,
            "gaze_direction": inputs.gaze_direction,
            "blink_rate_per_min": inputs.blink_rate_per_min,
            "hands_visible_count": inputs.hands_visible_count,
            "rapid_hand_movement": inputs.rapid_hand_movement,
            "phone_detected": bool(inputs.phone_detected),
            "unauthorized_materials_detected": bool(inputs.unauthorized_materials_detected),
            "multiple_persons_detected": bool(inputs.multiple_persons_detected),
        }

        key = f"proctor:metrics:{inputs.session_id}"
        try:
            client = redis_manager.get_client()
            await client.lpush(key, json.dumps(payload))
            await client.ltrim(key, 0, self.metrics_maxlen - 1)
            return True
        except Exception as exc:
            logger.warning(
                "[ProctorStoreRedis] Failed to append metrics | session_id=%s key=%s error=%r",
                inputs.session_id,
                key,
                exc,
            )
            return False

    async def append_alert(self, alert: ProctoringAlert) -> bool:
        key = f"proctor:alerts:{alert.session_id}"
        try:
            client = redis_manager.get_client()
            await client.lpush(key, json.dumps(asdict(alert)))
            await client.ltrim(key, 0, self.alerts_maxlen - 1)
            return True
        except Exception as exc:
            logger.warning(
                "[ProctorStoreRedis] Failed to append alert | session_id=%s key=%s error=%r",
                alert.session_id,
                key,
                exc,
            )
            return False

    async def get_latest_alerts(self, session_id: str, limit: int = 50) -> list[dict[str, Any]]:
        key = f"proctor:alerts:{session_id}"
        client = redis_manager.get_client()
        raw = await client.lrange(key, 0, max(0, int(limit) - 1))
        return [json.loads(item) for item in raw]

    async def get_latest_metrics(self, session_id: str, limit: int = 200) -> list[dict[str, Any]]:
        key = f"proctor:metrics:{session_id}"
        client = redis_manager.get_client()
        raw = await client.lrange(key, 0, max(0, int(limit) - 1))
        return [json.loads(item) for item in raw]


class ProctoringMongoStore:
    """
    Persist minimal metrics + alerts to MongoDB (Motor).

    Collections:
    - `proctor_metrics`
    - `proctor_alerts`
    """

    def __init__(self) -> None:
        self._metrics_col = "proctor_metrics"
        self._alerts_col = "proctor_alerts"

    async def append_metrics(self, inputs: ProctoringInputs) -> bool:
        payload: dict[str, Any] = {
            "session_id": inputs.session_id,
            "frame_id": int(inputs.frame_id),
            "timestamp": float(inputs.timestamp),
            "face_detected": bool(inputs.face_detected),
            "face_count": int(inputs.face_count),
            "head_yaw": inputs.head_yaw,
            "gaze_direction": inputs.gaze_direction,
            "blink_rate_per_min": inputs.blink_rate_per_min,
            "hands_visible_count": inputs.hands_visible_count,
            "rapid_hand_movement": inputs.rapid_hand_movement,
            "phone_detected": bool(inputs.phone_detected),
            "unauthorized_materials_detected": bool(inputs.unauthorized_materials_detected),
            "multiple_persons_detected": bool(inputs.multiple_persons_detected),
        }

        try:
            db = mongo_manager.get_db()
            await db[self._metrics_col].insert_one(payload)
            return True
        except Exception as exc:
            logger.warning(
                "[ProctorStoreMongo] Failed to append metrics | session_id=%s collection=%s error=%r",
                inputs.session_id,
                self._metrics_col,
                exc,
            )
            return False

    async def append_alert(self, alert: ProctoringAlert) -> bool:
        try:
            db = mongo_manager.get_db()
            await db[self._alerts_col].insert_one(asdict(alert))
            return True
        except Exception as exc:
            logger.warning(
                "[ProctorStoreMongo] Failed to append alert | session_id=%s collection=%s error=%r",
                alert.session_id,
                self._alerts_col,
                exc,
            )
            return False

    async def get_latest_alerts(self, session_id: str, limit: int = 50) -> list[dict[str, Any]]:
        db = mongo_manager.get_db()
        cursor = (
            db[self._alerts_col]
            .find({"session_id": session_id}, {"_id": 0})
            .sort("last_seen_at", -1)
            .limit(int(limit))
        )
        return [doc async for doc in cursor]

    async def get_latest_metrics(self, session_id: str, limit: int = 200) -> list[dict[str, Any]]:
        db = mongo_manager.get_db()
        cursor = (
            db[self._metrics_col]
            .find({"session_id": session_id}, {"_id": 0})
            .sort("timestamp", -1)
            .limit(int(limit))
        )
        return [doc async for doc in cursor]


class ProctoringDualStore:
    """
    Persist metrics + alerts to BOTH Redis (cache) and MongoDB (persistent DB).
    
    Benefits:
    - Redis for fast retrieval and caching
    - MongoDB for permanent storage and audit trail
    - Automatic failover if one backend fails
    """

    def __init__(self):
        self.redis_store = ProctoringRedisStore()
        self.mongo_store = ProctoringMongoStore()

    async def append_metrics(self, inputs: ProctoringInputs) -> bool:
        """Save to both Redis and MongoDB. Returns True if at least one succeeds."""
        redis_ok = await self.redis_store.append_metrics(inputs)
        mongo_ok = await self.mongo_store.append_metrics(inputs)
        
        if not (redis_ok or mongo_ok):
            logger.error(
                "[ProctorStoreDual] Both backends failed | session_id=%s frame_id=%s",
                inputs.session_id,
                inputs.frame_id,
            )
            return False
        
        if not mongo_ok:
            logger.warning(
                "[ProctorStoreDual] MongoDB write failed, Redis cached | session_id=%s frame_id=%s",
                inputs.session_id,
                inputs.frame_id,
            )
        
        return True

    async def append_alert(self, alert: ProctoringAlert) -> bool:
        """Save to both Redis and MongoDB. Returns True if at least one succeeds."""
        redis_ok = await self.redis_store.append_alert(alert)
        mongo_ok = await self.mongo_store.append_alert(alert)
        
        if not (redis_ok or mongo_ok):
            logger.error(
                "[ProctorStoreDual] Both backends failed | session_id=%s rule_id=%s",
                alert.session_id,
                alert.rule_id,
            )
            return False
        
        if not mongo_ok:
            logger.warning(
                "[ProctorStoreDual] MongoDB write failed, Redis cached | session_id=%s rule_id=%s",
                alert.session_id,
                alert.rule_id,
            )
        
        return True

    async def get_latest_alerts(self, session_id: str, limit: int = 50) -> list[dict[str, Any]]:
        """Prefer MongoDB (persistent), fallback to Redis."""
        try:
            return await self.mongo_store.get_latest_alerts(session_id, limit)
        except Exception:
            logger.warning("[ProctorStoreDual] MongoDB get_latest_alerts failed, using Redis")
            return await self.redis_store.get_latest_alerts(session_id, limit)

    async def get_latest_metrics(self, session_id: str, limit: int = 200) -> list[dict[str, Any]]:
        """Prefer MongoDB (persistent), fallback to Redis."""
        try:
            return await self.mongo_store.get_latest_metrics(session_id, limit)
        except Exception:
            logger.warning("[ProctorStoreDual] MongoDB get_latest_metrics failed, using Redis")
            return await self.redis_store.get_latest_metrics(session_id, limit)


def get_proctoring_store() -> ProctoringStore:
    backend = (getattr(settings, "PROCTOR_STORE_BACKEND", "dual") or "dual").lower().strip()
    if backend == "mongo":
        return ProctoringMongoStore()
    elif backend == "redis":
        return ProctoringRedisStore()
    else:
        # Default to dual for reliable persistence
        return ProctoringDualStore()
