from __future__ import annotations

import json
from datetime import datetime, timezone
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
    async def append_metrics_batch(self, items: list[ProctoringInputs]) -> bool: ...
    async def append_alerts_batch(self, items: list[ProctoringAlert]) -> bool: ...


def _iso_datetime(value: float | int | None) -> str | None:
    if value is None:
        return None
    return datetime.fromtimestamp(float(value), tz=timezone.utc).isoformat().replace("+00:00", "Z")


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class ProctoringRedisStore:
    """
    Fast-access cache store.
    """

    @staticmethod
    def _alerts_key(session_id: str) -> str:
        return f"proctor:alerts:{session_id}:alerts"

    @staticmethod
    def _alert_meta_key(session_id: str) -> str:
        return f"proctor:alerts:{session_id}:meta"

    @staticmethod
    def _metrics_summary_key(session_id: str) -> str:
        return f"proctor:metrics:{session_id}:summary"

    async def ensure_ready(self) -> None:
        return None

    async def append_metrics_batch(self, items: list[ProctoringInputs]) -> bool:
        if not items:
            return True
        client = redis_manager.get_client()
        try:
            async with client.pipeline(transaction=False) as pipe:
                for item in items:
                    summary_key = self._metrics_summary_key(item.session_id)
                    pipe.hincrby(summary_key, "summary.total_samples", 1)
                    pipe.hincrby(
                        summary_key, "summary.suspicious_samples", int(bool(item.suspicious))
                    )
                    pipe.hincrby(
                        summary_key, "summary.face_detected_samples", int(bool(item.face_detected))
                    )
                    pipe.hincrby(
                        summary_key, "summary.no_face_samples", int(not bool(item.face_detected))
                    )
                    pipe.hincrby(
                        summary_key,
                        "summary.phone_detected_samples",
                        int(bool(item.phone_detected)),
                    )
                    pipe.hincrby(
                        summary_key,
                        "summary.other_device_detected_samples",
                        int(bool(item.other_device_detected)),
                    )
                    pipe.hincrby(
                        summary_key,
                        "summary.unauthorized_materials_detected_samples",
                        int(bool(item.unauthorized_materials_detected)),
                    )
                    pipe.hincrby(
                        summary_key,
                        "summary.multiple_persons_detected_samples",
                        int(bool(item.multiple_persons_detected)),
                    )
                    pipe.hincrby(
                        summary_key,
                        "summary.rapid_hand_movement_samples",
                        int(bool(item.rapid_hand_movement)),
                    )
                    pipe.hincrbyfloat(
                        summary_key, "summary.cumulative_risk_score", float(item.risk_score)
                    )
                    pipe.hincrby(summary_key, "summary.cumulative_face_count", int(item.face_count))
                    pipe.hset(
                        summary_key,
                        mapping={
                            "session_id": item.session_id,
                            "updated_at": _iso_datetime(item.timestamp) or "",
                            "latest.timestamp": item.timestamp,
                            "latest.frame_id": item.frame_id,
                            "latest.face_count": item.face_count,
                            "latest.face_detected": int(bool(item.face_detected)),
                            "latest.gaze_direction": item.gaze_direction or "",
                        },
                    )
                    pipe.hsetnx(summary_key, "created_at", _iso_datetime(item.timestamp) or "")
                    pipe.hsetnx(summary_key, "summary.max_risk_score", float(item.risk_score))
                    pipe.hsetnx(summary_key, "summary.max_face_count", int(item.face_count))
                await pipe.execute()

            for item in items:
                summary_key = self._metrics_summary_key(item.session_id)
                max_risk = await client.hget(summary_key, "summary.max_risk_score")
                max_face = await client.hget(summary_key, "summary.max_face_count")
                updates: dict[str, Any] = {}
                if max_risk is None or float(item.risk_score) > float(max_risk):
                    updates["summary.max_risk_score"] = float(item.risk_score)
                if max_face is None or int(item.face_count) > int(max_face):
                    updates["summary.max_face_count"] = int(item.face_count)
                if updates:
                    await client.hset(summary_key, mapping=updates)
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
                    event = {
                        "timestamp": item.last_seen_at,
                        "frame_id": item.frame_id,
                        "rule_id": item.rule_id,
                        "label": item.label,
                        "severity": item.severity,
                        "risk_score": item.risk_score,
                        "started_at": item.started_at,
                        "last_seen_at": item.last_seen_at,
                        "occurrence_count": int(item.occurrence_count),
                        "evidence": asdict(item).get("evidence", {}),
                    }
                    alerts_key = self._alerts_key(item.session_id)
                    meta_key = self._alert_meta_key(item.session_id)
                    pipe.lpush(alerts_key, json.dumps(event, default=str))
                    pipe.ltrim(alerts_key, 0, cache_limit - 1)
                    pipe.hincrby(meta_key, "total_alert_count", int(item.occurrence_count))
                    pipe.hincrby(
                        meta_key, f"alert_type_count:{item.rule_id}", int(item.occurrence_count)
                    )
                    pipe.hset(
                        meta_key,
                        mapping={
                            "session_id": item.session_id,
                            "updated_at": _iso_datetime(item.last_seen_at) or "",
                            "last_alert_at": _iso_datetime(item.last_seen_at) or "",
                            "last_frame_id": item.frame_id,
                            "latest_alert_rule_id": item.rule_id,
                            "latest_alert_label": item.label,
                        },
                    )
                    pipe.hsetnx(meta_key, "created_at", _iso_datetime(item.started_at) or "")
                await pipe.execute()
            return True
        except Exception as exc:
            logger.warning("[ProctorStoreRedis] Failed to append alerts | error=%r", exc)
            return False


class ProctoringMongoStore:
    """
    Persistent audit store in MongoDB.

    Collections:
    - `proctor_metrics` (one document per session with summary and alert history)
    """

    _SUMMARY_DOC_KIND = "session_summary"

    @staticmethod
    def _legacy_metrics_flat_fields() -> dict[str, str]:
        return {
            "cumulative_face_count": "",
            "cumulative_risk_score": "",
            "events": "",
            "face_detected_samples": "",
            "last_face_count": "",
            "last_face_detected": "",
            "last_frame_id": "",
            "last_gaze_direction": "",
            "last_timestamp": "",
            "max_face_count": "",
            "max_risk_score": "",
            "multiple_persons_detected_samples": "",
            "no_face_samples": "",
            "other_device_detected_samples": "",
            "phone_detected_samples": "",
            "rapid_hand_movement_samples": "",
            "suspicious_samples": "",
            "total_samples": "",
            "unauthorized_materials_detected_samples": "",
        }

    def __init__(self) -> None:
        self._metrics_col = settings.PROCTOR_METRICS_COLLECTION

    async def ensure_ready(self) -> None:
        db = mongo_manager.get_db()
        await db[self._metrics_col].create_index(
            [("session_id", 1), ("doc_kind", 1)],
            unique=True,
            name="uniq_proctor_metrics_session_summary",
        )

    async def append_metrics_batch(self, items: list[ProctoringInputs]) -> bool:
        if not items:
            return True
        try:
            db = mongo_manager.get_db()
            grouped: dict[str, list[ProctoringInputs]] = {}
            for item in items:
                grouped.setdefault(item.session_id, []).append(item)

            operations = []
            for session_id, session_items in grouped.items():
                write_timestamp = _utc_now()
                sorted_items = sorted(
                    session_items, key=lambda metric: (metric.timestamp, metric.frame_id)
                )
                last_item = sorted_items[-1]
                total_samples = len(sorted_items)
                suspicious_samples = sum(int(bool(item.suspicious)) for item in sorted_items)
                face_detected_samples = sum(int(bool(item.face_detected)) for item in sorted_items)
                no_face_samples = total_samples - face_detected_samples
                phone_detected_samples = sum(
                    int(bool(item.phone_detected)) for item in sorted_items
                )
                other_device_detected_samples = sum(
                    int(bool(item.other_device_detected)) for item in sorted_items
                )
                unauthorized_materials_detected_samples = sum(
                    int(bool(item.unauthorized_materials_detected)) for item in sorted_items
                )
                multiple_persons_detected_samples = sum(
                    int(bool(item.multiple_persons_detected)) for item in sorted_items
                )
                rapid_hand_movement_samples = sum(
                    int(bool(item.rapid_hand_movement)) for item in sorted_items
                )
                cumulative_risk_score = sum(float(item.risk_score) for item in sorted_items)
                cumulative_face_count = sum(int(item.face_count) for item in sorted_items)
                max_risk_score = max(float(item.risk_score) for item in sorted_items)
                max_face_count = max(int(item.face_count) for item in sorted_items)

                operations.append(
                    UpdateOne(
                        {"session_id": session_id, "doc_kind": self._SUMMARY_DOC_KIND},
                        {
                            "$setOnInsert": {
                                "session_id": session_id,
                                "doc_kind": self._SUMMARY_DOC_KIND,
                                "created_at": write_timestamp,
                                "alerts": [],
                                "alert_summary.total_count": 0,
                                "alert_summary.by_type": {},
                            },
                            "$set": {
                                "updated_at": write_timestamp,
                                "latest.timestamp": last_item.timestamp,
                                "latest.frame_id": last_item.frame_id,
                                "latest.face_count": last_item.face_count,
                                "latest.face_detected": bool(last_item.face_detected),
                                "latest.gaze_direction": last_item.gaze_direction,
                            },
                            "$inc": {
                                "summary.total_samples": total_samples,
                                "summary.suspicious_samples": suspicious_samples,
                                "summary.face_detected_samples": face_detected_samples,
                                "summary.no_face_samples": no_face_samples,
                                "summary.phone_detected_samples": phone_detected_samples,
                                "summary.other_device_detected_samples": other_device_detected_samples,
                                "summary.unauthorized_materials_detected_samples": unauthorized_materials_detected_samples,
                                "summary.multiple_persons_detected_samples": multiple_persons_detected_samples,
                                "summary.rapid_hand_movement_samples": rapid_hand_movement_samples,
                                "summary.cumulative_risk_score": cumulative_risk_score,
                                "summary.cumulative_face_count": cumulative_face_count,
                            },
                            "$max": {
                                "summary.max_risk_score": max_risk_score,
                                "summary.max_face_count": max_face_count,
                            },
                            "$unset": self._legacy_metrics_flat_fields(),
                        },
                        upsert=True,
                    )
                )
            result = await db[self._metrics_col].bulk_write(operations, ordered=False)
            return bool(result.upserted_count or result.modified_count or result.matched_count)
        except Exception as exc:
            logger.error("[ProctorStoreMongo] Failed to append metrics | error=%r", exc)
            return False

    async def append_alerts_batch(self, items: list[ProctoringAlert]) -> bool:
        if not items:
            return True
        try:
            db = mongo_manager.get_db()
            history_limit = int(settings.PROCTOR_REDIS_MAX_ITEMS)
            grouped: dict[str, list[ProctoringAlert]] = {}
            for item in items:
                grouped.setdefault(item.session_id, []).append(item)

            operations = []
            for session_id, session_items in grouped.items():
                write_timestamp = _utc_now()
                sorted_items = sorted(
                    session_items, key=lambda alert: (alert.last_seen_at, alert.frame_id)
                )
                first_item = sorted_items[0]
                last_item = sorted_items[-1]
                alerts = []
                inc_counts: dict[str, int] = {
                    "total_alert_count": 0,
                    "alert_summary.total_count": 0,
                }

                for item in sorted_items:
                    alerts.append(
                        {
                            "timestamp": item.last_seen_at,
                            "frame_id": item.frame_id,
                            "rule_id": item.rule_id,
                            "label": item.label,
                            "severity": item.severity,
                            "risk_score": item.risk_score,
                            "started_at": item.started_at,
                            "last_seen_at": item.last_seen_at,
                            "occurrence_count": int(item.occurrence_count),
                            "evidence": asdict(item).get("evidence", {}),
                        }
                    )
                    inc_counts["total_alert_count"] += int(item.occurrence_count)
                    inc_counts[f"alert_type_counts.{item.rule_id}"] = inc_counts.get(
                        f"alert_type_counts.{item.rule_id}", 0
                    ) + int(item.occurrence_count)
                    inc_counts[f"alert_summary.by_type.{item.rule_id}"] = inc_counts.get(
                        f"alert_summary.by_type.{item.rule_id}", 0
                    ) + int(item.occurrence_count)

                operations.append(
                    UpdateOne(
                        {"session_id": session_id, "doc_kind": self._SUMMARY_DOC_KIND},
                        {
                            "$setOnInsert": {
                                "session_id": session_id,
                                "doc_kind": self._SUMMARY_DOC_KIND,
                                "created_at": write_timestamp,
                                "latest.timestamp": None,
                                "latest.frame_id": None,
                                "latest.face_count": 0,
                                "latest.face_detected": False,
                                "latest.gaze_direction": None,
                                "summary.total_samples": 0,
                                "summary.suspicious_samples": 0,
                                "summary.face_detected_samples": 0,
                                "summary.no_face_samples": 0,
                                "summary.phone_detected_samples": 0,
                                "summary.other_device_detected_samples": 0,
                                "summary.unauthorized_materials_detected_samples": 0,
                                "summary.multiple_persons_detected_samples": 0,
                                "summary.rapid_hand_movement_samples": 0,
                                "summary.cumulative_risk_score": 0.0,
                                "summary.cumulative_face_count": 0,
                                "summary.max_risk_score": 0.0,
                                "summary.max_face_count": 0,
                                "metadata.session_id": session_id,
                                "metadata.first_alert_at": _iso_datetime(first_item.started_at),
                            },
                            "$set": {
                                "updated_at": write_timestamp,
                                "metadata.updated_at": _iso_datetime(last_item.last_seen_at),
                                "metadata.last_alert_at": _iso_datetime(last_item.last_seen_at),
                                "metadata.last_frame_id": last_item.frame_id,
                                "metadata.latest_alert_rule_id": last_item.rule_id,
                                "metadata.latest_alert_label": last_item.label,
                            },
                            "$inc": inc_counts,
                            "$push": {
                                "alerts": {"$each": alerts, "$slice": -history_limit},
                            },
                            "$unset": self._legacy_metrics_flat_fields(),
                        },
                        upsert=True,
                    )
                )
            result = await db[self._metrics_col].bulk_write(operations, ordered=False)
            return bool(result.upserted_count or result.modified_count or result.matched_count)
        except Exception as exc:
            logger.error("[ProctorStoreMongo] Failed to append alerts | error=%r", exc)
            return False


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


@lru_cache(maxsize=1)
def get_proctoring_store() -> ProctoringStore:
    backend = settings.PROCTOR_STORE_BACKEND.strip().lower()
    if backend == "mongo":
        return ProctoringMongoStore()
    if backend == "redis":
        return ProctoringRedisStore()
    return ProctoringDualStore()
