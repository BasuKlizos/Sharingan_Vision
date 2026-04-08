"""
Proctoring Data Flush Service - Background Task

Follows SOLID Principles:
- Single Responsibility: Only handles flushing logic
- Open/Closed: Extensible for different storage backends
- Dependency Inversion: Injected dependencies (stores, config, logger)
"""

from __future__ import annotations

import asyncio
import json
from typing import Optional, Protocol

from app.core.config import settings
from app.core.redis import redis_manager
from app.logger import logger
from app.modules.proctoring.store import ProctoringStore
from app.modules.proctoring.types import ProctoringAlert, ProctoringInputs


class FlushStrategy(Protocol):
    """Interface for different flush strategies."""

    async def flush_metrics(self, session_id: str, metrics: list[dict]) -> tuple[int, int]:
        """Flush metrics. Returns (flushed_count, failed_count)."""
        ...

    async def flush_alerts(self, session_id: str, alerts: list[dict]) -> tuple[int, int]:
        """Flush alerts. Returns (flushed_count, failed_count)."""
        ...


class MongoFlushStrategy:
    """Flush strategy that persists to MongoDB."""

    def __init__(self, store: ProctoringStore):
        self.store = store

    async def flush_metrics(self, session_id: str, metrics: list[dict]) -> tuple[int, int]:
        """Flush metrics to MongoDB via store."""
        flushed = 0
        failed = 0

        for metric in metrics:
            try:
                inputs = ProctoringInputs(**metric)
                ok = await self.store.append_metrics(inputs)
                if ok:
                    flushed += 1
                else:
                    failed += 1
            except Exception as e:
                logger.error(
                    "[Flush] Failed to parse metric | session_id=%s error=%s",
                    session_id,
                    e,
                )
                failed += 1

        return flushed, failed

    async def flush_alerts(self, session_id: str, alerts: list[dict]) -> tuple[int, int]:
        """Flush alerts to MongoDB via store."""
        flushed = 0
        failed = 0

        for alert_dict in alerts:
            try:
                alert = ProctoringAlert(**alert_dict)
                ok = await self.store.append_alert(alert)
                if ok:
                    flushed += 1
                else:
                    failed += 1
            except Exception as e:
                logger.error(
                    "[Flush] Failed to parse alert | session_id=%s error=%s",
                    session_id,
                    e,
                )
                failed += 1

        return flushed, failed


class ProctoringFlushService:
    """
    Background service to flush proctoring data from Redis to MongoDB.

    Features:
    - Runs periodically based on FLUSH_INTERVAL_SECONDS
    - Batch processes all active sessions
    - Configurable batch sizes and intervals
    - Graceful error handling with logging
    - Singleton pattern (manages its own lifecycle)
    """

    def __init__(
        self,
        store: ProctoringStore,
        flush_interval: int = 30,
        batch_size_metrics: int = 100,
        batch_size_alerts: int = 50,
    ):
        self.store = store
        self.flush_interval = int(flush_interval)
        self.batch_size_metrics = int(batch_size_metrics)
        self.batch_size_alerts = int(batch_size_alerts)
        self.flush_strategy: FlushStrategy = MongoFlushStrategy(store)

        self._task: Optional[asyncio.Task] = None
        self._running = False

    async def start(self) -> None:
        """Start the background flush service."""
        if self._running:
            logger.warning("[ProctorFlush] Already running")
            return

        self._running = True
        self._task = asyncio.create_task(self._run_loop())
        logger.info(
            "[ProctorFlush] Service started | interval=%ds metrics_batch=%d alerts_batch=%d",
            self.flush_interval,
            self.batch_size_metrics,
            self.batch_size_alerts,
        )

    async def stop(self) -> None:
        """Stop the background flush service gracefully."""
        if not self._running:
            return

        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

        logger.info("[ProctorFlush] Service stopped")

    async def _run_loop(self) -> None:
        """Main event loop for periodic flushing."""
        try:
            while self._running:
                try:
                    await self._flush_all_sessions()
                except Exception as e:
                    logger.error("[ProctorFlush] Loop error | error=%s", e)

                # Sleep before next flush
                await asyncio.sleep(self.flush_interval)

        except asyncio.CancelledError:
            logger.debug("[ProctorFlush] Loop cancelled")
        except Exception as e:
            logger.error("[ProctorFlush] Unexpected error | error=%s", e)
            self._running = False

    async def _flush_all_sessions(self) -> None:
        """Discover and flush all active sessions from Redis."""
        try:
            client = redis_manager.get_client()

            # Find all active metrics keys
            metrics_keys = await client.keys("proctor:metrics:*")
            alerts_keys = await client.keys("proctor:alerts:*")

            if not metrics_keys and not alerts_keys:
                return

            total_metrics_flushed = 0
            total_metrics_failed = 0
            total_alerts_flushed = 0
            total_alerts_failed = 0

            # Process metrics
            for key in metrics_keys or []:
                try:
                    session_id = key.decode() if isinstance(key, bytes) else key
                    session_id = session_id.replace("proctor:metrics:", "")
                    
                    flushed, failed = await self._flush_metrics_for_session(session_id)
                    total_metrics_flushed += flushed
                    total_metrics_failed += failed
                except Exception as e:
                    logger.error("[ProctorFlush] Failed to flush metrics | key=%s error=%s", key, e)

            # Process alerts
            for key in alerts_keys or []:
                try:
                    session_id = key.decode() if isinstance(key, bytes) else key
                    session_id = session_id.replace("proctor:alerts:", "")
                    
                    flushed, failed = await self._flush_alerts_for_session(session_id)
                    total_alerts_flushed += flushed
                    total_alerts_failed += failed
                except Exception as e:
                    logger.error("[ProctorFlush] Failed to flush alerts | key=%s error=%s", key, e)

            if total_metrics_flushed > 0 or total_alerts_flushed > 0:
                logger.info(
                    "[ProctorFlush] Cycle complete | metrics_flushed=%d metrics_failed=%d "
                    "alerts_flushed=%d alerts_failed=%d",
                    total_metrics_flushed,
                    total_metrics_failed,
                    total_alerts_flushed,
                    total_alerts_failed,
                )

        except Exception as e:
            logger.error("[ProctorFlush] Failed to discover sessions | error=%s", e)

    async def _flush_metrics_for_session(self, session_id: str) -> tuple[int, int]:
        """Flush metrics for a specific session in batches."""
        client = redis_manager.get_client()
        key = f"proctor:metrics:{session_id}"
        total_flushed = 0
        total_failed = 0

        while True:
            try:
                # Fetch batch from Redis (from tail, FIFO order preservation)
                raw_metrics = await client.lrange(
                    key, -self.batch_size_metrics, -1
                )

                if not raw_metrics:
                    break

                metrics = [json.loads(item) for item in raw_metrics]

                # Flush to MongoDB via strategy
                flushed, failed = await self.flush_strategy.flush_metrics(session_id, metrics)
                total_flushed += flushed
                total_failed += failed

                # Remove flushed items from Redis
                if flushed > 0:
                    await client.ltrim(key, 0, -(flushed + 1))

                # Break if we got less than batch size (reached end)
                if len(raw_metrics) < self.batch_size_metrics:
                    break

            except Exception as e:
                logger.error(
                    "[ProctorFlush] Metrics batch flush failed | session_id=%s error=%s",
                    session_id,
                    e,
                )
                break

        return total_flushed, total_failed

    async def _flush_alerts_for_session(self, session_id: str) -> tuple[int, int]:
        """Flush alerts for a specific session in batches."""
        client = redis_manager.get_client()
        key = f"proctor:alerts:{session_id}"
        total_flushed = 0
        total_failed = 0

        while True:
            try:
                # Fetch batch from Redis (from tail, FIFO order preservation)
                raw_alerts = await client.lrange(
                    key, -self.batch_size_alerts, -1
                )

                if not raw_alerts:
                    break

                alerts = [json.loads(item) for item in raw_alerts]

                # Flush to MongoDB via strategy
                flushed, failed = await self.flush_strategy.flush_alerts(session_id, alerts)
                total_flushed += flushed
                total_failed += failed

                # Remove flushed items from Redis
                if flushed > 0:
                    await client.ltrim(key, 0, -(flushed + 1))

                # Break if we got less than batch size (reached end)
                if len(raw_alerts) < self.batch_size_alerts:
                    break

            except Exception as e:
                logger.error(
                    "[ProctorFlush] Alerts batch flush failed | session_id=%s error=%s",
                    session_id,
                    e,
                )
                break

        return total_flushed, total_failed

    async def flush_session_on_demand(self, session_id: str) -> None:
        """Flush a specific session immediately (called on session end)."""
        try:
            flushed_metrics, failed_metrics = await self._flush_metrics_for_session(session_id)
            flushed_alerts, failed_alerts = await self._flush_alerts_for_session(session_id)

            logger.info(
                "[ProctorFlush] On-demand flush | session_id=%s metrics_flushed=%d "
                "metrics_failed=%d alerts_flushed=%d alerts_failed=%d",
                session_id,
                flushed_metrics,
                failed_metrics,
                flushed_alerts,
                failed_alerts,
            )
        except Exception as e:
            logger.error("[ProctorFlush] On-demand flush failed | session_id=%s error=%s", session_id, e)


# Singleton instance
_flush_service_instance: Optional[ProctoringFlushService] = None


def get_flush_service() -> ProctoringFlushService:
    """Get or create the flush service singleton."""
    global _flush_service_instance
    if _flush_service_instance is None:
        from app.modules.proctoring.store import get_proctoring_store

        store = get_proctoring_store()
        flush_interval = int(getattr(settings, "PROCTOR_FLUSH_INTERVAL_SECONDS", 30))
        _flush_service_instance = ProctoringFlushService(
            store=store,
            flush_interval=flush_interval,
        )
    return _flush_service_instance
