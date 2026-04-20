"""
Proctoring Data Flush Service - Background Task
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Optional, Protocol

from app.core.config import settings
from app.logger import logger
from app.modules.proctoring.store import ProctoringStore, get_proctoring_store
from app.modules.proctoring.types import ProctoringAlert, ProctoringInputs


class FlushStrategy(Protocol):
    async def ensure_ready(self) -> None: ...
    async def flush_metrics(self, items: list[ProctoringInputs]) -> bool: ...
    async def flush_alerts(self, items: list[ProctoringAlert]) -> bool: ...


class StoreFlushStrategy:
    def __init__(self, store: ProctoringStore) -> None:
        self._store = store

    async def ensure_ready(self) -> None:
        await self._store.ensure_ready()

    async def flush_metrics(self, items: list[ProctoringInputs]) -> bool:
        return await self._store.append_metrics_batch(items)

    async def flush_alerts(self, items: list[ProctoringAlert]) -> bool:
        return await self._store.append_alerts_batch(items)


@dataclass(slots=True)
class _FlushRuntime:
    metrics_queue: asyncio.Queue[ProctoringInputs]
    alerts_queue: asyncio.Queue[ProctoringAlert]
    metrics_task: asyncio.Task[None]
    alerts_task: asyncio.Task[None]


@dataclass(slots=True)
class _MetricSamplingState:
    last_sampled_at: float = 0.0
    last_suspicious_at: float = 0.0


class ProctoringFlushService:
    """
    Buffered background flusher for proctoring metrics and alerts.
    """

    def __init__(self, strategy: FlushStrategy | None = None) -> None:
        self._strategy = strategy or StoreFlushStrategy(get_proctoring_store())
        self._runtime: Optional[_FlushRuntime] = None
        self._started = False
        self._metric_sampling: dict[str, _MetricSamplingState] = {}

    async def start(self) -> None:
        if self._started:
            return

        await self._strategy.ensure_ready()
        queue_size = int(settings.PROCTOR_FLUSH_QUEUE_SIZE)
        metrics_queue: asyncio.Queue[ProctoringInputs] = asyncio.Queue(maxsize=queue_size)
        alerts_queue: asyncio.Queue[ProctoringAlert] = asyncio.Queue(maxsize=queue_size)
        metrics_task = asyncio.create_task(self._run_metrics_loop(metrics_queue))
        alerts_task = asyncio.create_task(self._run_alerts_loop(alerts_queue))
        self._runtime = _FlushRuntime(metrics_queue, alerts_queue, metrics_task, alerts_task)
        self._started = True
        logger.info("[ProctorFlush] Started background flush service")

    async def stop(self) -> None:
        if not self._started or self._runtime is None:
            return

        await self._drain_and_flush(self._runtime.metrics_queue, item_type="metrics")
        await self._drain_and_flush(self._runtime.alerts_queue, item_type="alerts")

        self._runtime.metrics_task.cancel()
        self._runtime.alerts_task.cancel()
        await asyncio.gather(
            self._runtime.metrics_task,
            self._runtime.alerts_task,
            return_exceptions=True,
        )
        self._runtime = None
        self._started = False
        self._metric_sampling.clear()
        logger.info("[ProctorFlush] Stopped background flush service")

    def submit_metric(self, item: ProctoringInputs) -> None:
        if self._runtime is None:
            logger.debug("[ProctorFlush] Metric dropped because service is not started")
            return
        if not self._should_store_metric(item):
            return
        self._enqueue(self._runtime.metrics_queue, item, "metrics")

    def submit_alert(self, item: ProctoringAlert) -> None:
        if self._runtime is None:
            logger.debug("[ProctorFlush] Alert dropped because service is not started")
            return
        self._enqueue(self._runtime.alerts_queue, item, "alerts")

    def submit_frame(self, inputs: ProctoringInputs, alerts: list[ProctoringAlert]) -> None:
        self.submit_metric(inputs)
        for alert in alerts:
            self.submit_alert(alert)

    def _should_store_metric(self, item: ProctoringInputs) -> bool:
        state = self._metric_sampling.setdefault(item.session_id, _MetricSamplingState())
        suspicious = bool(item.suspicious) or float(item.risk_score) >= float(
            settings.PROCTOR_HIGH_RISK_THRESHOLD
        )
        if suspicious:
            min_interval = float(settings.PROCTOR_SUSPICIOUS_SAMPLE_SECONDS)
            if (item.timestamp - state.last_suspicious_at) < min_interval:
                return False
            state.last_suspicious_at = item.timestamp
            state.last_sampled_at = item.timestamp
            return True

        min_interval = float(settings.PROCTOR_METRIC_SAMPLE_SECONDS)
        if (item.timestamp - state.last_sampled_at) < min_interval:
            return False
        state.last_sampled_at = item.timestamp
        return True

    def _enqueue(self, queue: asyncio.Queue, item: object, item_type: str) -> None:
        try:
            queue.put_nowait(item)
        except asyncio.QueueFull:
            logger.warning("[ProctorFlush] %s queue full, item dropped", item_type)

    async def _run_metrics_loop(self, queue: asyncio.Queue[ProctoringInputs]) -> None:
        await self._run_loop(queue, item_type="metrics")

    async def _run_alerts_loop(self, queue: asyncio.Queue[ProctoringAlert]) -> None:
        await self._run_loop(queue, item_type="alerts")

    async def _run_loop(self, queue: asyncio.Queue, item_type: str) -> None:
        batch_size = int(settings.PROCTOR_FLUSH_BATCH_SIZE)
        interval_s = float(settings.PROCTOR_FLUSH_INTERVAL_SECONDS)
        try:
            while True:
                item = await queue.get()
                items = [item]
                started = asyncio.get_running_loop().time()

                while len(items) < batch_size:
                    remaining = interval_s - (asyncio.get_running_loop().time() - started)
                    if remaining <= 0:
                        break
                    try:
                        items.append(await asyncio.wait_for(queue.get(), timeout=remaining))
                    except asyncio.TimeoutError:
                        break

                if item_type == "metrics":
                    await self._strategy.flush_metrics(items)
                else:
                    await self._strategy.flush_alerts(items)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.error("[ProctorFlush] %s loop crashed | error=%r", item_type, exc)

    async def _drain_and_flush(self, queue: asyncio.Queue, item_type: str) -> None:
        if queue.empty():
            return
        items: list = []
        while not queue.empty():
            try:
                items.append(queue.get_nowait())
            except asyncio.QueueEmpty:
                break
        if not items:
            return
        if item_type == "metrics":
            await self._strategy.flush_metrics(items)
        else:
            await self._strategy.flush_alerts(items)


_flush_service_instance: Optional[ProctoringFlushService] = None


def get_flush_service() -> ProctoringFlushService:
    global _flush_service_instance
    if _flush_service_instance is None:
        _flush_service_instance = ProctoringFlushService()
    return _flush_service_instance
