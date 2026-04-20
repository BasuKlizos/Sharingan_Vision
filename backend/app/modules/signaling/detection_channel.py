"""
WebRTC Data Channel Manager for sending detection data to frontend
"""

import json
import asyncio
from typing import Optional
from aiortc import RTCDataChannel

from app.logger import logger
from app.modules.monitoring.schemas import CurrentViewData
from app.modules.monitoring.store import session_monitoring_store


class DetectionDataChannelManager:
    """Manages sending detection data through WebRTC data channel"""

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.channel: Optional[RTCDataChannel] = None
        self.is_ready = False

    def set_channel(self, channel: RTCDataChannel):
        """Set the data channel reference"""
        self.channel = channel

        @channel.on("open")
        def on_open():
            self.is_ready = True
            logger.info(f"[DetectionChannel] CHANNEL OPENED | session_id={self.session_id}")

        @channel.on("error")
        def on_error(error):
            logger.error(
                f"[DetectionChannel] CHANNEL ERROR | session_id={self.session_id} | "
                f"error={error} | channel_state={channel.readyState}"
            )

        @channel.on("close")
        def on_close():
            self.is_ready = False
            logger.warning(f"[DetectionChannel] CHANNEL CLOSED | session_id={self.session_id}")

        @channel.on("message")
        def on_message(message):
            self._handle_incoming_message(message)

        # If channel is already open (unlikely but possible), mark as ready
        if channel.readyState == "open":
            self.is_ready = True

    def _handle_incoming_message(self, message):
        if not isinstance(message, str):
            logger.debug(
                f"[DetectionChannel] Ignored non-text message | session_id={self.session_id}"
            )
            return

        try:
            payload = json.loads(message)
        except json.JSONDecodeError:
            logger.warning(
                f"[DetectionChannel] Failed to parse incoming message | session_id={self.session_id}"
            )
            return

        message_type = payload.get("type")
        if message_type != "current_view":
            logger.debug(
                f"[DetectionChannel] Ignored incoming message type | session_id={self.session_id} "
                f"type={message_type}"
            )
            return

        current_view_payload = payload.get("currentView")
        if not isinstance(current_view_payload, dict):
            logger.warning(
                f"[DetectionChannel] Missing currentView payload | session_id={self.session_id}"
            )
            return

        try:
            current_view = CurrentViewData.model_validate(current_view_payload)
        except Exception as exc:
            logger.warning(
                f"[DetectionChannel] Invalid current view payload | session_id={self.session_id} error={exc}"
            )
            return

        updated_session = session_monitoring_store.update_current_view_for_session(
            session_id=self.session_id,
            current_view=current_view,
        )
        logger.info(
            f"[DetectionChannel] Current view received via WebRTC | session_id={self.session_id} "
            f"received_at={asyncio.get_event_loop().time():.3f} "
            f"min_x={current_view.min_x:.2f} max_x={current_view.max_x:.2f} "
            f"min_y={current_view.min_y:.2f} max_y={current_view.max_y:.2f}"
        )
        if updated_session and updated_session.latest_zone_assessment:
            zone_assessment = updated_session.latest_zone_assessment
            logger.info(
                f"[DetectionChannel] Zone assessment | session_id={self.session_id} "
                f"status={zone_assessment['status']} "
                f"drift_percent={zone_assessment['driftPercent']:.2f}"
            )

    async def cleanup(self):
        """Clean up resources"""
        if self.channel:
            close_result = self.channel.close()
            if asyncio.iscoroutine(close_result):
                await close_result

        logger.info(f"[DetectionChannel] Cleaned up | session_id={self.session_id}")
