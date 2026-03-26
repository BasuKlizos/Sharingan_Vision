"""
WebRTC Data Channel Manager for sending detection data to frontend
"""
import json
import asyncio
from typing import Optional
from aiortc import RTCDataChannel

from app.logger import logger
from app.modules.signaling.detection_schema import DetectionFrame


class DetectionDataChannelManager:
    """Manages sending detection data through WebRTC data channel"""
    
    def __init__(self, session_id: str, max_queue_size: int = 500):
        self.session_id = session_id
        self.channel: Optional[RTCDataChannel] = None
        self.is_ready = False
        self._max_queue_size = max_queue_size
        self._message_queue: Optional[asyncio.Queue] = None
        self._ready_event: Optional[asyncio.Event] = None
        self._sender_task: Optional[asyncio.Task] = None
        self._buffered_count = 0
        self._initialized = False
    
    def _init_async_objects(self):
        """Initialize async objects in the correct event loop context"""
        if self._initialized:
            return
        try:
            self._ready_event = asyncio.Event()
            self._message_queue = asyncio.Queue(maxsize=self._max_queue_size)
            self._initialized = True
        except RuntimeError as e:
            logger.error(
                f"[DetectionChannel] Failed to initialize async objects | "
                f"session_id={self.session_id} | error={e}"
            )
        
    def set_channel(self, channel: RTCDataChannel):
        """Set the data channel reference"""
        self.channel = channel
        self._init_async_objects()
        
        @channel.on("open")
        def on_open():
            self.is_ready = True
            if self._ready_event:
                self._ready_event.set()
            logger.info(
                f"[DetectionChannel] CHANNEL OPENED | session_id={self.session_id} | "
                f"buffered_messages={self._buffered_count}"
            )
            # Start sender task to flush buffered messages
            if not self._sender_task or self._sender_task.done():
                try:
                    self._sender_task = asyncio.create_task(self._flush_buffered_messages())
                except RuntimeError as e:
                    logger.error(
                        f"[DetectionChannel] Failed to create flush task | "
                        f"session_id={self.session_id} | error={e}"
                    )
        
        @channel.on("error")
        def on_error(error):
            logger.error(
                f"[DetectionChannel] CHANNEL ERROR | session_id={self.session_id} | "
                f"error={error} | channel_state={channel.readyState}"
            )
        
        @channel.on("close")
        def on_close():
            self.is_ready = False
            if self._ready_event:
                self._ready_event.clear()
            logger.warning(
                f"[DetectionChannel] CHANNEL CLOSED | session_id={self.session_id}"
            )
        
        # If channel is already open (unlikely but possible), mark as ready
        if channel.readyState == "open":
            self.is_ready = True
            if self._ready_event:
                self._ready_event.set()
        
        # Start a timeout monitor to detect stuck channels
        asyncio.create_task(self._monitor_channel_state())
    
    async def _monitor_channel_state(self):
        """Monitor channel state for issues like stuck 'connecting' state"""
        try:
            for attempt in range(1, 6):  # Monitor for up to 30 seconds (5 * 5sec intervals)
                await asyncio.sleep(5)
                
                if not self.channel:
                    break
                
                state = self.channel.readyState
                if state != "open" and not self.is_ready:
                    logger.warning(
                        f"[DetectionChannel] WARNING - Stuck state detected | session_id={self.session_id} | "
                        f"attempt={attempt} | readyState={state}"
                    )
                else:
                    # Channel is either open or has been closed
                    break
        except Exception as e:
            pass
    
    async def _flush_buffered_messages(self):
        """Flush buffered messages when channel opens"""
        if not self._message_queue:
            logger.warning(
                f"[DetectionChannel] Message queue not available for flushing | "
                f"session_id={self.session_id}"
            )
            return
        
        timeout = 10  # Max time to flush messages
        start_time = asyncio.get_event_loop().time()
        
        try:
            while not self._message_queue.empty():
                if asyncio.get_event_loop().time() - start_time > timeout:
                    logger.warning(
                        f"[DetectionChannel] Flushing timed out | session_id={self.session_id} | "
                        f"remaining={self._message_queue.qsize()}"
                    )
                    break
                
                try:
                    message = self._message_queue.get_nowait()
                    if self.is_ready and self.channel:
                        self.channel.send(message)
                        self._buffered_count -= 1
                except asyncio.QueueEmpty:
                    break
                except Exception as e:
                    logger.error(
                        f"[DetectionChannel] Error flushing message | "
                        f"session_id={self.session_id} | error={e}"
                    )
                    break
        except Exception as e:
            logger.error(
                f"[DetectionChannel] Error in flush task | session_id={self.session_id} | error={e}"
            )
    
    async def send_detection_frame(self, detection_frame: DetectionFrame):
        """Send detection frame to frontend, with buffering if channel not ready"""
        if not self.channel:
            logger.warning(
                f"[DetectionChannel] Channel not set, cannot send frame | "
                f"session_id={self.session_id}"
            )
            return
        
        # Ensure async objects are initialized
        self._init_async_objects()
        
        try:
            message = detection_frame.to_json()
        except Exception as e:
            logger.error(
                f"[DetectionChannel] Failed to serialize detection frame | "
                f"session_id={self.session_id} | error={e}"
            )
            return
        
        # If channel is ready, send immediately
        if self.is_ready:
            try:
                # Check buffered amount before sending (per spec)
                buffered_amount = getattr(self.channel, 'bufferedAmount', 0)
                if buffered_amount > 65536:  # 64KB threshold
                    logger.warning(
                        f"[DetectionChannel] Send buffer full | "
                        f"session_id={self.session_id} | buffered={buffered_amount}"
                    )
                    # Still try to send but may be dropped
                
                self.channel.send(message)
            except Exception as e:
                logger.error(
                    f"[DetectionChannel] Failed to send detection frame | "
                    f"session_id={self.session_id} | error={e}"
                )
            return
        
        # Channel not ready - buffer the message
        if self._message_queue:
            try:
                self._message_queue.put_nowait(message)
                self._buffered_count += 1
            except asyncio.QueueFull:
                logger.warning(
                    f"[DetectionChannel] Message queue full, dropping frame | "
                    f"session_id={self.session_id}"
                )
        else:
            logger.warning(
                f"[DetectionChannel] Message queue not initialized, cannot buffer frame | "
                f"session_id={self.session_id}"
            )
    
    def send_status_message(self, status: str, message: str = ""):
        """Send status message to frontend"""
        if not self.is_ready or not self.channel:
            return
        
        try:
            msg = json.dumps({
                "type": "status",
                "status": status,
                "message": message,
            })
            self.channel.send(msg)
        except Exception as e:
            logger.error(
                f"[DetectionChannel] Failed to send status message | "
                f"session_id={self.session_id} | error={e}"
            )
    
    async def cleanup(self):
        """Clean up resources"""
        if self._sender_task:
            self._sender_task.cancel()
            try:
                await self._sender_task
            except asyncio.CancelledError:
                pass
        
        if self.channel:
            await self.channel.close()
        
        logger.info(f"[DetectionChannel] Cleaned up | session_id={self.session_id}")
