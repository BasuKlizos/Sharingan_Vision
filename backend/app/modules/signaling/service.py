import asyncio
import numpy as np
import time
from typing import Dict
from fastapi import WebSocket

from aiortc import RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaBlackhole

from app.modules.signaling.interfaces import BaseConnectionManager
from app.modules.signaling.detection_channel import DetectionDataChannelManager
from app.modules.signaling.detection_schema import DetectionFrame
from app.modules.detection.yolo_detector import YoloDetector
from app.api.utils.utils import generate_session_id
from app.common.exceptions import InvalidMessageError
from app.core.config import settings
from app.logger import logger

PEER_CONNECTIONS: Dict[str, RTCPeerConnection] = {}
DETECTION_CHANNELS: Dict[str, DetectionDataChannelManager] = {}

class SignalingService:

    def __init__(self, manager: BaseConnectionManager):
        self.manager = manager

    async def handle_message(self, room_id: str, sender: WebSocket, message: Dict):

        role = message.get("role")
        msg_type = message.get("type")

        if msg_type not in [
            "join",
            "ready",
            "offer",
            "answer",
            "ice-candidate",
            "leave",
        ]:
            raise InvalidMessageError(f"Invalid message type: {msg_type}")

        if msg_type == "offer" and role != "candidate":
            raise InvalidMessageError("Only candidate can send offer")

        if msg_type == "answer" and role != "interviewer":
            raise InvalidMessageError("Only interviewer can send answer")
        
        if msg_type == "join":
            return

        await self.manager.relay(room_id, sender, message)

class WebRTCService:

    async def _process_video_track(self, track, session_id: str):
        
        logger.info(f"[WebRTC] Start video processing | session_id={session_id}")

        last_inference_ts = 0.0
        frame_id = 0
        detector = YoloDetector(
            model_path=getattr(settings, "YOLO_MODEL_PATH", "yolov8n.pt"),
            conf_threshold=float(getattr(settings, "YOLO_CONF_THRESHOLD", 0.25)),
        )

        while True:
            try:
                frame = await track.recv()

                # logger.debug(
                #     f"[VideoFrame] pts={frame.pts}, size={frame.width}x{frame.height}"
                # )

                if not getattr(settings, "ENABLE_YOLO", False):
                    continue

                fps = int(getattr(settings, "DETECTION_FPS", 5))
                if fps > 0:
                    now = time.monotonic()
                    if now - last_inference_ts < (1.0 / fps):
                        continue
                    last_inference_ts = now

                try:
                    import cv2 
                except Exception as e:
                    logger.error(
                        f"[WebRTC] OpenCV not installed; cannot run YOLO | session_id={session_id} | error={e}"
                    )
                    await asyncio.sleep(1)
                    continue

                img = frame.to_ndarray(format="bgr24")

                # Optional: downscale for speed while keeping aspect ratio
                max_w = int(getattr(settings, "YOLO_MAX_WIDTH", 640))
                if max_w > 0 and img.shape[1] > max_w:
                    scale = max_w / float(img.shape[1])
                    img = cv2.resize(
                        img,
                        (max_w, int(img.shape[0] * scale)),
                        interpolation=cv2.INTER_AREA,
                    )

                detections = await asyncio.to_thread(detector.detect, img)
                if detections:
                    summary = ", ".join(
                        f"{d.class_name}:{d.confidence:.2f}" for d in detections[:5]
                    )
                    logger.info(
                        f"[YOLO] detections={len(detections)} | session_id={session_id} | {summary}"
                    )
                    
                    # Send detections to frontend via data channel
                    detection_frame = DetectionFrame.from_yolo_detections(
                        frame_id=frame_id,
                        timestamp=time.time(),
                        yolo_detections=detections,
                    )
                    
                    channel_manager = DETECTION_CHANNELS.get(session_id)
                    if channel_manager:
                        logger.debug(f"[WebRTC] Sending detection frame | session_id={session_id} | frame_id={frame_id}")
                        await channel_manager.send_detection_frame(detection_frame)
                    else:
                        logger.warning(f"[WebRTC] No channel manager for session_id={session_id}, cannot send detections")
                        
                frame_id += 1

            except Exception as e:
                logger.error(f"[WebRTC] Video frame error: {e}")
                break

    async def _process_audio_track(self, track, session_id: str):
        logger.info(f"[WebRTC] Start audio processing | session_id={session_id}")

        while True:
            try:
                frame = await track.recv()

                # Convert to numpy
                audio = frame.to_ndarray()

                logger.debug(
                    f"[AudioFrame] pts={frame.pts}, "
                    f"samples={frame.samples}, "
                    f"rate={frame.sample_rate}, "
                    f"channels={audio.shape[0]} "
                    f"audio {audio}"
                )

                # simple speech detection
                volume = np.abs(audio).mean()

                if volume > 1000:  # tune threshold
                    logger.debug(f"[Audio] Speaking detected | session_id={session_id}")

                # TODO: send to ASR / analysis pipeline

            except Exception as e:
                logger.error(f"[WebRTC] Audio frame error: {e}")
                break

    def _setup_track_handlers(self, pc: RTCPeerConnection, session_id: str):
        @pc.on("track")
        def on_track(track):
            logger.info(
                f"[WebRTC] Track received | kind={track.kind} | session_id={session_id}"
            )

            if track.kind == "video":
                task = asyncio.create_task(self._process_video_track(track, session_id))
                track.task = task
            # elif track.kind == "audio":
            #     task = asyncio.create_task(self._process_audio_track(track, session_id))
            #     track.task = task

            @track.on("ended")
            async def on_ended():
                logger.info(
                    f"[WebRTC] Track ended | kind={track.kind} | session_id={session_id}"
                )

    def _setup_datachannel_handler(self, pc: RTCPeerConnection, session_id: str):
        @pc.on("datachannel")
        def on_datachannel(channel):
            logger.info(
                f"[WebRTC] DataChannel received | label={channel.label} | session_id={session_id}"
            )

            @channel.on("message")
            def on_message(message):
                logger.debug(
                    f"[WebRTC] DataChannel message | session_id={session_id} | message={message}"
                )
        
        @pc.on("connectionstatechange")
        def on_connectionstatechange():
            logger.info(
                f"[WebRTC] Connection state changed | state={pc.connectionState} | session_id={session_id}"
            )
        
        @pc.on("iceconnectionstatechange")
        def on_iceconnectionstatechange():
            logger.info(
                f"[WebRTC] ICE connection state changed | state={pc.iceConnectionState} | session_id={session_id}"
            )
    
    def _create_detection_datachannel(self, pc: RTCPeerConnection, session_id: str) -> DetectionDataChannelManager:
        """Create outbound data channel for sending detection data"""
        try:
            logger.info(
                f"[DetectionChannel] Attempting to create channel | session_id={session_id} | "
                f"pc_state={pc.connectionState} | ice_state={pc.iceConnectionState}"
            )
            
            # Check if already created
            if session_id in DETECTION_CHANNELS and DETECTION_CHANNELS[session_id].channel:
                logger.info(
                    f"[DetectionChannel] Channel already exists for session | session_id={session_id}"
                )
                return DETECTION_CHANNELS[session_id]
            
            detection_channel = pc.createDataChannel("detections")
            logger.info(
                f"[DetectionChannel] createDataChannel() called | session_id={session_id} | "
                f"channel_object={detection_channel is not None} | readyState={detection_channel.readyState}"
            )
            
            channel_manager = DetectionDataChannelManager(session_id)
            channel_manager.set_channel(detection_channel)
            
            DETECTION_CHANNELS[session_id] = channel_manager
            
            logger.info(
                f"[DetectionChannel] Channel created and stored | session_id={session_id} | "
                f"readyState={detection_channel.readyState} | is_ready={channel_manager.is_ready}"
            )
            
            return channel_manager
        except Exception as e:
            logger.error(
                f"[DetectionChannel] FAILED to create channel | session_id={session_id} | "
                f"error_type={type(e).__name__} | error={e}",
                exc_info=True
            )
            raise

    async def _handle_sdp_offer(self, pc: RTCPeerConnection, sdp: str, type: str, session_id: str):
        try:
            offer = RTCSessionDescription(sdp=sdp, type=type)
            await pc.setRemoteDescription(offer)
        except Exception as e:
            logger.error(
                f"[WebRTC] Failed to set remote description | session_id={session_id} | error={str(e)}"
            )
            raise InvalidMessageError("Invalid SDP offer")

        # Create detection channel AFTER offer is set but BEFORE answer is created
        try:
            self._create_detection_datachannel(pc, session_id)
        except Exception as e:
            logger.error(
                f"[DetectionChannel] Error creating channel | session_id={session_id} | error={e}",
                exc_info=True
            )

        try:
            answer = await pc.createAnswer()
            await pc.setLocalDescription(answer)
            
            answer_sdp = pc.localDescription.sdp
            
            if "detections" in answer_sdp:
                logger.info(
                    f"[DetectionChannel] Channel found in answer SDP | session_id={session_id}"
                )
            else:
                logger.warning(
                    f"[DetectionChannel] WARNING - Channel NOT in answer SDP | session_id={session_id}"
                )
        except Exception as e:
            logger.error(
                f"[WebRTC] Failed to create answer | session_id={session_id} | error={str(e)}"
            )
            raise RuntimeError("Failed to create WebRTC answer")

        return {
            "sdp": pc.localDescription.sdp,
            "type": pc.localDescription.type,
            "session_id": session_id,
        }

    async def handle_offer(self, sdp: str, type: str):
        session_id = generate_session_id()
        pc = RTCPeerConnection()

        try:
            logger.info(f"[WebRTC] Creating new peer connection | session_id={session_id}")

            PEER_CONNECTIONS[session_id] = pc

            self._setup_track_handlers(pc, session_id)
            self._setup_datachannel_handler(pc, session_id)
            # NOTE: Detection data channel will be created in _handle_sdp_offer after receiving offer

            return await self._handle_sdp_offer(pc, sdp, type, session_id)

        except Exception:
            logger.exception(
                f"[WebRTC] Critical failure in handle_offer | session_id={session_id}"
            )

            # Cleanup to prevent memory leak
            await self._cleanup(session_id)

            raise

    async def _cleanup(self, session_id: str):
        pc = PEER_CONNECTIONS.pop(session_id, None)
        channel_manager = DETECTION_CHANNELS.pop(session_id, None)

        if channel_manager:
            try:
                await channel_manager.cleanup()
            except Exception:
                logger.error(
                    f"[DetectionChannel] Error cleaning up detection channel | session_id={session_id}"
                )

        if pc:
            try:
                await pc.close()
                logger.info(
                    f"[WebRTC] Peer connection closed | session_id={session_id}"
                )
            except Exception:
                logger.error(
                    f"[WebRTC] Error closing peer connection | session_id={session_id}"
                )
