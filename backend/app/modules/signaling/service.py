import asyncio
import numpy as np
import time
from typing import Dict
from fastapi import WebSocket
import cv2 

from aiortc import RTCPeerConnection, RTCSessionDescription

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
        logger.info(
            "[WebRTC] YOLO config | session_id=%s | enabled=%s | model_path=%s | conf=%s | max_width=%s | fps=%s",
            session_id,
            getattr(settings, "ENABLE_YOLO", False),
            getattr(settings, "YOLO_MODEL_PATH", "yolov8n.pt"),
            getattr(settings, "YOLO_CONF_THRESHOLD", 0.25),
            getattr(settings, "YOLO_MAX_WIDTH", 640),
            getattr(settings, "DETECTION_FPS", 5),
        )

        while True:
            try:
                frame = await track.recv()

                # logger.debug(
                #     f"[VideoFrame] pts={frame.pts}, size={frame.width}x{frame.height}"
                # )

                if not getattr(settings, "ENABLE_YOLO", False):
                    continue

                fps = int(getattr(settings, "DETECTION_FPS", 10))
                if fps > 0:
                    now = time.monotonic()
                    if now - last_inference_ts < (1.0 / fps):
                        continue
                    last_inference_ts = now

                img = frame.to_ndarray(format="bgr24")

                # Optional: downscale for speed while keeping aspect ratio
                max_w = int(getattr(settings, "YOLO_MAX_WIDTH", 640))
                if max_w > 0 and img.shape[1] > max_w:
                    scale = max_w / float(img.shape[1])
                    logger.debug(
                        "[WebRTC] Resizing frame | session_id=%s | from=%sx%s | to_width=%s",
                        session_id,
                        img.shape[1],
                        img.shape[0],
                        max_w,
                    )
                    img = cv2.resize(
                        img,
                        (max_w, int(img.shape[0] * scale)),
                        interpolation=cv2.INTER_AREA,
                    )

                t0 = time.monotonic()
                detections = await asyncio.to_thread(detector.detect, img)
                infer_ms = (time.monotonic() - t0) * 1000.0
                logger.debug(
                    "[WebRTC] YOLO inference time | session_id=%s | ms=%.1f",
                    session_id,
                    infer_ms,
                )
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

    def _setup_detection_channel(self, channel, session_id: str):
        """Set up detection data channel received from frontend"""
        logger.info(
            f"[DetectionChannel] Incoming detection channel from frontend | "
            f"session_id={session_id} | readyState={channel.readyState}"
        )
        
        # Create channel manager for this incoming channel
        channel_manager = DetectionDataChannelManager(session_id)
        channel_manager.set_channel(channel)
        DETECTION_CHANNELS[session_id] = channel_manager
        
        logger.info(
            f"[DetectionChannel] Channel manager set up for incoming channel | "
            f"session_id={session_id} | is_ready={channel_manager.is_ready}"
        )

    def _setup_datachannel_handler(self, pc: RTCPeerConnection, session_id: str):
        @pc.on("datachannel")
        def on_datachannel(channel):
            logger.info(
                f"[WebRTC] DataChannel received | label={channel.label} | session_id={session_id}"
            )

            # Handle detection channel from frontend
            if channel.label == "detections":
                self._setup_detection_channel(channel, session_id)
            
            @channel.on("message")
            def on_message(message):
                logger.debug(
                    f"[WebRTC] DataChannel message | session_id={session_id} | "
                    f"label={channel.label} | message={message}"
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
    


    async def _handle_sdp_offer(self, pc: RTCPeerConnection, sdp: str, type: str, session_id: str):
        try:
            offer = RTCSessionDescription(sdp=sdp, type=type)
            await pc.setRemoteDescription(offer)
        except Exception as e:
            logger.error(
                f"[WebRTC] Failed to set remote description | session_id={session_id} | error={str(e)}"
            )
            raise InvalidMessageError("Invalid SDP offer")

        # Frontend will create the detection channel, we just handle it in _setup_datachannel_handler
        try:
            answer = await pc.createAnswer()
            await pc.setLocalDescription(answer)
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
