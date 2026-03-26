import asyncio
import numpy as np
import time
from typing import Dict
from fastapi import WebSocket
import cv2 

from aiortc import RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaBlackhole

from app.modules.signaling.interfaces import BaseConnectionManager
from app.modules.detection.yolo_detector import YoloDetector
from app.api.utils.utils import generate_session_id
from app.common.exceptions import InvalidMessageError
from app.core.config import settings
from app.logger import logger

PEER_CONNECTIONS: Dict[str, RTCPeerConnection] = {}

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

                logger.debug(
                    f"[VideoFrame] pts={frame.pts}, size={frame.width}x{frame.height}"
                )

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

    async def _handle_sdp_offer(self, pc: RTCPeerConnection, sdp: str, type: str, session_id: str):
        try:
            offer = RTCSessionDescription(sdp=sdp, type=type)
            await pc.setRemoteDescription(offer)
        except Exception as e:
            logger.error(
                f"[WebRTC] Failed to set remote description | session_id={session_id} | error={str(e)}"
            )
            raise InvalidMessageError("Invalid SDP offer")

        try:
            answer = await pc.createAnswer()
            await pc.setLocalDescription(answer)
        except Exception as e:
            logger.error(
                f"[WebRTC] Failed to create answer | session_id={session_id} | error={str(e)}"
            )
            raise RuntimeError("Failed to create WebRTC answer")

        logger.info(
            f"[WebRTC] Answer created successfully | session_id={session_id}"
        )

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
