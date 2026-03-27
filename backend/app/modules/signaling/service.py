import time
import json
import asyncio
import numpy as np
from typing import Dict
from fastapi import WebSocket

import cv2
from aiortc import RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaBlackhole


from app.modules.signaling.interfaces import BaseConnectionManager
from app.api.utils.utils import generate_session_id
from app.common.exceptions import InvalidMessageError
from app.logger import logger
from app.modules.detection.mediapipe_face import MediaPipeFaceDetector
from app.modules.analytics.face_analyzer import FaceAnalyzer
from app.api.utils.draw_utils import draw_landmarks

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
    def __init__(self):
        self.face_detector = MediaPipeFaceDetector()
        self.face_analyzer = FaceAnalyzer()
        self.data_channels = {}
    
    async def _process_video_track(self, track, session_id: str):
        logger.info(f"[WebRTC] Start video processing | session_id={session_id}")

        target_fps = 15
        frame_interval = 1 / target_fps
        frame_count = 0

        while True:
            start_time = time.time()

            try:
                frame = await track.recv()
                frame_count += 1

                # Skip frames
                if frame_count % 2 != 0:
                    continue

                img = frame.to_ndarray(format="bgr24")
                img = cv2.resize(img, (640, 480))

                # Detection
                results = self.face_detector.detect(img)

                # Analysis
                analysis = self.face_analyzer.analyze(
                    results,
                    img.shape,
                    session_id=session_id
                )

                # Drawing
                # img = draw_landmarks(img, results)

                # Logging alerts
                if analysis["alerts"]:
                    logger.warning(
                        f"[AI] Alerts: {analysis['alerts']} | session_id={session_id}"
                    )
                
                if frame_count % 5 == 0:
                    self._send_analysis(session_id, analysis)

            except Exception as e:
                logger.error(f"[WebRTC] Video frame error: {e}")
                break

            # FPS throttle
            elapsed = time.time() - start_time
            sleep_time = frame_interval - elapsed

            if sleep_time > 0:
                await asyncio.sleep(sleep_time)
    def _send_analysis(self, session_id: str, analysis: dict):
        channel = self.data_channels.get(session_id)

        if not channel:
            return

        logger.debug(
            f"[DataChannel] state={channel.readyState} | session_id={session_id}"
        )

        if channel.readyState != "open":
            self.data_channels.pop(session_id, None)
            return

        try:
            payload = json.dumps({
                "type": "face_analysis",
                "data": analysis
            })

            channel.send(payload)

            logger.debug(
                f"[DataChannel] Sent analysis | session_id={session_id} | payload={payload}"
            )

        except Exception as e:
            logger.error(
                f"[DataChannel] Send failed | session_id={session_id} | error={e}"
            )

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
                f"[WebRTC] DataChannel connected | session_id={session_id}"
            )

            self.data_channels[session_id] = channel

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
                self.data_channels.pop(session_id, None)
                logger.info(
                    f"[WebRTC] Peer connection closed | session_id={session_id}"
                )
            except Exception:
                logger.error(
                    f"[WebRTC] Error closing peer connection | session_id={session_id}"
                )