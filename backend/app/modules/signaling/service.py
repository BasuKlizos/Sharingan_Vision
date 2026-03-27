import time
import json
import asyncio
import numpy as np
import time
from typing import Dict
from fastapi import WebSocket
import cv2 

import cv2
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
from app.modules.detection.mediapipe_face import MediaPipeFaceDetector
from app.modules.analytics.face_analyzer import FaceAnalyzer
from app.api.utils.draw_utils import draw_landmarks

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
    def __init__(self):
        self.face_detector = MediaPipeFaceDetector()
        self.face_analyzer = FaceAnalyzer()
        self.data_channels = {}
    
    async def _process_video_track(self, track, session_id: str):
        """Process incoming video track and run detection models"""
        logger.info(f"[WebRTC] Start video processing | session_id={session_id}")

        target_fps = 15
        frame_interval = 1 / target_fps
        frame_count = 0

        last_inference_ts = 0.0
        frame_id = 0
        
        # Initialize detector
        if not getattr(settings, "ENABLE_YOLO", False):
            logger.info(f"[WebRTC] YOLO disabled for session_id={session_id}")
            return
        
        detector = self._init_detector(session_id)

        while True:
            start_time = time.time()

            try:
                frame = await track.recv()
                frame_count += 1

                # Skip frames
                # if frame_count % 2 != 0:
                #     continue
                
                # Check if we should process this frame based on FPS
                should_process, last_inference_ts = self._should_process_frame(last_inference_ts)
                if not should_process:
                    frame_id += 1
                    continue

                # Convert frame to ndarray
                img = frame.to_ndarray(format="bgr24")

                # Prepare frame (resize if needed)
                img = self._prepare_frame(img, session_id)

                # Run YOLO detection
                detections = await self._run_detection(img, detector, session_id)
                
                # TODO: Run MediaPipe detection here and merge results
                # mediapipe_detections = await self._run_mediapipe_detection(img, mediapipe_detector, session_id)
                # combined_detections = self._merge_detections(detections, mediapipe_detections)
                # Detection
                results = self.face_detector.detect(img)
                
                # Analysis
                analysis = self.face_analyzer.analyze(
                    results,
                    img.shape,
                    session_id=session_id
                )

                if analysis["alerts"]:
                    logger.warning(
                        f"[AI] Alerts: {analysis['alerts']} | session_id={session_id}"
                    )

                # Send detection results to frontend
                await self._send_detection_results(detections, frame_id, session_id)
                
                frame_id += 1
                
                if frame_count % 5 == 0:
                    self._send_analysis(session_id, analysis)
            except Exception as e:
                logger.error(f"[WebRTC] Video frame error: {e}")
                break
        

            # FPS throttle
            elapsed = time.time() - start_time
            sleep_time = frame_interval - elapsed
            logger.debug(f"sleep_time={sleep_time}")
            # if sleep_time > 0:
            #     await asyncio.sleep(sleep_time)

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
                f"[WebRTC] DataChannel connected | session_id={session_id}"
            )

            # Handle detection channel from frontend
            if channel.label == "detections":
                self._setup_detection_channel(channel, session_id)
            

            self.data_channels[session_id] = channel

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
                self.data_channels.pop(session_id, None)
                logger.info(
                    f"[WebRTC] Peer connection closed | session_id={session_id}"
                )
            except Exception:
                logger.error(
                    f"[WebRTC] Error closing peer connection | session_id={session_id}"
                )
