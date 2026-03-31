import asyncio
import numpy as np
import time
import json
import cv2
from typing import Dict
from aiortc import RTCPeerConnection, RTCSessionDescription

from app.modules.signaling.detection_channel import DetectionDataChannelManager
from app.modules.signaling.detection_schema import DetectionFrame
from app.modules.detection.yolo_detector import YoloDetector
from app.modules.detection.mediapipe_face import MediaPipeFaceDetector
from app.modules.analytics.face_analyzer import FaceAnalyzer
from app.api.utils.utils import generate_session_id
from app.common.exceptions import InvalidMessageError
from app.core.config import settings
from app.logger import logger

# Global state trackers
PEER_CONNECTIONS: Dict[str, RTCPeerConnection] = {}
DETECTION_CHANNELS: Dict[str, DetectionDataChannelManager] = {}

class WebRTCService:
    def __init__(self):
        # Initialize detectors once at service level or per session
        self.face_detector = MediaPipeFaceDetector()
        self.face_analyzer = FaceAnalyzer()
        self.data_channels = {}  # Direct references to aiortc channels

    def _init_yolo_detector(self, session_id: str):
        """Initialize YOLO detector with settings"""
        return YoloDetector(
            model_path=getattr(settings, "YOLO_MODEL_PATH", "yolov8n.pt"),
            conf_threshold=float(getattr(settings, "YOLO_CONF_THRESHOLD", 0.25)),
        )

    def _should_process_frame(self, last_inference_ts: float) -> tuple[bool, float]:
        """Throttle processing based on DETECTION_FPS"""
        fps = int(getattr(settings, "DETECTION_FPS", 10))
        if fps <= 0: return True, last_inference_ts
        
        now = time.monotonic()
        if now - last_inference_ts < (1.0 / fps):
            return False, last_inference_ts
        return True, now

    async def _process_video_track(self, track, session_id: str):
        """Main Loop: Processes video and runs both YOLO and MediaPipe"""
        logger.info(f"[WebRTC] Start video processing | session_id={session_id}")
        
        last_inference_ts = 0.0
        frame_id = 0
        total_frames = 0
        processed_frames = 0
        skipped_frames = 0
        
        start_time = time.monotonic()

        # Initialize YOLO if enabled
        yolo_enabled = getattr(settings, "ENABLE_YOLO", False)
        yolo_detector = self._init_yolo_detector(session_id) if yolo_enabled else None

        while True:
            try:
                frame = await track.recv()
                total_frames += 1
                
                # FPS Throttling
                should_process, last_inference_ts = self._should_process_frame(last_inference_ts)
                if not should_process:
                    skipped_frames += 1
                    continue
                processed_frames += 1

                # Convert to numpy/OpenCV format
                img = frame.to_ndarray(format="bgr24")
                
                # 1. Run MediaPipe Face Analysis (Always run if track exists)
                face_results = await asyncio.to_thread(
                    self.face_detector.detect,
                    img
                )   
                face_analysis = self.face_analyzer.analyze(
                    face_results, img.shape, session_id=session_id
                )

                # 2. Run YOLO Detection (If enabled)
                yolo_detections = []
                if yolo_detector:
                    yolo_detections = await asyncio.to_thread(yolo_detector.detect, img)

                # 3. Send Unified Results
                self._send_combined_results(
                    session_id, frame_id, face_analysis, yolo_detections
                )

                frame_id += 1
                now = time.monotonic()
                if now - start_time >= 1.0:
                    incoming_fps = total_frames / (now - start_time)
                    processing_fps = processed_frames / (now - start_time)

                    logger.info(
                        f"[FPS] session={session_id} | "
                        f"incoming={incoming_fps:.2f} FPS | "
                        f"processed={processing_fps:.2f} FPS | "
                        f"skipped={skipped_frames}"
                    )

                    # reset counters
                    total_frames = 0
                    processed_frames = 0
                    skipped_frames = 0
                    start_time = now


            except Exception as e:
                logger.error(f"[WebRTC] Video processing error: {e}")
                break

    def _send_combined_results(
        self,
        session_id,
        frame_id,
        face_analysis,
        yolo_detections
    ):
        raw_channel = self.data_channels.get(session_id)

        if not raw_channel:
            return

        try:
            # SERIALIZE YOLO DETECTIONS
            serialized_detections = [
                {
                    "class_id": det.class_id,
                    "class_name": det.class_name,
                    "confidence": det.confidence,
                    "bbox": {
                        "x1": det.xyxy[0],
                        "y1": det.xyxy[1],
                        "x2": det.xyxy[2],
                        "y2": det.xyxy[3],
                    }
                }
                for det in (yolo_detections or [])
            ]

            payload = {
                "type": "detection_frame",
                "frame_id": frame_id,
                "timestamp": time.time(),

                "face": face_analysis or {
                    "alerts": [],
                    "faces": [],
                    "face_count": 0
                },

                "yolo": {
                    "detection_count": len(serialized_detections),
                    "detections": serialized_detections
                }
            }

            raw_channel.send(json.dumps(payload))

        except Exception as e:
            logger.error(f"[WebRTC] Combined data send error: {e}")

    def _setup_track_handlers(self, pc: RTCPeerConnection, session_id: str):
        @pc.on("track")
        def on_track(track):
            if track.kind == "video":
                track.task = asyncio.create_task(self._process_video_track(track, session_id))

            @track.on("ended")
            async def on_ended():
                logger.info(f"[WebRTC] Track ended | session_id={session_id}")

    def _setup_datachannel_handler(self, pc: RTCPeerConnection, session_id: str):
        @pc.on("datachannel")
        def on_datachannel(channel):
            if channel.label == "detections":
                # Setup YOLO manager
                manager = DetectionDataChannelManager(session_id)
                manager.set_channel(channel)
                DETECTION_CHANNELS[session_id] = manager
                # Setup Mediapipe direct reference
                self.data_channels[session_id] = channel

        @pc.on("connectionstatechange")
        def on_connectionstatechange():
            if pc.connectionState in ["failed", "closed"]:
                asyncio.create_task(self._cleanup(session_id))

    async def handle_offer(self, sdp: str, type: str):
        session_id = generate_session_id()
        pc = RTCPeerConnection()
        PEER_CONNECTIONS[session_id] = pc

        self._setup_track_handlers(pc, session_id)
        self._setup_datachannel_handler(pc, session_id)

        offer = RTCSessionDescription(sdp=sdp, type=type)
        await pc.setRemoteDescription(offer)
        
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)

        return {
            "sdp": pc.localDescription.sdp,
            "type": pc.localDescription.type,
            "session_id": session_id,
        }

    async def _cleanup(self, session_id: str):
        pc = PEER_CONNECTIONS.pop(session_id, None)
        DETECTION_CHANNELS.pop(session_id, None)
        self.data_channels.pop(session_id, None)

        if pc:
            await pc.close()
            logger.info(f"[WebRTC] Cleaned up session: {session_id}")