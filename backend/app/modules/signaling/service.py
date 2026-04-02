import asyncio
import time
import json
from typing import Dict, Tuple, Optional
from aiortc import RTCPeerConnection, RTCSessionDescription

from app.modules.signaling.detection_channel import DetectionDataChannelManager
from app.modules.signaling.detection_schema import DetectionFrame, CropOffset
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

    def _should_process_frame(self, last_inference_ts: float, detection_fps: int) -> tuple[bool, float]:
        """
        Determine if current frame should be processed based on DETECTION_FPS.
        
        If DETECTION_FPS=1, process 1 frame per second.
        If DETECTION_FPS=2, process 2 frames per second, etc.
        
        Args:
            last_inference_ts: Timestamp of last processed frame
            detection_fps: Target detection FPS from settings
            
        Returns:
            Tuple of (should_process: bool, current_timestamp: float)
        """
        if detection_fps <= 0:
            return True, last_inference_ts
        
        now = time.monotonic()
        time_since_last = now - last_inference_ts
        min_interval = 1.0 / detection_fps
        
        if time_since_last < min_interval:
            return False, last_inference_ts
        
        return True, now

    def _calculate_crop_offset(self, img, crop_enabled: bool = False) -> Tuple[any, CropOffset]:
        """Apply cropping if enabled and return cropped image with offset info
        
        Returns:
            Tuple of (cropped_image, CropOffset object)
        """
        h, w = img.shape[:2]
        crop_offset = CropOffset(
            original_width=w,
            original_height=h,
            cropped_width=w,
            cropped_height=h,
            x_offset=0,
            y_offset=0
        )
        
        # If cropping is enabled, apply it (e.g., crop center 80% of image)
        if crop_enabled and getattr(settings, "ENABLE_CROP", False):
            crop_percent = float(getattr(settings, "CROP_PERCENT", 0.8))
            crop_w = int(w * crop_percent)
            crop_h = int(h * crop_percent)
            
            # Center crop
            x_start = (w - crop_w) // 2
            y_start = (h - crop_h) // 2
            x_end = x_start + crop_w
            y_end = y_start + crop_h
            
            cropped_img = img[y_start:y_end, x_start:x_end]
            
            crop_offset = CropOffset(
                original_width=w,
                original_height=h,
                cropped_width=crop_w,
                cropped_height=crop_h,
                x_offset=x_start,
                y_offset=y_start
            )
            
            return cropped_img, crop_offset
        
        return img, crop_offset

    async def _process_video_track(self, track, session_id: str):
        """
        Process video track with frame throttling for both MediaPipe and YOLO.
        
        Throttles incoming frames based on DETECTION_FPS:
        - All incoming frames are tracked
        - Only DETECTION_FPS frames per second trigger detection (MediaPipe + YOLO)
        - Results are sent to frontend for each processed frame
        """
        detection_fps = int(getattr(settings, "DETECTION_FPS", 10))
        enable_yolo = getattr(settings, "ENABLE_YOLO", False)
        enable_crop = getattr(settings, "ENABLE_CROP", False)
        
        logger.info(
            f"[Detection] Start processing | session_id={session_id} "
            f"detection_fps={detection_fps} yolo_enabled={enable_yolo} crop_enabled={enable_crop}"
        )
        
        # Initialize detectors
        yolo_detector = self._init_yolo_detector(session_id) if enable_yolo else None
        
        # Frame tracking
        last_inference_ts = 0.0
        frame_id = 0
        total_frames = 0
        processed_frames = 0
        skipped_frames = 0
        stats_start_time = time.monotonic()

        while True:
            try:
                frame = await track.recv()
                total_frames += 1
                
                # Check if this frame should be processed (frame throttling)
                should_process, last_inference_ts = self._should_process_frame(
                    last_inference_ts, detection_fps
                )
                
                if not should_process:
                    skipped_frames += 1
                    continue
                
                # Frame will be processed - increment counter
                processed_frames += 1

                # Convert frame to OpenCV format (RGB)
                img = frame.to_ndarray(format="rgb24")
                
                # Apply cropping if enabled
                img_to_process, crop_offset = self._calculate_crop_offset(
                    img, crop_enabled=enable_crop
                )
                
                # Run both MediaPipe and YOLO on the throttled frame (concurrently)
                async def detect_face():
                    """MediaPipe face detection and analysis"""
                    face_results = await asyncio.to_thread(
                        self.face_detector.detect,
                        img_to_process
                    )
                    return self.face_analyzer.analyze(
                        face_results, img_to_process.shape, session_id=session_id
                    )

                async def detect_yolo():
                    """YOLO object detection"""
                    if yolo_detector:
                        return await asyncio.to_thread(yolo_detector.detect, img_to_process)
                    return []

                # Run detections in parallel (MediaPipe + YOLO)
                face_analysis, yolo_detections = await asyncio.gather(
                    detect_face(),
                    detect_yolo()
                )

                # Send results to frontend
                self._send_combined_results(
                    session_id, frame_id, face_analysis, yolo_detections, crop_offset
                )

                frame_id += 1
                
                # Print statistics every second
                now = time.monotonic()
                if now - stats_start_time >= 1.0:
                    incoming_fps = total_frames / (now - stats_start_time)
                    actual_processing_fps = processed_frames / (now - stats_start_time)

                    logger.info(
                        f"[Stats] session={session_id} | "
                        f"incoming_fps={incoming_fps:.1f} | "
                        f"processing_fps={actual_processing_fps:.1f}/{detection_fps} | "
                        f"frames_processed={processed_frames} | "
                        f"frames_skipped={skipped_frames}"
                    )

                    # Reset counters
                    total_frames = 0
                    processed_frames = 0
                    skipped_frames = 0
                    stats_start_time = now

            except asyncio.CancelledError:
                logger.info(f"[Detection] Processing cancelled | session_id={session_id}")
                raise
            except Exception as e:
                logger.error(f"[Detection] Error during processing | session_id={session_id} error={e}")
                raise

    def _send_combined_results(
        self,
        session_id: str,
        frame_id: int,
        face_analysis: dict,
        yolo_detections: list,
        crop_offset: CropOffset = None,
    ):
        """Send detection results (MediaPipe + YOLO) to frontend via WebRTC data channel."""
        raw_channel = self.data_channels.get(session_id)
        if not raw_channel:
            return

        try:
            # Serialize YOLO detections with crop offset transformation
            detection_frame = DetectionFrame.from_yolo_detections(
                frame_id=frame_id,
                timestamp=time.time(),
                yolo_detections=yolo_detections,
                crop_offset=crop_offset or CropOffset(),
            )
            frame_dict = detection_frame.to_dict()

            # Build unified payload with both MediaPipe and YOLO results
            payload = {
                "type": "detection_frame",
                "frame_id": frame_id,
                "timestamp": frame_dict["timestamp"],
                "face": face_analysis or {"alerts": [], "faces": [], "face_count": 0},
                "yolo": {
                    "detection_count": frame_dict["detection_count"],
                    "detections": frame_dict["detections"],
                },
                "crop_offset": frame_dict.get("crop_offset", {}),
            }

            raw_channel.send(json.dumps(payload))

        except Exception as e:
            logger.error(f"[Detection] Failed to send results | session_id={session_id} error={e}")

    def _setup_track_handlers(self, pc: RTCPeerConnection, session_id: str):
        @pc.on("track")
        def on_track(track):
            if track.kind == "video":
                logger.info(f"[Detection] Video track started | session_id={session_id} kind={track.kind}")
                track.task = asyncio.create_task(self._process_video_track(track, session_id))

            @track.on("ended")
            async def on_ended():
                logger.info(f"[Detection] Video track ended | session_id={session_id}")

    def _setup_datachannel_handler(self, pc: RTCPeerConnection, session_id: str):
        @pc.on("datachannel")
        def on_datachannel(channel):
            if channel.label == "detections":
                logger.info(f"[WebRTC] Data channel established | session_id={session_id} label={channel.label}")
                manager = DetectionDataChannelManager(session_id)
                manager.set_channel(channel)
                DETECTION_CHANNELS[session_id] = manager
                self.data_channels[session_id] = channel

        @pc.on("connectionstatechange")
        def on_connectionstatechange():
            logger.debug(f"[WebRTC] Connection state | session_id={session_id} state={pc.connectionState}")
            if pc.connectionState in ["failed", "closed"]:
                logger.info(f"[WebRTC] Connection ended | session_id={session_id} state={pc.connectionState}")
                pc.cleanup_task = asyncio.create_task(self._cleanup(session_id))
                
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
        """Clean up session resources."""
        pc = PEER_CONNECTIONS.pop(session_id, None)
        DETECTION_CHANNELS.pop(session_id, None)
        self.data_channels.pop(session_id, None)

        if pc:
            await pc.close()
            logger.info(f"[WebRTC] Session cleanup complete | session_id={session_id}")