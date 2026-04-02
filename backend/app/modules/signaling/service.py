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
_CLEANUP_SCHEDULED: set = set()  # Track which sessions have cleanup scheduled to prevent double-cleanup
_CLEANUP_SCHEDULED: set = set()  # Track which sessions have cleanup scheduled

class WebRTCService:
    def __init__(self):
        # Initialize detectors once at service level or per session
        self.face_detector = MediaPipeFaceDetector()
        self.face_analyzers = {}
        self.data_channels = {}  # Direct references to aiortc channels

    def _init_yolo_detector(self, session_id: str):
        """Initialize YOLO detector with settings"""
        return YoloDetector(
            model_path=getattr(settings, "YOLO_MODEL_PATH", "yolov8n.pt"),
            conf_threshold=float(getattr(settings, "YOLO_CONF_THRESHOLD", 0.25)),
        )

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
        Real-time detection pipeline with 2-task architecture:
        
        Task 1 (Receiver):
        - Continuously receives frames from WebRTC track
        - Stores only the latest frame in memory
        - Old unprocessed frames are automatically dropped
        - Runs independently at WebRTC frame rate (~30 FPS from browser)
        
        Task 2 (Processor):  
        - Runs periodically at configured DETECTION_FPS (e.g., 1, 2, 5, 10)
        - Grabs the latest available frame
        - Runs MediaPipe + YOLO in parallel
        - Sends results to frontend
        - Skips if no new frame is available
        
        This ensures:
        - Fresh frames are always processed (not old stale ones)
        - No buildup of frame queues
        - Receiver never blocks
        - Proper real-time analytics behavior
        """
        detection_fps = int(getattr(settings, "DETECTION_FPS", 10))
        enable_yolo = getattr(settings, "ENABLE_YOLO", False)
        enable_crop = getattr(settings, "ENABLE_CROP", False)
        
        # Initialize detectors
        yolo_detector = self._init_yolo_detector(session_id) if enable_yolo else None
        
        # Shared state between receiver and processor tasks
        latest_frame = None
        frame_lock = asyncio.Lock()
        stop_processing = asyncio.Event()

        # ============================================
        # TASK 1: Frame Receiver (continuous drain)
        # ============================================
        async def receiver_task():
            nonlocal latest_frame
            try:
                while not stop_processing.is_set():
                    try:
                        # Don't use timeout - just wait for frame or let it fail naturally
                        frame = await track.recv()
                        
                        # Update latest frame (overwrite stale frame)
                        async with frame_lock:
                            latest_frame = frame
                        
                    except Exception:
                        break
                        
            except asyncio.CancelledError:
                pass
            except Exception as e:
                logger.error(f"[Receiver] Unexpected error | session_id={session_id} | error={e}")
            finally:
                # CRITICAL FIX #1: Stop processor when receiver exits
                stop_processing.set()
        
        # ============================================
        # TASK 2: Frame Processor (periodic detection)
        # ============================================
        async def processor_task():
            nonlocal latest_frame
            frame_id = 0
            interval = 1.0 / detection_fps
            
            try:
                while not stop_processing.is_set():
                    loop_start = time.monotonic()
                    
                    # Grab latest frame if available
                    frame = None
                    async with frame_lock:
                        if latest_frame is not None:
                            frame = latest_frame
                            latest_frame = None
                    
                    # Process frame if available
                    if frame is not None:
                        try:
                            # Convert frame to OpenCV format (RGB)
                            img = frame.to_ndarray(format="rgb24")
                            
                            # Apply cropping if enabled
                            img_to_process, crop_offset = self._calculate_crop_offset(
                                img, crop_enabled=enable_crop
                            )
                            
                            # Run both MediaPipe and YOLO on the frame (concurrent)
                            async def detect_face():
                                face_results = await asyncio.to_thread(
                                    self.face_detector.detect,
                                    img_to_process
                                )
                                # FIX #4: Analyzer should be initialized once per session, not per-frame
                                analyzer = self.face_analyzers.get(session_id)
                                if not analyzer:
                                    analyzer = FaceAnalyzer()
                                    self.face_analyzers[session_id] = analyzer

                                face_analysis = analyzer.analyze(
                                    face_results,
                                    img_to_process.shape,
                                    session_id=session_id
                                )
                                
                                return face_analysis

                            async def detect_yolo():
                                if not yolo_detector:
                                    return []
                                detections = await asyncio.to_thread(yolo_detector.detect, img_to_process)
                                return detections

                            # Run detections in parallel
                            face_analysis, yolo_detections = await asyncio.gather(
                                detect_face(),
                                detect_yolo()
                            )
                            person_count = len(yolo_detections)
                            face_count = face_analysis.get("face_count", 0)

                            alerts = list(face_analysis.get("alerts") or [])

                            if person_count > 1:
                                alerts.append("MULTIPLE_PERSONS")

                            if person_count >= 1 and face_count == 0:
                                alerts.append("PERSON_PRESENT_NO_FACE")

                            face_analysis["alerts"] = list(set(alerts))

                            # Send results to frontend
                            self._send_combined_results(
                                session_id, frame_id, face_analysis, yolo_detections, crop_offset
                            )
                            
                            frame_id += 1
                            
                        except Exception as e:
                            logger.error(
                                f"[Processor] Error processing frame | session_id={session_id} "
                                f"frame_id={frame_id} error={e}"
                            )
                    else:
                        # Frame not available - this is normal if receiver hasn't sent any yet
                        # Stats will show incoming_fps=0 if this persists
                        pass
                    

                    
                    # Sleep until next processing interval
                    loop_end = time.monotonic()
                    loop_time = loop_end - loop_start
                    sleep_time = max(0, interval - loop_time)
                    
                    if sleep_time > 0:
                        await asyncio.sleep(sleep_time)
                        
            except asyncio.CancelledError:
                pass
            except Exception as e:
                logger.error(f"[Processor] Error | session_id={session_id} | error={e}")
        
        # ============================================
        # Run both tasks concurrently
        # ============================================
        receiver = asyncio.create_task(receiver_task())
        processor = asyncio.create_task(processor_task())
        
        try:
            # FIX #2: Use asyncio.wait with FIRST_COMPLETED for better error handling
            # If one task fails, cancel the other instead of waiting for both
            done, pending = await asyncio.wait(
                {receiver, processor},
                return_when=asyncio.FIRST_COMPLETED
            )
            
            # One of the tasks completed - check if it's an error
            for task in done:
                try:
                    await task
                except Exception as e:
                    logger.error(f"[Detection] Task failed with error | session_id={session_id} error={e}")
                    # Cancel all pending tasks
                    for pending_task in pending:
                        pending_task.cancel()
                    raise
                    
        except asyncio.CancelledError:
            stop_processing.set()
            receiver.cancel()
            processor.cancel()
            try:
                await asyncio.gather(receiver, processor, return_exceptions=True)
            except:
                pass
        except Exception as e:
            logger.error(f"[Detection] Pipeline error | session_id={session_id} error={e}")
            stop_processing.set()
            # Ensure both tasks are cancelled
            receiver.cancel()
            processor.cancel()
            try:
                await asyncio.gather(receiver, processor, return_exceptions=True)
            except:
                pass
        finally:
            pass

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
            logger.error(f"[Send] Failed | session_id={session_id} error={e}")


    def _setup_track_handlers(self, pc: RTCPeerConnection, session_id: str):
        @pc.on("track")
        def on_track(track):
            if track.kind == "video":
                track.task = asyncio.create_task(self._process_video_track(track, session_id))

            @track.on("ended")
            async def on_ended():
                # Cancel the processing task to prevent resource leak
                if hasattr(track, 'task') and track.task:
                    track.task.cancel()
                    try:
                        await track.task
                    except asyncio.CancelledError:
                        pass

    def _setup_datachannel_handler(self, pc: RTCPeerConnection, session_id: str):
        @pc.on("datachannel")
        def on_datachannel(channel):
            if channel.label == "detections":
                manager = DetectionDataChannelManager(session_id)
                manager.set_channel(channel)
                DETECTION_CHANNELS[session_id] = manager
                self.data_channels[session_id] = channel

        @pc.on("connectionstatechange")
        def on_connectionstatechange():
            if pc.connectionState in ["failed", "closed"]:
                # FIX #9: Prevent double cleanup by checking if already scheduled
                if session_id not in _CLEANUP_SCHEDULED:
                    _CLEANUP_SCHEDULED.add(session_id)
                    asyncio.create_task(self._cleanup(session_id))
                
    async def handle_offer(self, sdp: str, type: str):
        session_id = generate_session_id()
        self.face_analyzers[session_id] = FaceAnalyzer()
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
        # FIX #5/#6: Properly cleanup the manager and call its cleanup method
        manager = DETECTION_CHANNELS.pop(session_id, None)
        self.data_channels.pop(session_id, None)
        analyzer = self.face_analyzers.pop(session_id, None)

        # Call manager.cleanup() before discarding
        if manager:
            try:
                await manager.cleanup()
            except Exception as e:
                logger.error(f"[DetectionChannel] Cleanup failed | session_id={session_id} | error={e}")

        if pc:
            await pc.close()
        if analyzer:
            analyzer.reset()