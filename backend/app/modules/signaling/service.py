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
        self.face_analyzers = {}
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
        
        logger.info(
            f"[Detection] Starting 2-task pipeline | session_id={session_id} "
            f"detection_fps={detection_fps} yolo_enabled={enable_yolo} crop_enabled={enable_crop}"
        )
        
        # Initialize detectors
        yolo_detector = self._init_yolo_detector(session_id) if enable_yolo else None
        
        # Shared state between receiver and processor tasks
        latest_frame = None
        frame_available = asyncio.Event()
        frame_lock = asyncio.Lock()
        
        # Statistics
        stats = {
            "frames_received": 0,
            "frames_skipped": 0,
            "frames_processed": 0,
            "stats_start_time": time.monotonic(),
        }
        
        stop_processing = asyncio.Event()

        # ============================================
        # TASK 1: Frame Receiver (continuous drain)
        # ============================================
        async def receiver_task():
            nonlocal latest_frame
            logger.info(f"[Receiver] Task started | session_id={session_id}")
            frame_count = 0
            try:
                while not stop_processing.is_set():
                    try:
                        logger.debug(f"[Receiver] Waiting for frame (no timeout) | session_id={session_id} | frames_so_far={frame_count}")
                        # Don't use timeout - just wait for frame or let it fail naturally
                        frame = await track.recv()
                        frame_count += 1
                        stats["frames_received"] += 1
                        
                        logger.debug(
                            f"[Receiver] Frame received | session_id={session_id} "
                            f"frame_count={frame_count} total={stats['frames_received']}"
                        )
                        
                        # Update latest frame (overwrite stale frame)
                        async with frame_lock:
                            if latest_frame is not None:
                                stats["frames_skipped"] += 1
                                logger.debug(
                                    f"[Receiver] Stale frame overwritten | session_id={session_id} "
                                    f"skipped_total={stats['frames_skipped']}"
                                )
                            latest_frame = frame
                            frame_available.set()
                        
                    except Exception as e:
                        logger.info(
                            f"[Receiver] track.recv() failed or track ended | session_id={session_id} | "
                            f"error_type={type(e).__name__} error_msg={str(e)} | "
                            f"frames_received_so_far={frame_count}"
                        )
                        break
                        
            except asyncio.CancelledError:
                logger.info(f"[Receiver] Task cancelled | session_id={session_id} | frames_received={frame_count}")
            except Exception as e:
                logger.error(
                    f"[Receiver] Unexpected error | session_id={session_id} | "
                    f"error={type(e).__name__}:{e}"
                )
            finally:
                logger.info(
                    f"[Receiver] Task ended | session_id={session_id} | "
                    f"final_frames_received={stats['frames_received']}"
                )
        
        # ============================================
        # TASK 2: Frame Processor (periodic detection)
        # ============================================
        async def processor_task():
            nonlocal latest_frame
            frame_id = 0
            interval = 1.0 / detection_fps
            
            logger.info(
                f"[Processor] Task started | session_id={session_id} | "
                f"detection_fps={detection_fps} interval={interval:.3f}s"
            )
            
            try:
                while not stop_processing.is_set():
                    loop_start = time.monotonic()
                    
                    # Grab latest frame if available
                    frame = None
                    async with frame_lock:
                        if latest_frame is not None:
                            frame = latest_frame
                            latest_frame = None
                            frame_available.clear()
                    
                    # Process frame if available
                    if frame is not None:
                        try:
                            logger.debug(f"[Processor] Processing frame {frame_id} | session_id={session_id}")
                            
                            # Convert frame to OpenCV format (RGB)
                            loop_start_convert = time.monotonic()
                            img = frame.to_ndarray(format="rgb24")
                            convert_time = time.monotonic() - loop_start_convert
                            
                            # Apply cropping if enabled
                            img_to_process, crop_offset = self._calculate_crop_offset(
                                img, crop_enabled=enable_crop
                            )
                            
                            logger.debug(
                                f"[Processor] Frame converted | session_id={session_id} "
                                f"time={convert_time:.3f}s"
                            )
                            
                            # Run both MediaPipe and YOLO on the frame (concurrent)
                            async def detect_face():
                                logger.debug(f"[MediaPipe] Starting detection | session_id={session_id}")
                                mp_start = time.monotonic()
                                face_results = await asyncio.to_thread(
                                    self.face_detector.detect,
                                    img_to_process
                                )
                                mp_time = time.monotonic() - mp_start
                                logger.debug(
                                    f"[MediaPipe] Detection complete | session_id={session_id} "
                                    f"time={mp_time:.3f}s"
                                )
                                analyzer = self.face_analyzers.get(session_id)

                                if not analyzer:
                                    analyzer = FaceAnalyzer()
                                    self.face_analyzers[session_id] = analyzer

                                return analyzer.analyze(
                                    face_results,
                                    img_to_process.shape,
                                    session_id=session_id
                                )

                            async def detect_yolo():
                                if not yolo_detector:
                                    return []
                                logger.debug(f"[YOLO] Starting detection | session_id={session_id}")
                                yolo_start = time.monotonic()
                                detections = await asyncio.to_thread(yolo_detector.detect, img_to_process)
                                yolo_time = time.monotonic() - yolo_start
                                logger.debug(
                                    f"[YOLO] Detection complete | session_id={session_id} "
                                    f"detections={len(detections)} time={yolo_time:.3f}s"
                                )
                                return detections

                            # Run detections in parallel
                            inference_start = time.monotonic()
                            face_analysis, yolo_detections = await asyncio.gather(
                                detect_face(),
                                detect_yolo()
                            )
                            inference_time = time.monotonic() - inference_start
                            
                            logger.debug(
                                f"[Processor] Detection pipeline complete | session_id={session_id} "
                                f"face_alerts={len(face_analysis.get('alerts', []) if face_analysis else [])} "
                                f"yolo_detections={len(yolo_detections)} time={inference_time:.3f}s"
                            )

                            # Send results to frontend
                            send_start = time.monotonic()
                            self._send_combined_results(
                                session_id, frame_id, face_analysis, yolo_detections, crop_offset
                            )
                            send_time = time.monotonic() - send_start
                            logger.debug(
                                f"[Processor] Results sent | session_id={session_id} time={send_time:.3f}s"
                            )
                            
                            frame_id += 1
                            stats["frames_processed"] += 1
                            
                        except Exception as e:
                            logger.error(
                                f"[Processor] Error processing frame | session_id={session_id} "
                                f"frame_id={frame_id} error={e}"
                            )
                    else:
                        # Frame not available - this is normal if receiver hasn't sent any yet
                        # Stats will show incoming_fps=0 if this persists
                        pass
                    
                    # Print statistics every 1 second
                    now = time.monotonic()
                    elapsed = now - stats["stats_start_time"]
                    if elapsed >= 1.0:
                        incoming_fps = stats["frames_received"] / elapsed
                        processing_fps = stats["frames_processed"] / elapsed
                        
                        logger.info(
                            f"[Pipeline] Statistics | session_id={session_id} | "
                            f"incoming_fps={incoming_fps:.1f} | "
                            f"processing_fps={processing_fps:.1f}/{detection_fps} | "
                            f"frames_processed={stats['frames_processed']} | "
                            f"frames_skipped={stats['frames_skipped']}"
                        )
                        
                        # Reset counters
                        stats["frames_received"] = 0
                        stats["frames_processed"] = 0
                        stats["frames_skipped"] = 0
                        stats["stats_start_time"] = now
                    
                    # Sleep until next processing interval
                    loop_end = time.monotonic()
                    loop_time = loop_end - loop_start
                    sleep_time = max(0, interval - loop_time)
                    
                    if sleep_time > 0:
                        await asyncio.sleep(sleep_time)
                        
            except asyncio.CancelledError:
                logger.info(f"[Processor] Task cancelled | session_id={session_id}")
            except Exception as e:
                logger.error(f"[Processor] Error in processor task | session_id={session_id} error={e}")
        
        # ============================================
        # Run both tasks concurrently
        # ============================================
        receiver = asyncio.create_task(receiver_task())
        processor = asyncio.create_task(processor_task())
        
        logger.info(
            f"[Detection] Both tasks started | session_id={session_id} | "
            f"receiver_task_id={id(receiver)} processor_task_id={id(processor)}"
        )
        
        try:
            # Wait for tasks to complete (until one fails or connection closes)
            await asyncio.gather(receiver, processor)
        except asyncio.CancelledError:
            logger.info(f"[Detection] Pipeline cancelled | session_id={session_id}")
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
        finally:
            logger.info(f"[Detection] Pipeline shutdown | session_id={session_id} "
                       f"final_processed={stats['frames_processed']} "
                       f"final_skipped={stats['frames_skipped']}")

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
                # Cancel the processing task to prevent resource leak
                if hasattr(track, 'task') and track.task:
                    track.task.cancel()
                    try:
                        await track.task
                    except asyncio.CancelledError:
                        logger.debug(f"[Detection] Video track task cancelled | session_id={session_id}")

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
        DETECTION_CHANNELS.pop(session_id, None)
        self.data_channels.pop(session_id, None)
        analyzer = self.face_analyzers.pop(session_id, None)

        if pc:
            await pc.close()
            logger.info(f"[WebRTC] Session cleanup complete | session_id={session_id}")
        if analyzer:
            logger.info(f"[Analyzer] Resetting analyzer state | session_id={session_id}")
            analyzer.reset()