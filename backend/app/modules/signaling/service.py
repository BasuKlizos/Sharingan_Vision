import asyncio
import time
import json
from typing import Dict, Tuple, Optional
from aiortc import RTCPeerConnection, RTCSessionDescription, RTCIceServer, RTCConfiguration

from app.modules.signaling.detection_channel import DetectionDataChannelManager
from app.modules.signaling.detection_schema import DetectionFrame, CropOffset
from app.modules.detection.yolo_detector import YoloDetector
from app.modules.detection.mediapipe_face import MediaPipeFaceDetector
from app.modules.analytics.face_analyzer import FaceAnalyzer
from app.modules.monitoring.schemas import CurrentViewData
from app.api.utils.utils import generate_session_id
from app.common.exceptions import InvalidMessageError
from app.core.config import settings
from app.logger import logger
from app.modules.monitoring.store import session_monitoring_store

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

    def _get_or_create_face_analyzer(self, session_id: str) -> FaceAnalyzer:
        analyzer = self.face_analyzers.get(session_id)
        if analyzer:
            return analyzer

        analyzer = FaceAnalyzer()
        self.face_analyzers[session_id] = analyzer
        return analyzer

    async def _detect_face_analysis(self, img_to_process, session_id: str) -> dict:
        face_results = await asyncio.to_thread(self.face_detector.detect, img_to_process)
        analyzer = self._get_or_create_face_analyzer(session_id)
        return analyzer.analyze(face_results, img_to_process.shape, session_id=session_id)

    async def _detect_yolo_detections(self, yolo_detector, img_to_process) -> list:
        if not yolo_detector:
            return []
        return await asyncio.to_thread(yolo_detector.detect, img_to_process)

    def _merge_detection_alerts(self, face_analysis: dict, yolo_detections: list) -> dict:
        # Filter the list to only count actual people
        actual_people = [d for d in yolo_detections if d.class_name == "person"]
        person_count = len(actual_people)
        
        # You can also count the devices separately if you need them!
        device_count = len(yolo_detections) - person_count
        
        face_count = face_analysis.get("face_count", 0)
        alerts = list(face_analysis.get("alerts") or [])

        # Logic: If YOLO sees a body (back turned, side view) but MediaPipe Mesh fails to find a face
        if person_count >= 1 and face_count == 0:
            alerts.append("PERSON_PRESENT_NO_FACE")

        session_id = face_analysis.get("session_id")
        if session_id:
            session = session_monitoring_store.get_session(session_id)
            if session and session.calibration is not None and session.latest_zone_assessment is not None:
                try:
                    zone_assessment = session.latest_zone_assessment
                    face_analysis["zone_assessment"] = zone_assessment
                    if session.latest_current_view is not None:
                        face_analysis["webrtc_current_view"] = session.latest_current_view.model_dump(
                            by_alias=True
                        )

                except Exception as exc:
                    logger.warning(
                        f"[Send] Failed to assess zone alert | session_id={session_id} error={exc}"
                    )
            
        face_analysis["person_count"] = person_count
        face_analysis["device_count"] = device_count # Helpful for your electronics focus!
        face_analysis["alerts"] = list(set(alerts))
        return face_analysis

    async def _process_single_frame(
        self,
        frame,
        session_id: str,
        frame_id: int,
        enable_crop: bool,
        yolo_detector,
    ) -> None:
        img = frame.to_ndarray(format="rgb24")
        img_to_process, crop_offset = self._calculate_crop_offset(img, crop_enabled=enable_crop)

        face_analysis, yolo_detections = await asyncio.gather(
            self._detect_face_analysis(img_to_process, session_id),
            self._detect_yolo_detections(yolo_detector, img_to_process),
        )
        current_view = (face_analysis or {}).get("current_view")
        if current_view is not None:
            try:
                session_monitoring_store.update_current_view_for_session(
                    session_id=session_id,
                    current_view=CurrentViewData.model_validate(current_view),
                )
            except Exception as exc:
                logger.warning(
                    f"[Process] Failed to update current view from face analysis | "
                    f"session_id={session_id} frame_id={frame_id} error={exc}"
                )
        face_analysis = self._merge_detection_alerts(face_analysis, yolo_detections)

        self._send_combined_results(
            session_id, frame_id, face_analysis, yolo_detections, crop_offset
        )

    async def _receiver_task_loop(
        self,
        track,
        session_id: str,
        frame_state: dict,
        frame_lock: asyncio.Lock,
        stop_processing: asyncio.Event,
    ) -> None:
        try:
            while not stop_processing.is_set():
                try:
                    frame = await track.recv()
                    async with frame_lock:
                        frame_state["latest_frame"] = frame
                except Exception:
                    break
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"[Receiver] Unexpected error | session_id={session_id} | error={e}")
        finally:
            stop_processing.set()

    async def _pop_latest_frame(self, frame_state: dict, frame_lock: asyncio.Lock):
        async with frame_lock:
            frame = frame_state.get("latest_frame")
            frame_state["latest_frame"] = None
        return frame

    async def _processor_task_loop(
        self,
        session_id: str,
        detection_fps: int,
        enable_crop: bool,
        yolo_detector,
        frame_state: dict,
        frame_lock: asyncio.Lock,
        stop_processing: asyncio.Event,
    ) -> None:
        frame_id = 0
        interval = 1.0 / detection_fps

        try:
            while not stop_processing.is_set():
                loop_start = time.monotonic()
                frame = await self._pop_latest_frame(frame_state, frame_lock)

                if frame is not None:
                    try:
                        await self._process_single_frame(
                            frame=frame,
                            session_id=session_id,
                            frame_id=frame_id,
                            enable_crop=enable_crop,
                            yolo_detector=yolo_detector,
                        )
                        frame_id += 1
                    except Exception as e:
                        logger.error(
                            f"[Processor] Error processing frame | session_id={session_id} "
                            f"frame_id={frame_id} error={e}"
                        )

                sleep_time = max(0, interval - (time.monotonic() - loop_start))
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"[Processor] Error | session_id={session_id} | error={e}")

    async def _cancel_detection_tasks(self, *tasks: asyncio.Task) -> None:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    async def _await_detection_tasks(
        self,
        receiver: asyncio.Task,
        processor: asyncio.Task,
        session_id: str,
    ) -> None:
        done, pending = await asyncio.wait(
            {receiver, processor},
            return_when=asyncio.FIRST_COMPLETED
        )

        for task in done:
            try:
                await task
            except Exception as e:
                logger.error(f"[Detection] Task failed with error | session_id={session_id} error={e}")
                for pending_task in pending:
                    pending_task.cancel()
                await asyncio.gather(*pending, return_exceptions=True)
                raise

        if pending:
            await asyncio.gather(*pending, return_exceptions=True)

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
        frame_state = {"latest_frame": None}
        frame_lock = asyncio.Lock()
        stop_processing = asyncio.Event()
        receiver = asyncio.create_task(
            self._receiver_task_loop(
                track=track,
                session_id=session_id,
                frame_state=frame_state,
                frame_lock=frame_lock,
                stop_processing=stop_processing,
            )
        )
        processor = asyncio.create_task(
            self._processor_task_loop(
                session_id=session_id,
                detection_fps=detection_fps,
                enable_crop=enable_crop,
                yolo_detector=yolo_detector,
                frame_state=frame_state,
                frame_lock=frame_lock,
                stop_processing=stop_processing,
            )
        )
        
        try:
            await self._await_detection_tasks(receiver, processor, session_id)
        except asyncio.CancelledError:
            stop_processing.set()
            await self._cancel_detection_tasks(receiver, processor)
        except Exception as e:
            logger.error(f"[Detection] Pipeline error | session_id={session_id} error={e}")
            stop_processing.set()
            await self._cancel_detection_tasks(receiver, processor)
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
                "face": face_analysis or {"alerts": [], "faces": [], "face_count": 0, "person_count": 0},
                "yolo": {
                    "detection_count": frame_dict["detection_count"],
                    "detections": frame_dict["detections"],
                },
                "crop_offset": frame_dict.get("crop_offset", {}),
            }

            logger.debug(
                f"[Send] Detection frame geometry | session_id={session_id} frame_id={frame_id} "
                f"source_width={payload['crop_offset'].get('original_width', 0)} "
                f"source_height={payload['crop_offset'].get('original_height', 0)} "
                f"x_offset={payload['crop_offset'].get('x_offset', 0)} "
                f"y_offset={payload['crop_offset'].get('y_offset', 0)} "
                f"detections={payload['yolo']['detection_count']}"
            )

            current_view = (face_analysis or {}).get("current_view")
            if current_view is not None:
                logger.debug(
                    f"[Send] Current view payload | session_id={session_id} frame_id={frame_id} "
                    f"min_x={current_view['minX']:.2f} max_x={current_view['maxX']:.2f} "
                    f"min_y={current_view['minY']:.2f} max_y={current_view['maxY']:.2f}"
                )

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

        config = RTCConfiguration(
            iceServers=[
                RTCIceServer(urls="stun:stun.l.google.com:19302"),
                RTCIceServer(
                    urls=[
                        f"{settings.TURN_SERVER}?transport=udp",
                        f"{settings.TURN_SERVER}?transport=tcp",
                    ],
                    username=settings.TURN_USERNAME,
                    credential=settings.TURN_PASSWORD,
                ),
            ]
        )
   
        session_id = generate_session_id()
        self.face_analyzers[session_id] = FaceAnalyzer()
        pc = RTCPeerConnection(configuration=config)
        PEER_CONNECTIONS[session_id] = pc
        session_monitoring_store.register_session(session_id, webrtc_status="connected")

        self._setup_track_handlers(pc, session_id)
        self._setup_datachannel_handler(pc, session_id)

        offer = RTCSessionDescription(sdp=sdp, type=type)
        await pc.setRemoteDescription(offer)
        
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)
        
        while pc.iceGatheringState != "complete":
            await asyncio.sleep(0.1)

        logger.debug(f"[WebRTC] Offer handled | session_id={session_id} connection_state={pc.connectionState} ice_gathering_state={pc.iceGatheringState}") 

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
        session_monitoring_store.mark_session_status(session_id, "closed")
