import asyncio
import numpy as np
from typing import Dict
from fastapi import WebSocket

from aiortc import RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaBlackhole

from app.modules.signaling.interfaces import BaseConnectionManager
from app.api.utils.utils import generate_session_id
from app.common.exceptions import InvalidMessageError
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

    async def handle_offer(self, sdp: str, type: str):
        session_id = generate_session_id()
        pc = RTCPeerConnection()

        try:
            logger.info(f"[WebRTC] Creating new peer connection | session_id={session_id}")

            PEER_CONNECTIONS[session_id] = pc

            # ---------------- TRACK HANDLING ----------------
            @pc.on("track")
            def on_track(track):
                logger.info(
                    f"[WebRTC] Track received | kind={track.kind} | session_id={session_id}"
                )

                # ---------------- VIDEO ----------------
                if track.kind == "video":

                    async def process_video():
                        logger.info(f"[WebRTC] Start video processing | session_id={session_id}")

                        while True:
                            try:
                                frame = await track.recv()

                                logger.debug(
                                    f"[VideoFrame] pts={frame.pts}, size={frame.width}x{frame.height}"
                                )

                                # TODO: Send to CV pipeline
                                # img = frame.to_ndarray(format="bgr24")

                            except Exception as e:
                                logger.error(f"[WebRTC] Video frame error: {e}")
                                break

                    asyncio.create_task(process_video())

                # ---------------- AUDIO ----------------
                elif track.kind == "audio":

                    async def process_audio():
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

                    asyncio.create_task(process_audio())

                # ---------------- TRACK END ----------------
                @track.on("ended")
                async def on_ended():
                    logger.info(
                        f"[WebRTC] Track ended | kind={track.kind} | session_id={session_id}"
                    )

            # ---------------- DATA CHANNEL ----------------
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

            # ---------------- SDP FLOW ----------------
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

        except Exception as e:
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
            except Exception as e:
                logger.error(
                    f"[WebRTC] Error closing peer connection | session_id={session_id} | error={str(e)}"
                )