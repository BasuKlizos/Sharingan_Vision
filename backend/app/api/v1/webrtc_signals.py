from typing import Annotated

from fastapi import APIRouter, HTTPException, status, Depends
from fastapi.responses import JSONResponse

from app.modules.signaling.schemas import OfferRequest, AnswerResponse
from app.common.exceptions import InvalidMessageError
from app.api.utils.webrtc_signals_utils import get_webrtc_service
from app.logger import logger
from app.modules.signaling.service import WebRTCService

router = APIRouter()

@router.post("/webrtc/offer", response_model=AnswerResponse)
async def handle_offer(
    payload: OfferRequest,
    service: Annotated[WebRTCService, Depends(get_webrtc_service)],
):
    """Process WebRTC SDP offer and return SDP answer with session_id."""
    try:
        logger.info(f"[API] Incoming WebRTC offer | type={payload.type}")

        response = await service.handle_offer(
            payload.sdp,
            payload.type
        )

        logger.info(
            f"[API] Offer processed | session_id={response.get('session_id')}"
        )

        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content=response
        )

    except InvalidMessageError as e:
        logger.warning(f"[API] Invalid offer | error={str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )

    except Exception:
        logger.exception("[API] WebRTC offer failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )