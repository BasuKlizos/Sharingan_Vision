from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import JSONResponse

from app.api.utils.precheck_utils import get_lighting_precheck_service
from app.logger import logger
from app.modules.precheck.schemas import LightingPrecheckRequest, LightingPrecheckResponse
from app.modules.precheck.service import LightingPrecheckService

router = APIRouter()


@router.post("/lighting", response_model=LightingPrecheckResponse)
async def lighting_precheck(
    payload: LightingPrecheckRequest,
    service: Annotated[LightingPrecheckService, Depends(get_lighting_precheck_service)],
):
    try:
        logger.info(f"[API] Incoming lighting precheck | frames={len(payload.frames)}")
        response = service.evaluate_frames(payload.frames)
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content=response.model_dump(),
        )
    except Exception:
        logger.exception("[API] Lighting precheck failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error",
        )
