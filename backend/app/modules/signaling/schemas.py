from typing import Any, Dict, Literal, Optional
from pydantic import BaseModel, Field


class SignalMessage(BaseModel):
    type: Literal[
        "join",
        "ready",
        "offer",
        "answer",
        "ice-candidate",
        "leave",
        "error",
    ]
    data: Dict[str, Any] = Field(default_factory=dict)
    sender_id: Optional[str] = None
    role: Optional[Literal["candidate", "interviewer"]] = None


class OfferRequest(BaseModel):
    sdp: str
    type: str


class AnswerResponse(BaseModel):
    sdp: str
    type: str
    session_id: str