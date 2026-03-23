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