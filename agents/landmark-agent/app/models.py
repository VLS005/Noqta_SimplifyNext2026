from typing import Literal

from pydantic import BaseModel


class LandmarkSetRequest(BaseModel):
    session_id: str
    description: str


class LandmarkSetResponse(BaseModel):
    status: str = "ok"
    session_id: str


class LandmarkScanRequest(BaseModel):
    session_id: str
    image_base64: str
    image_media_type: str = "image/jpeg"
    # DEMO_MODE only: force a canned found / not_found result. Ignored live.
    demo_result: str | None = None


class LandmarkScanResponse(BaseModel):
    visible: bool
    guidance: str
    confidence: str


class LandmarkClearRequest(BaseModel):
    session_id: str


class LandmarkClearResponse(BaseModel):
    status: str = "cleared"


class VisionVerdict(BaseModel):
    visible: bool
    position: Literal["left", "center", "right"] | None = None
    distance: Literal["near", "far"] | None = None
    confidence: Literal["low", "medium", "high"] = "low"
    raw_text: str = ""
