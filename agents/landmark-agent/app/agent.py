"""Core Landmark Verification Agent: session landmark store + scan."""

from __future__ import annotations

import logging
import threading

from fastapi import HTTPException

from app.config import Settings
from app.models import LandmarkScanResponse, VisionVerdict
from app.tools.vision import VisionTool, guidance_from

logger = logging.getLogger(__name__)

_DEMO_FOUND = VisionVerdict(
    visible=True,
    position="left",
    distance="near",
    confidence="high",
    raw_text="demo",
)
_DEMO_NOT_FOUND = VisionVerdict(
    visible=False,
    position=None,
    distance=None,
    confidence="medium",
    raw_text="demo",
)


class LandmarkAgent:
    def __init__(self, settings: Settings, vision: VisionTool | None = None):
        self.settings = settings
        self.vision = vision or VisionTool(settings)
        self._landmarks: dict[str, str] = {}
        self._demo_toggles: dict[str, int] = {}
        self._lock = threading.Lock()

    def set_landmark(self, session_id: str, description: str) -> None:
        with self._lock:
            self._landmarks[session_id] = description
            self._demo_toggles[session_id] = 0

    def clear_landmark(self, session_id: str) -> bool:
        with self._lock:
            self._demo_toggles.pop(session_id, None)
            return self._landmarks.pop(session_id, None) is not None

    def get_landmark(self, session_id: str) -> str | None:
        with self._lock:
            return self._landmarks.get(session_id)

    def scan(
        self,
        session_id: str,
        image_base64: str,
        image_media_type: str,
        demo_result: str | None = None,
    ) -> LandmarkScanResponse:
        description = self.get_landmark(session_id)
        if description is None:
            raise HTTPException(
                status_code=404,
                detail="No landmark set for this session. Call POST /landmark/set first.",
            )

        if self.settings.demo_mode:
            verdict = self._demo_verdict(session_id, demo_result)
        else:
            verdict = self.vision.inspect(description, image_base64, image_media_type)

        return LandmarkScanResponse(
            visible=verdict.visible,
            guidance=guidance_from(verdict),
            confidence=verdict.confidence,
        )

    def _demo_verdict(self, session_id: str, demo_result: str | None) -> VisionVerdict:
        if demo_result == "found":
            return _DEMO_FOUND
        if demo_result == "not_found":
            return _DEMO_NOT_FOUND
        with self._lock:
            n = self._demo_toggles.get(session_id, 0)
            self._demo_toggles[session_id] = n + 1
        # Alternate: first scan not found, then found, then not found, ...
        return _DEMO_NOT_FOUND if n % 2 == 0 else _DEMO_FOUND
