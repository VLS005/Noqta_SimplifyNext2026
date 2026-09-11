"""Bedrock Converse vision call with a triggered still image.

Never streams video. boto3 clients are built via make_aws_client() so Bedrock
and a later STS check share the same credential kwargs from Settings — never
silently fall through to ~/.aws when hackathon keys are in .env.
"""

from __future__ import annotations

import base64
import json
import logging
import re

import boto3

from app.config import Settings
from app.models import VisionVerdict

logger = logging.getLogger(__name__)

_PROMPT = """You are helping a blind traveller find a landmark.

Landmark to find: {description}

Look at the snapshot. Return ONLY JSON with this exact shape:
{{
  "visible": true or false,
  "position": "left" or "center" or "right" or null,
  "distance": "near" or "far" or null,
  "confidence": "low" or "medium" or "high"
}}

If the landmark is not clearly visible, set visible to false and position to null.
Do not guess. Safety is more important than being helpful.
"""


def make_aws_client(settings: Settings, service_name: str, injected=None):
    """Build a boto3 client with Settings credentials when present.

    Injected clients (tests) are returned unchanged. If .env has no keys,
    boto3's default chain is used — that case is logged at startup /health.
    """
    if injected is not None:
        return injected
    kwargs: dict = {"region_name": settings.aws_region}
    kwargs.update(settings.boto3_credential_kwargs())
    return boto3.client(service_name, **kwargs)


def _image_format(media_type: str) -> str:
    subtype = (media_type or "image/jpeg").split("/")[-1].lower()
    return "jpeg" if subtype in {"jpg", "jpeg"} else subtype


def guidance_from(verdict: VisionVerdict) -> str:
    """Short spoken-instruction style line for voice/haptic output."""
    if not verdict.visible:
        return "Not visible yet, keep moving"
    position = verdict.position or "center"
    distance = verdict.distance or "near"
    if position == "left":
        return "Slightly to your left, keep walking"
    if position == "right":
        return "Slightly to your right, keep walking"
    if distance == "far":
        return "Straight ahead, keep walking"
    return "Straight ahead, about 10 steps"


def parse_verdict(text: str) -> VisionVerdict:
    """Parse model JSON. Malformed output → visible=false, never raise."""
    match = re.search(r"\{.*\}", text or "", flags=re.DOTALL)
    if not match:
        logger.warning("Bedrock vision response had no JSON: %s", text)
        return VisionVerdict(visible=False, confidence="low", raw_text=text or "")
    try:
        payload = json.loads(match.group(0))
        visible = bool(payload.get("visible"))
        position = payload.get("position")
        if position not in {"left", "center", "right"}:
            position = None
        distance = payload.get("distance")
        if distance not in {"near", "far"}:
            distance = None
        confidence = payload.get("confidence")
        if confidence not in {"low", "medium", "high"}:
            confidence = "low"
        return VisionVerdict(
            visible=visible,
            position=position,
            distance=distance,
            confidence=confidence,
            raw_text=text,
        )
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        logger.warning("Bedrock vision JSON parse failed (%s): %s", exc, text)
        return VisionVerdict(visible=False, confidence="low", raw_text=text or "")


class VisionTool:
    def __init__(self, settings: Settings, bedrock_client=None):
        self.settings = settings
        self._bedrock = bedrock_client

    def inspect(self, description: str, image_base64: str, media_type: str) -> VisionVerdict:
        raw = base64.b64decode(image_base64, validate=True)
        image_format = _image_format(media_type)
        client = make_aws_client(self.settings, "bedrock-runtime", self._bedrock)
        response = client.converse(
            modelId=self.settings.bedrock_vision_model_id,
            messages=[{
                "role": "user",
                "content": [
                    {"image": {"format": image_format, "source": {"bytes": raw}}},
                    {"text": _PROMPT.format(description=description)},
                ],
            }],
            inferenceConfig={"maxTokens": 300, "temperature": 0},
        )
        text = "".join(
            block.get("text", "")
            for block in response["output"]["message"]["content"]
        )
        logger.info("Bedrock vision raw output: %s", text)
        return parse_verdict(text)
