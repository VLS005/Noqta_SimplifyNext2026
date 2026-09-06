"""
Parses raw voice transcripts to resolve which route the user selected.
Acts as the voice input processing sub-agent.
"""
from __future__ import annotations

import json
import logging
import re
import boto3

from backend.routing_agent.models import VoiceRouteSelectionMessage, RoutePlan
from backend.routing_agent.bedrock_client import BedrockClient
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# ─── Keyword maps ─────────────────────────────────────────────────────────────
_INDEX_WORDS: dict[str, int] = {
    "1": 1, "one": 1, "first": 1,
    "2": 2, "two": 2, "second": 2,
    "3": 3, "three": 3, "third": 3,
}
_ATTRIBUTE_KEYWORDS: list[tuple[list[str], str]] = [
    (["step-free", "step free", "accessible", "accessibility", "no steps"], "step_free"),
    (["shorter", "short", "quick", "faster", "fast", "quicker"], "shortest"),
    (["quiet", "quieter", "quietest", "park", "peaceful", "less busy"], "quietest"),
]

_VOICE_SYSTEM_PROMPT = """\
You are a voice input parser for a navigation app for blind users.
The user has been presented with a list of route options and has spoken a selection.
Resolve which route number (1, 2, or 3) they are choosing based on their words.
Return ONLY a JSON object: {"resolved_index": <integer 1-3>, "confidence": <float 0-1>}
If you cannot determine the choice, return {"resolved_index": null, "confidence": 0}
"""


class VoiceParseError(Exception):
    """Raised when voice input cannot be resolved to a route."""


class VoiceAgent:
    """
    Resolves a raw voice transcript to a route_id from the candidate list.

    Resolution order (fastest to most expensive):
    1. Exact index match ("route 1", "the first one")
    2. Attribute keyword match ("the step-free one", "the shorter one")
    3. Bedrock Claude disambiguation
    """

    def __init__(self, bedrock: BedrockClient) -> None:
        self._bedrock = bedrock
        self._polly = boto3.client('polly', region_name='us-east-1')

    async def resolve(
        self,
        msg: VoiceRouteSelectionMessage,
        candidates: list[RoutePlan],
    ) -> VoiceRouteSelectionMessage:
        """
        Parse msg.raw_voice_transcript and return an updated message
        with resolved_route_id populated.
        Raises VoiceParseError if resolution fails completely.
        """
        transcript = msg.raw_voice_transcript.strip().lower()
        logger.info(
            "VoiceAgent: resolving transcript",
            extra={"session_id": msg.session_id, "transcript": transcript},
        )

        # Step 1 — exact index
        route_id = self._try_index_match(transcript, candidates)
        if route_id:
            logger.info("VoiceAgent: resolved via index match", extra={"route_id": route_id})
            return msg.model_copy(update={"resolved_route_id": route_id})

        # Step 2 — attribute keywords
        route_id = self._try_attribute_match(transcript, candidates)
        if route_id:
            logger.info("VoiceAgent: resolved via attribute match", extra={"route_id": route_id})
            return msg.model_copy(update={"resolved_route_id": route_id})

        # Step 3 — Bedrock disambiguation
        route_id = await self._try_bedrock_resolve(transcript, candidates)
        if route_id:
            logger.info("VoiceAgent: resolved via Bedrock", extra={"route_id": route_id})
            return msg.model_copy(update={"resolved_route_id": route_id})

        raise VoiceParseError(
            f"Could not resolve voice transcript to a route: '{msg.raw_voice_transcript}'"
        )

    # ─── TTS Generation (Amazon Polly) ────────────────────────────────────────

    def generate_tts_audio(self, text: str) -> tuple[bytes, list[dict]]:
        """
        Generates TTS audio and speech marks (for text highlighting) using Amazon Polly.
        Returns a tuple: (mp3_audio_bytes, list_of_speech_mark_dicts)
        """
        logger.info("VoiceAgent: generating TTS audio and speech marks via Polly")
        
        # Request speech marks (JSON)
        marks_response = self._polly.synthesize_speech(
            Text=text,
            OutputFormat='json',
            SpeechMarkTypes=['word'],
            VoiceId='Joanna'
        )
        marks_stream = marks_response.get('AudioStream')
        speech_marks = []
        if marks_stream:
            for line in marks_stream.read().decode('utf-8').split('\n'):
                if line.strip():
                    speech_marks.append(json.loads(line))
                    
        # Request actual audio (MP3)
        audio_response = self._polly.synthesize_speech(
            Text=text,
            OutputFormat='mp3',
            VoiceId='Joanna'
        )
        audio_bytes = audio_response.get('AudioStream').read() if 'AudioStream' in audio_response else b''
        
        return audio_bytes, speech_marks

    # ─── Resolution strategies ────────────────────────────────────────────────

    @staticmethod
    def _try_index_match(transcript: str, candidates: list[RoutePlan]) -> str | None:
        """Match 'route 1', 'the first one', 'option two', etc."""
        # Strip "route", "option", "number" prefixes
        cleaned = re.sub(r"\b(route|option|number|the)\b", "", transcript).strip()
        for token, idx in _INDEX_WORDS.items():
            if re.search(rf"\b{re.escape(token)}\b", cleaned):
                one_based = idx - 1
                if 0 <= one_based < len(candidates):
                    return candidates[one_based].id
        return None

    @staticmethod
    def _try_attribute_match(transcript: str, candidates: list[RoutePlan]) -> str | None:
        """Match semantic attributes to route labels/properties."""
        for keywords, attr in _ATTRIBUTE_KEYWORDS:
            if any(kw in transcript for kw in keywords):
                if attr == "step_free":
                    for c in candidates:
                        if c.accessibility_score.step_free:
                            return c.id
                elif attr == "shortest":
                    shortest = min(candidates, key=lambda r: r.distance_m)
                    return shortest.id
                elif attr == "quietest":
                    quietest = min(candidates, key=lambda r: r.accessibility_score.crowding_estimate)
                    return quietest.id
        return None

    async def _try_bedrock_resolve(
        self, transcript: str, candidates: list[RoutePlan]
    ) -> str | None:
        """Use Claude to disambiguate unclear voice input."""
        options_text = "\n".join(
            f"Route {i + 1}: {c.label} — {int(c.distance_m)}m, "
            f"{'step-free' if c.accessibility_score.step_free else 'has steps'}, "
            f"score {c.accessibility_score.composite_score:.2f}"
            for i, c in enumerate(candidates)
        )
        user_prompt = (
            f"The user said: '{transcript}'\n\n"
            f"Available routes:\n{options_text}\n\n"
            "Which route number (1, 2, or 3) did they choose?"
        )
        raw = await self._bedrock.invoke_model(_VOICE_SYSTEM_PROMPT, user_prompt)
        try:
            data = json.loads(raw)
            idx = data.get("resolved_index")
            if idx is not None and 1 <= idx <= len(candidates):
                return candidates[idx - 1].id
        except (json.JSONDecodeError, TypeError, KeyError):
            pass
        return None
