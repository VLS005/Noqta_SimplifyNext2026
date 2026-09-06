"""
Converts standard navigation instructions into concise, blind-friendly micro-instructions.
Uses Bedrock (Claude) to ensure sensory and tactile cues are included.
"""
from __future__ import annotations

import json
import logging

from backend.routing_agent.models import RoutePlan, MobilityProfile
from backend.routing_agent.bedrock_client import BedrockClient
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are a navigation assistant for blind and low-vision users.
Your task is to translate standard navigation instructions into concise micro-instructions.
The route will contain both [WALKING] and [TRANSIT] segments.

Rules:
1. For [WALKING] segments, apply blind-friendly micro-instructions. Use sensory cues: tactile (feel the kerb), auditory (listen for traffic), proximity (around 15 steps ahead).
2. For [TRANSIT] segments (e.g. taking a bus/subway), KEEP the instructions exactly as provided but format them nicely. Do not add tactile or auditory walking cues since the user is in a vehicle.
3. Each [WALKING] instruction must be 15 words or fewer.
4. Never use cardinal directions (north, south, east, west, NE, SW, etc.) or measured distance (ex: 3 metres, 5 metres).
5. Reference physical landmarks the user can touch or hear.
6. Write for text-to-speech — no symbols, no abbreviations, full words only.
7. Return ONLY a JSON array of strings, one string per instruction. No other text.

Example input:
[WALKING] Walk straight
[WALKING] Turn right at the crosswalk
[TRANSIT] Take bus 10 towards City Center for 5 stops

Example output:
["Walk straight until you feel the raised tactile dots underfoot.", "Take a right turn in around 15 steps. Keep close to the wall on right to minimise injuries.", "Take bus 10 towards City Center for 5 stops."]
"""

_VISION_SYSTEM_PROMPT = """\
You are an accessibility safety assistant for a blind user. 
You will receive a raw, dynamically generated description of objects detected by the user's camera.
Your task is to translate this raw description into a single, concise micro-instruction (under 15 words) suitable for real-time Text-To-Speech (TTS).

Rules for Prioritization (Highest to Lowest):
1. Overhead obstructions or major risks (e.g., branches, poles, trees, buildings).
2. Small obstacles that are easy to trip over or miss.
3. Standard obstacles like chairs, tables, or people.

Return ONLY a plain text string with the safety instruction. Do not wrap in JSON. Do not include any extra commentary.
"""


class InstructionParsingAgent:
    """
    Converts raw navigation instruction strings into blind-friendly micro-instructions
    via a Bedrock (Claude) call. Falls back to lightly-cleaned versions of the
    raw instructions if Bedrock is unavailable.
    """

    def __init__(self, bedrock: BedrockClient) -> None:
        self._bedrock = bedrock

    async def translate(
        self,
        route: RoutePlan,
        profile: MobilityProfile,
    ) -> RoutePlan:
        """
        Translate raw_instructions for a RoutePlan and return an updated
        RoutePlan with micro_instructions populated.
        """
        if not route.raw_instructions:
            logger.warning(
                "InstructionParsingAgent: no raw_instructions to translate",
                extra={"route_id": route.id},
            )
            return route

        user_prompt = self._build_user_prompt(route, profile)
        raw_response = await self._bedrock.invoke_model(_SYSTEM_PROMPT, user_prompt)
        micro = self._parse_response(raw_response, route.raw_instructions)

        aligned = []
        for i, raw_instr in enumerate(route.raw_instructions):
            if raw_instr.startswith("[TRANSIT]"):
                aligned.append(raw_instr)
            else:
                aligned.append(micro[i] if i < len(micro) else raw_instr)
        micro = aligned
        print(
            f"[InstructionParsingAgent] Route '{route.label}' | "
            f"raw_instructions={len(route.raw_instructions)} | "
            f"micro_instructions={len(micro)}"
        )
        for i, instr in enumerate(micro):
            print(f"[InstructionParsingAgent]   [{i+1}] {instr}")

        logger.info(
            "InstructionParsingAgent: translated",
            extra={"route_id": route.id, "count": len(micro)},
        )
        return route.model_copy(update={"micro_instructions": micro})

    async def translate_all(
        self, routes: list[RoutePlan], profile: MobilityProfile
    ) -> list[RoutePlan]:
        """Translate micro-instructions for all candidate routes."""
        import asyncio
        return list(await asyncio.gather(*(self.translate(r, profile) for r in routes)))

    async def parse_vision_instruction(self, raw_guidance_text: str) -> str:
        """
        Translates raw geometric guidance text (from VisionAgent) into 
        a concise TTS safety command using Bedrock.
        """
        user_prompt = f"Raw camera detection data:\n{raw_guidance_text}\n\nGenerate the safety instruction."
        try:
            response = await self._bedrock.invoke_model(_VISION_SYSTEM_PROMPT, user_prompt)
            # Clean up quotes if LLM wraps the text
            return response.strip(' "\'')
        except Exception as e:
            logger.error(f"InstructionParsingAgent: vision parse failed: {e}")
            return raw_guidance_text

    # ─── Private helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _build_user_prompt(route: RoutePlan, profile: MobilityProfile) -> str:
        verbosity_note = {
            "brief": "Use very brief, minimal instructions — essential cues only.",
            "standard": "Use clear, helpful instructions with key sensory cues.",
            "detailed": "Use detailed instructions with all available sensory and tactile cues.",
        }.get(profile.instruction_verbosity, "")

        instructions_text = "\n".join(
            f"{i + 1}. {instr}" for i, instr in enumerate(route.raw_instructions)
        )
        waypoint_labels = "\n".join(
            f"  Waypoint {wp.sequence}: {wp.label}"
            + (" [fine motor — user will need vision assistance here]" if wp.fine_motor_required else "")
            for wp in route.waypoints
        )

        return (
            f"Route: {route.origin_label} → {route.destination_label}\n"
            f"User vision level: {profile.vision_level}\n"
            f"Verbosity preference: {profile.instruction_verbosity}. {verbosity_note}\n\n"
            f"Waypoints:\n{waypoint_labels}\n\n"
            f"Raw instructions to translate:\n{instructions_text}\n\n"
            "Translate these into blind-friendly micro-instructions. "
            "Return a JSON array of strings."
        )

    @staticmethod
    def _parse_response(raw: str, fallback: list[str]) -> list[str]:
        """
        Parse the Bedrock JSON response. Falls back to minimally-cleaned
        raw instructions if parsing fails.
        """
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list) and all(isinstance(s, str) for s in parsed):
                return [s.strip() for s in parsed if s.strip()]
        except (json.JSONDecodeError, TypeError):
            pass

        # Try to extract JSON array from within a larger response
        try:
            start = raw.index("[")
            end = raw.rindex("]") + 1
            parsed = json.loads(raw[start:end])
            if isinstance(parsed, list):
                return [s.strip() for s in parsed if isinstance(s, str) and s.strip()]
        except (ValueError, json.JSONDecodeError):
            pass

        logger.warning("InstructionParsingAgent: could not parse Bedrock response — using raw fallback")
        # Minimal fallback: strip cardinal directions from raw instructions
        cardinal_words = {"north", "south", "east", "west", "northeast", "northwest", "southeast", "southwest"}
        cleaned = []
        for instr in fallback:
            words = instr.split()
            filtered = [w for w in words if w.lower() not in cardinal_words]
            cleaned.append(" ".join(filtered))
        return cleaned
