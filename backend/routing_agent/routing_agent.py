"""
Central orchestrator for the Routing Agent — imports and coordinates all sub-agents.
- This is the ONLY file the API entry point (main.py) needs to interact with.
"""
from __future__ import annotations

import logging

from backend.routing_agent.communication_agent import PersonalizationAdapter, ObstructionAdapter
from backend.routing_agent.vision_agent import VisionAgent, ProximityGateError
from backend.routing_agent.models import (
    LandmarkConfirmationMessage,
    LockedRouteNotificationMessage,
    ObstructionReportMessage,
    PaceUpdateMessage,
    RouteChoicePromptMessage,
    RouteOption,
    RouteRequestMessage,
    StartTimerEventMessage,
    VisionRequestMessage,
    VoiceRouteSelectionMessage,
    LockedRoute,
    RoutePlan,
    RouteSelectionState,
    MobilityProfile,
    UserSession,
)
from backend.routing_agent.direction_finder_agent import DirectionFinderAgent
from backend.routing_agent.accessibility_agent import AccessibilityAgent
from backend.routing_agent.eta_agent import ETAAgent
from backend.routing_agent.instruction_parsing_agent import InstructionParsingAgent
from backend.routing_agent.voice_agent import VoiceAgent, VoiceParseError
from backend.routing_agent.route_lock import RouteLockService, RouteLockError
from backend.routing_agent.bedrock_client import BedrockClient
from backend.shared.vector_store import VectorStore, get_vector_store

logger = logging.getLogger(__name__)


class RoutingAgentError(Exception):
    """Base exception for RoutingAgent errors."""


class SessionNotFoundError(RoutingAgentError):
    """Raised when a session_id is not found in the store."""


class RoutingAgent:
    """
    Central orchestrator — wires all sub-agents and exposes async methods
    that the API entry point (main.py) calls.
    """

    def __init__(
        self,
        store: VectorStore | None = None,
        bedrock: BedrockClient | None = None,
    ) -> None:
        self._store = store or get_vector_store()
        self._bedrock = bedrock or BedrockClient()

        # Sub-agents
        self._direction_finder = DirectionFinderAgent(self._bedrock)
        self._accessibility = AccessibilityAgent()
        self._eta = ETAAgent(store=self._store)
        self._instruction_parser = InstructionParsingAgent(self._bedrock)
        self._voice = VoiceAgent(self._bedrock)
        self._lock_svc = RouteLockService(self._store)

        # Communication adapters
        self._personalization = PersonalizationAdapter()
        self._obstruction = ObstructionAdapter()
        self._vision = VisionAgent()

        logger.info("RoutingAgent: initialised with all sub-agents")

    # ══════════════════════════════════════════════════════════════════════════
    # SESSION MANAGEMENT
    # ══════════════════════════════════════════════════════════════════════════

    async def create_session(self, user_id: str, profile: MobilityProfile | None = None) -> UserSession:
        """Create and persist a new navigation session for a user."""
        session = UserSession(
            user_id=user_id,
            profile=profile or MobilityProfile(),
        )
        pk, sk = VectorStore.session_keys(user_id, session.session_id)
        await self._store.put(pk, sk, {
            "session_json": session.model_dump_json(),
            "user_id": user_id,
        })

        # Seed a test pace entry in the store (0.8 m/s default)
        # This ensures ETAAgent can look up pace even in development
        pace_pk, pace_sk = f"PACE#{user_id}", "PACE#current"
        existing_pace = await self._store.get(pace_pk, pace_sk)
        if not existing_pace:
            default_pace = profile.preferred_pace_mps if profile else 0.8
            await self._store.put(pace_pk, pace_sk, {
                "average_pace_mps": default_pace,
                "user_id": user_id,
                "source": "seed",
            })
            print(f"[RoutingAgent] Seeded test pace entry for user '{user_id}': {default_pace} m/s")

        logger.info("RoutingAgent: session created", extra={"session_id": session.session_id, "user_id": user_id})
        print(f"[RoutingAgent] Session created: {session.session_id} for user '{user_id}'")
        return session

    async def get_session(self, session_id: str, user_id: str) -> UserSession:
        """Retrieve an active session. Raises SessionNotFoundError if not found."""
        pk, sk = VectorStore.session_keys(user_id, session_id)
        item = await self._store.get(pk, sk)
        if not item or "session_json" not in item:
            raise SessionNotFoundError(f"Session '{session_id}' not found for user '{user_id}'")
        return UserSession.model_validate_json(item["session_json"])

    # ══════════════════════════════════════════════════════════════════════════
    # CORE ROUTING PIPELINE
    # ══════════════════════════════════════════════════════════════════════════

    async def handle_route_request(self, msg: RouteRequestMessage) -> RouteChoicePromptMessage:
        """
        Full route-planning pipeline:
        1. Get/create session
        2. DirectionFinderAgent → candidate routes (Google Routes + OSM APIs)
        3. AccessibilityAgent → score and rank
        4. ETAAgent → personalised ETA (DynamoDB pace lookup)
        5. InstructionParsingAgent → blind-friendly micro-instructions
        6. RouteLockService → persist candidates, transition state
        7. Build and return RouteChoicePromptMessage
        """
        print(f"\n{'='*70}")
        print(f"[RoutingAgent] === ROUTE REQUEST PIPELINE START ===")
        print(f"{'='*70}")
        logger.info("RoutingAgent: handling route request", extra={"session_id": msg.session_id})

        # Step 1 — session
        try:
            session = await self.get_session(msg.session_id, msg.user_id)
        except SessionNotFoundError:
            session = await self.create_session(msg.user_id)
        session.touch()
        profile = session.profile
        print(f"[RoutingAgent] Step 1 — Session: {session.session_id} | User: {msg.user_id}")
        print(f"[RoutingAgent] Profile: vision={profile.vision_level}, pace={profile.preferred_pace_mps} m/s, step_free={profile.prefers_step_free}")

        # Step 2 — generate candidates via REAL APIs (Google Routes + OSM)
        print(f"\n[RoutingAgent] Step 2 — Calling DirectionFinderAgent (Google Routes + OSM)...")
        candidates = await self._direction_finder.generate_candidates(
            session_id=session.session_id,
            origin_label=msg.origin_label,
            destination_label=msg.destination_label,
            profile=profile,
            origin_lat=msg.origin_lat,
            origin_lon=msg.origin_lon,
            destination_lat=msg.destination_lat,
            destination_lon=msg.destination_lon,
        )
        print(f"[RoutingAgent] Step 2 — DirectionFinderAgent returned {len(candidates)} candidates")

        # Step 3 — accessibility scoring (dynamic weights from profile)
        print(f"\n[RoutingAgent] Step 3 — Calling AccessibilityAgent (dynamic weights)...")
        candidates = self._accessibility.score_all(candidates, profile)
        print(f"[RoutingAgent] Step 3 — AccessibilityAgent scored and ranked {len(candidates)} candidates")

        # Step 4 — personalised ETA (with DynamoDB pace lookup)
        print(f"\n[RoutingAgent] Step 4 — Calling ETAAgent (DynamoDB pace lookup)...")
        candidates = await self._eta.estimate_all(candidates, profile, user_id=msg.user_id)
        print(f"[RoutingAgent] Step 4 — ETAAgent computed ETA for {len(candidates)} candidates")

        # Step 5 — instruction translation
        print(f"\n[RoutingAgent] Step 5 — Calling InstructionParsingAgent (Gemini)...")
        candidates = await self._instruction_parser.translate_all(candidates, profile)
        print(f"[RoutingAgent] Step 5 — InstructionParsingAgent translated {len(candidates)} candidates")

        # Step 6 — persist + state transition
        await self._lock_svc.present_candidates(session.session_id, candidates)
        print(f"[RoutingAgent] Step 6 — Candidates persisted, state → CANDIDATES_PRESENTED")

        # Step 7 — build prompt
        prompt_msg = self._build_route_choice_prompt(session.session_id, candidates)
        print(f"\n[RoutingAgent] Step 7 — Route choice prompt built")
        print(f"[RoutingAgent] Prompt text: {prompt_msg.prompt_text}")
        print(f"{'='*70}")
        print(f"[RoutingAgent] === ROUTE REQUEST PIPELINE COMPLETE ===")
        print(f"{'='*70}\n")

        logger.info("RoutingAgent: route choice prompt built", extra={"options": len(candidates)})
        return prompt_msg

    async def handle_voice_selection(
        self, msg: VoiceRouteSelectionMessage
    ) -> tuple[LockedRoute, LockedRouteNotificationMessage]:
        """
        Process voice route selection:
        1. VoiceAgent resolves transcript → route_id
        2. RouteLockService locks the route
        3. PersonalizationAdapter dispatches StartTimerEvent
        4. Returns (LockedRoute, notification for UI)
        """
        logger.info("RoutingAgent: handling voice selection", extra={"transcript": msg.raw_voice_transcript})

        candidates = await self._lock_svc.get_candidates(msg.session_id)
        if not candidates:
            raise RouteLockError(f"No candidates found for session '{msg.session_id}'. Request a route first.")

        resolved_msg = await self._voice.resolve(msg, candidates)
        route_id = resolved_msg.resolved_route_id

        locked = await self._lock_svc.lock(msg.session_id, route_id)
        logger.info("RoutingAgent: route locked", extra={"route_id": route_id, "label": locked.route_plan.label})

        await self._dispatch_start_timer(locked)

        first_instruction = (
            locked.route_plan.micro_instructions[0]
            if locked.route_plan.micro_instructions
            else locked.route_plan.raw_instructions[0]
            if locked.route_plan.raw_instructions
            else "Proceed to the starting point."
        )
        notification = LockedRouteNotificationMessage(
            session_id=msg.session_id,
            locked_route_id=locked.route_plan.id,
            first_micro_instruction=first_instruction,
            total_waypoints=len(locked.route_plan.waypoints),
            estimated_duration_s=locked.route_plan.estimated_duration_s,
        )
        return locked, notification

    # ══════════════════════════════════════════════════════════════════════════
    # INBOUND MESSAGE DISPATCHER
    # ══════════════════════════════════════════════════════════════════════════

    async def handle_inbound(self, agent_type: str, raw: dict) -> dict:
        """Route inbound messages from external agents to the correct handler."""
        handlers = {
            "obstruction": self._handle_obstruction,
            "pace": self._handle_pace_update,
            "landmark": self._handle_landmark_confirmation,
            "vision_response": self._handle_vision_response,
        }
        handler = handlers.get(agent_type)
        if handler is None:
            logger.warning("RoutingAgent: unknown agent_type", extra={"agent_type": agent_type})
            return {"status": "rejected", "reason": f"Unknown agent type: {agent_type}"}
        return await handler(raw)

    async def _handle_obstruction(self, raw: dict) -> dict:
        msg = self._obstruction.parse(raw)
        return {"status": "received", "message_id": msg.message_id, "action": "no-op"}

    async def _handle_pace_update(self, raw: dict) -> dict:
        msg = PaceUpdateMessage.model_validate(raw)
        logger.info("RoutingAgent: pace update (no-op)", extra={"pace_mps": msg.current_pace_mps})
        return {"status": "received", "message_id": msg.message_id, "action": "no-op"}

    async def _handle_landmark_confirmation(self, raw: dict) -> dict:
        msg = LandmarkConfirmationMessage.model_validate(raw)
        logger.info("RoutingAgent: landmark confirmation (no-op)", extra={"confirmed": msg.confirmed})
        return {"status": "received", "message_id": msg.message_id, "action": "no-op"}

    async def _handle_vision_response(self, raw: dict) -> dict:
        msg = self._vision.parse(raw)
        return {"status": "received", "message_id": msg.message_id, "guidance_text": msg.guidance_text}

    # ══════════════════════════════════════════════════════════════════════════
    # ON-DEMAND VISION (5 m gate)
    # ══════════════════════════════════════════════════════════════════════════

    async def request_vision_assist(
        self, session_id: str, waypoint_id: str, user_distance_m: float, context: str,
    ) -> dict:
        """Dispatch on-demand vision request. Enforces ≤ 5 m proximity gate."""
        from pydantic import ValidationError as PydanticValidationError

        settings_gate = 5.0
        if user_distance_m > settings_gate:
            reason = (
                f"Vision request rejected: user is {user_distance_m:.1f} m from waypoint "
                f"(gate is {settings_gate} m). Vision only allowed within {settings_gate} m."
            )
            return {"status": "gate_rejected", "reason": reason}

        try:
            request = VisionRequestMessage(
                session_id=session_id, waypoint_id=waypoint_id,
                user_distance_m=user_distance_m, context=context,
            )
        except PydanticValidationError as exc:
            return {"status": "gate_rejected", "reason": str(exc)}

        try:
            dispatched = await self._vision.dispatch_request(request)
            return {"status": "dispatched" if dispatched else "unreachable", "message_id": request.message_id}
        except ProximityGateError as exc:
            return {"status": "gate_rejected", "reason": str(exc)}

    # ══════════════════════════════════════════════════════════════════════════
    # QUERY
    # ══════════════════════════════════════════════════════════════════════════

    async def get_locked_route(self, session_id: str) -> LockedRoute | None:
        return await self._lock_svc.get_locked_route(session_id)

    async def get_route_state(self, session_id: str) -> RouteSelectionState:
        return await self._lock_svc.get_state(session_id)

    # ══════════════════════════════════════════════════════════════════════════
    # PRIVATE HELPERS
    # ══════════════════════════════════════════════════════════════════════════

    @staticmethod
    def _build_route_choice_prompt(session_id: str, candidates: list[RoutePlan]) -> RouteChoicePromptMessage:
        options = []
        for i, route in enumerate(candidates, start=1):
            highlights = []
            if route.accessibility_score.step_free:
                highlights.append("Step-free throughout")
            if route.accessibility_score.tactile_paving:
                highlights.append("Tactile paving")
            if route.accessibility_score.crowding_estimate < 0.3:
                highlights.append("Low footfall")

            first_instr = (
                route.micro_instructions[0] if route.micro_instructions
                else route.raw_instructions[0] if route.raw_instructions
                else "Follow the path ahead."
            )
            options.append(RouteOption(
                route_id=route.id, index=i, label=route.label,
                rank=route.rank,
                duration_s=route.estimated_duration_s, distance_m=route.distance_m,
                accessibility_composite=route.accessibility_score.composite_score,
                first_micro_instruction=first_instr, highlights=highlights,
            ))

        parts = [f"I found {len(options)} routes."]
        for opt in options:
            duration_text = ETAAgent.format_duration(opt.duration_s)
            highlight_text = f"  {', '.join(opt.highlights)}." if opt.highlights else ""
            parts.append(
                f"Route {opt.index} (rank {opt.rank}): {opt.label}. "
                f"{duration_text}.{highlight_text}"
            )
        parts.append("Say route 1, route 2, or route 3 to choose. Or say 'accessible', 'shorter', or 'quieter'.")
        prompt_text = " ".join(parts)

        return RouteChoicePromptMessage(
            session_id=session_id, route_options=options, prompt_text=prompt_text,
        )

    async def _dispatch_start_timer(self, locked: LockedRoute) -> None:
        msg = StartTimerEventMessage(
            session_id=locked.session_id,
            user_id=locked.route_plan.session_id,
            locked_route_id=locked.route_plan.id,
            estimated_duration_s=locked.route_plan.estimated_duration_s,
            destination_label=locked.route_plan.destination_label,
            route_distance_m=locked.route_plan.distance_m,
        )
        await self._personalization.send(msg)
