"""
Central orchestrator for the Routing Agent — imports and coordinates all sub-agents.
- This is the ONLY file the API entry point (main.py) needs to interact with.
"""
from __future__ import annotations

import logging
from dotenv import load_dotenv

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

load_dotenv()

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
        self._accessibility = AccessibilityAgent(self._bedrock)
        self._eta = ETAAgent(store=self._store)
        self._instruction_parser = InstructionParsingAgent(self._bedrock)
        self._voice = VoiceAgent(self._bedrock)
        self._lock_svc = RouteLockService(self._store)

        # Communication adapters
        self._personalization = PersonalizationAdapter()
        self._obstruction = ObstructionAdapter()
        self._vision = VisionAgent()

        logger.info("RoutingAgent: initialised with all sub-agents")

    # ─── Live Navigation Simulator ────────────────────────────────────────────

    async def _simulate_live_navigation(self, route_plan: RoutePlan):
        """Simulates walking a route by playing instructions sequentially."""
        import asyncio
        logger.info("RoutingAgent: Starting Live Navigation Simulation...")
        
        # Initialize pause event for interruption
        self._nav_pause_event = asyncio.Event()
        self._nav_pause_event.set()
        self._nav_player = None
        
        instructions = route_plan.micro_instructions if route_plan.micro_instructions else route_plan.raw_instructions
        
        for i, instruction in enumerate(instructions):
            # Wait if navigation is paused
            await self._nav_pause_event.wait()
            
            logger.info(f"RoutingAgent: [SIMULATOR] Step {i+1}/{len(instructions)}: {instruction}")
            audio_bytes, _ = self._voice.generate_tts_audio(instruction)
            if audio_bytes:
                with open("temp_nav_tts.mp3", "wb") as f:
                    f.write(audio_bytes)
                self._nav_player = await asyncio.create_subprocess_exec("afplay", "temp_nav_tts.mp3")
            
            # Wait 5 seconds to simulate walking to the next waypoint
            # Loop quickly so we can break early if interrupted
            for _ in range(50):
                await asyncio.sleep(0.1)
                if not self._nav_pause_event.is_set():
                    break
            
            # If we were paused during walking, wait until we resume before going to next step
            await self._nav_pause_event.wait()
            
        logger.info("RoutingAgent: Live Navigation Simulation Complete.")

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
        candidates = await self._accessibility.score_all(candidates, profile)
        print(f"[RoutingAgent] Step 3 — AccessibilityAgent scored and ranked {len(candidates)} candidates")

        # Step 4 — personalised ETA (with DynamoDB pace lookup)
        print(f"\n[RoutingAgent] Step 4 — Calling ETAAgent (DynamoDB pace lookup)...")
        candidates = await self._eta.estimate_all(candidates, profile, user_id=msg.user_id)
        print(f"[RoutingAgent] Step 4 — ETAAgent computed ETA for {len(candidates)} candidates")

        # Step 5 — instruction translation
        print(f"\n[RoutingAgent] Step 5 — Calling InstructionParsingAgent (Gemini)...")
        candidates = await self._instruction_parser.translate_all(candidates, profile)
        print(f"[RoutingAgent] Step 5 — InstructionParsingAgent translated {len(candidates)} candidates")

        # Step 6 — Auto-lock the highest ranked route (Route 1)
        top_route = candidates[0]
        second_route = candidates[1] if len(candidates) > 1 else None



        await self._lock_svc.present_candidates(session.session_id, candidates)
        locked = await self._lock_svc.lock(session.session_id, top_route.id)
        
        print(f"[RoutingAgent] Step 6 — Auto-locked Route '{locked.route_plan.label}'")

        # Step 7 — Dispatch start timer
        await self._dispatch_start_timer(locked)
        
        first_instruction = (
            locked.route_plan.micro_instructions[0]
            if locked.route_plan.micro_instructions
            else locked.route_plan.raw_instructions[0]
            if locked.route_plan.raw_instructions
            else "Proceed to the starting point."
        )
        notification = LockedRouteNotificationMessage(
            session_id=session.session_id,
            locked_route_id=locked.route_plan.id,
            first_micro_instruction=first_instruction,
            total_waypoints=len(locked.route_plan.waypoints),
            estimated_duration_s=locked.route_plan.estimated_duration_s,
        )

        print(f"{'='*70}")
        print(f"[RoutingAgent] === ROUTE REQUEST PIPELINE COMPLETE ===")
        print(f"{'='*70}\n")

        print(f"[RoutingAgent] CONSOLIDATED AGENT OUTPUTS:")
        for i, r in enumerate(candidates, 1):
            print(f"\n--- Route {i}: {r.label} ---")
            print(f"  [DirectionFinderAgent] Distance: {r.distance_m:.0f} m, Waypoints: {len(r.waypoints)}")
            print(f"  [AccessibilityAgent]   Score: {r.accessibility_score.composite_score:.4f}, Rank: {r.rank}")
            print(f"                         Evidence: StepFree={r.accessibility_score.step_free}, Tactile={r.accessibility_score.tactile_paving}, Lighting={r.accessibility_score.lighting_quality.value}, Crowding={r.accessibility_score.crowding_estimate}, ObstructionRisk={r.accessibility_score.obstruction_risk}")
            print(f"  [ETAAgent]             ETA: {ETAAgent.format_duration(r.estimated_duration_s)} ({r.estimated_duration_s}s)")
            print(f"  [InstructionParsing]   {len(r.micro_instructions)} micro-instructions generated:")
            for j, instr in enumerate(r.micro_instructions, 1):
                print(f"    {j}. {instr}")
        print(f"\n{'='*70}\n")

        # --- POST-PIPELINE AUDIT ---
        print(f"\n[RoutingAgent] POST-PIPELINE AUDIT: Prompting LLM for Audit...")
        audit_system_prompt = "You are an accessibility navigation agent. Audit the route options presented to the user. Explain the instruction, ETA, and accessibility evidence for each route."
        audit_info = ""
        for i, r in enumerate(candidates):
            audit_info += f"\nRoute {i+1}: {r.label} (score: {r.accessibility_score.composite_score:.4f}, ETA: {r.estimated_duration_s}s, instructions: {len(r.micro_instructions) or len(r.raw_instructions)})\n"
            audit_info += f"Accessibility Evidence: step_free={r.accessibility_score.step_free}, tactile={r.accessibility_score.tactile_paving}, lighting={r.accessibility_score.lighting_quality.value}, crowding={r.accessibility_score.crowding_estimate}, obstruction={r.accessibility_score.obstruction_risk}\n"
            
        audit_user_prompt = (
            f"User Profile: vision level = {profile.vision_level}, pace = {profile.preferred_pace_mps} m/s, "
            f"prefers step-free = {profile.prefers_step_free}, prefers tactile = {profile.prefers_tactile_paving}.\n"
            f"Route Options:\n{audit_info}\n"
            f"Please provide the audit for each route."
        )
        try:
            audit_explanation = await self._bedrock.invoke_model(audit_system_prompt, audit_user_prompt, model_tier="nova-pro")
            print(f"\n[RoutingAgent] POST-PIPELINE AUDIT RESULT:\n{audit_explanation}\n")
        except Exception as e:
            print(f"\n[RoutingAgent] POST-PIPELINE AUDIT FAILED: {e}\n")
        # --- END POST-PIPELINE AUDIT ---
        
        # Start background live navigation simulation
        import asyncio
        asyncio.create_task(self._simulate_live_navigation(locked.route_plan))

        logger.info("RoutingAgent: route auto-locked", extra={"locked_route_id": locked.route_plan.id})
        return locked, notification

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

        import asyncio
        asyncio.create_task(self._simulate_live_navigation(locked.route_plan))

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
        if msg.severity == "high":
            return {"status": "received", "message_id": msg.message_id, "action": "replan_required"}
        return {"status": "received", "message_id": msg.message_id, "action": "no-op"}

    async def _handle_pace_update(self, raw: dict) -> dict:
        msg = PaceUpdateMessage.model_validate(raw)
        logger.info("RoutingAgent: pace update", extra={"pace_mps": msg.current_pace_mps})
        pace_pk, pace_sk = f"PACE#{msg.user_id}", "PACE#current"
        await self._store.put(pace_pk, pace_sk, {
            "average_pace_mps": msg.current_pace_mps,
            "user_id": msg.user_id,
            "source": "dynamodb",
        })
        return {"status": "received", "message_id": msg.message_id, "action": "persisted"}

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

        async def on_obstruction(guidance_text: str):
            import asyncio
            logger.info("RoutingAgent: Received obstruction event from VisionAgent.")
            
            # 1. Pause navigation simulator
            if hasattr(self, '_nav_pause_event'):
                self._nav_pause_event.clear()
            
            # 2. Stop current navigation audio if playing
            if hasattr(self, '_nav_player') and self._nav_player:
                try:
                    self._nav_player.terminate()
                except ProcessLookupError:
                    pass
                    
            # 3. Play a ping/haptic alert immediately
            ping_proc = await asyncio.create_subprocess_exec("afplay", "/System/Library/Sounds/Glass.aiff")
            await ping_proc.wait()
            
            refined = await self._instruction_parser.parse_vision_instruction(guidance_text)
            logger.info(f"RoutingAgent: Refined Instruction -> {refined}")
            
            # Generate TTS
            audio_bytes, _ = self._voice.generate_tts_audio(refined)
            if audio_bytes:
                with open("temp_tts.mp3", "wb") as f:
                    f.write(audio_bytes)
                # Play audio locally for testing and wait for it to finish
                warn_proc = await asyncio.create_subprocess_exec("afplay", "temp_tts.mp3")
                await warn_proc.wait()
                
            # 4. Resume navigation
            if hasattr(self, '_nav_pause_event'):
                self._nav_pause_event.set()

        try:
            dispatched = await self._vision.send(request, on_obstruction_callback=on_obstruction)
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
