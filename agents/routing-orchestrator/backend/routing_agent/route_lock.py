"""
State machine for the route selection lifecycle backed by VectorStore.
Manages transitions between UNLOCKED, CANDIDATES_PRESENTED, and LOCKED states.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime

from backend.routing_agent.models import LockedRoute, RouteSelectionState, RoutePlan
from backend.shared.vector_store import get_vector_store, VectorStore

logger = logging.getLogger(__name__)


class RouteLockError(Exception):
    """Raised on invalid state transitions."""


class RouteLockService:
    """
    Thread-safe (asyncio) route lock state machine backed by VectorStore.
    One instance per application (singleton injected via DI).
    """

    def __init__(self, store: VectorStore | None = None) -> None:
        self._store = store or get_vector_store()
        # Local cache: session_id → state (reduces store lookups)
        self._state_cache: dict[str, RouteSelectionState] = {}
        self._candidates_cache: dict[str, list[RoutePlan]] = {}

    # ─── State read ───────────────────────────────────────────────────────────

    async def get_state(self, session_id: str) -> RouteSelectionState:
        if session_id in self._state_cache:
            return self._state_cache[session_id]
        pk, sk = VectorStore.route_state_keys(session_id)
        item = await self._store.get(pk, sk)
        if item:
            state = RouteSelectionState(item.get("state", RouteSelectionState.UNLOCKED))
        else:
            state = RouteSelectionState.UNLOCKED
        self._state_cache[session_id] = state
        return state

    async def get_locked_route(self, session_id: str) -> LockedRoute | None:
        """Return the LockedRoute for a session, or None if not locked."""
        pk, sk = VectorStore.locked_route_keys(session_id)
        item = await self._store.get(pk, sk)
        if item and "locked_route_json" in item:
            return LockedRoute.model_validate_json(item["locked_route_json"])
        return None

    async def get_candidates(self, session_id: str) -> list[RoutePlan]:
        """Return the candidates presented in the last planning cycle."""
        if session_id in self._candidates_cache:
            return self._candidates_cache[session_id]
        pk, sk = VectorStore.route_plan_keys(session_id, "candidates")
        item = await self._store.get(pk, sk)
        if item and "candidates_json" in item:
            data = json.loads(item["candidates_json"])
            return [RoutePlan.model_validate(r) for r in data]
        return []

    # ─── State transitions ────────────────────────────────────────────────────

    async def present_candidates(
        self, session_id: str, candidates: list[RoutePlan]
    ) -> None:
        """
        Transition to CANDIDATES_PRESENTED and persist route options.
        Called after route planning + scoring + translation is complete.
        """
        current = await self.get_state(session_id)
        if current == RouteSelectionState.LOCKED:
            raise RouteLockError(
                f"Session {session_id} is LOCKED — unlock before presenting new candidates."
            )
        # Persist candidates
        pk, sk = VectorStore.route_plan_keys(session_id, "candidates")
        await self._store.put(pk, sk, {
            "candidates_json": json.dumps([r.model_dump(mode="json") for r in candidates]),
            "updated_at": datetime.utcnow().isoformat(),
        })
        self._candidates_cache[session_id] = candidates
        # Update state
        await self._set_state(session_id, RouteSelectionState.CANDIDATES_PRESENTED)
        logger.info(
            "RouteLockService: CANDIDATES_PRESENTED",
            extra={"session_id": session_id, "count": len(candidates)},
        )

    async def lock(self, session_id: str, route_id: str) -> LockedRoute:
        """
        Transition to LOCKED using the chosen route_id.
        Route selection is always via voice input (confirmed_by='voice').
        Raises RouteLockError if not in CANDIDATES_PRESENTED state or route_id unknown.
        """
        current = await self.get_state(session_id)
        if current != RouteSelectionState.CANDIDATES_PRESENTED:
            raise RouteLockError(
                f"Cannot lock from state '{current}'. "
                "Must be in CANDIDATES_PRESENTED state."
            )
        candidates = await self.get_candidates(session_id)
        chosen = next((r for r in candidates if r.id == route_id), None)
        if chosen is None:
            raise RouteLockError(
                f"Route ID '{route_id}' not found in candidates for session {session_id}."
            )
        locked = LockedRoute(
            route_plan=chosen,
            session_id=session_id,
            confirmed_by="voice",
        )
        # Persist locked route
        pk, sk = VectorStore.locked_route_keys(session_id)
        await self._store.put(pk, sk, {
            "locked_route_json": locked.model_dump_json(),
            "locked_at": locked.locked_at.isoformat(),
        })
        await self._set_state(session_id, RouteSelectionState.LOCKED)
        logger.info(
            "RouteLockService: LOCKED",
            extra={"session_id": session_id, "route_id": route_id, "label": chosen.label},
        )
        return locked

    async def unlock(self, session_id: str) -> None:
        """
        Reset to UNLOCKED (e.g. user wants to re-plan).
        Clears the locked route and candidates from the store.
        """
        # Remove locked route
        pk_l, sk_l = VectorStore.locked_route_keys(session_id)
        await self._store.delete(pk_l, sk_l)
        # Remove candidates
        pk_c, sk_c = VectorStore.route_plan_keys(session_id, "candidates")
        await self._store.delete(pk_c, sk_c)
        self._candidates_cache.pop(session_id, None)
        await self._set_state(session_id, RouteSelectionState.UNLOCKED)
        logger.info("RouteLockService: UNLOCKED", extra={"session_id": session_id})

    # ─── Helpers ──────────────────────────────────────────────────────────────

    async def _set_state(self, session_id: str, state: RouteSelectionState) -> None:
        self._state_cache[session_id] = state
        pk, sk = VectorStore.route_state_keys(session_id)
        await self._store.put(pk, sk, {
            "state": state.value,
            "updated_at": datetime.utcnow().isoformat(),
        })
