"""Calculate personalised transit ETA without treating bus/train distance as walking."""
from __future__ import annotations
import logging
from backend.routing_agent.config import get_settings
from backend.routing_agent.models import MobilityProfile, RoutePlan
from backend.shared.vector_store import VectorStore, get_vector_store
logger = logging.getLogger(__name__)

class ETAAgent:
    def __init__(self, store: VectorStore | None = None) -> None:
        self._settings = get_settings()
        self._store = store or get_vector_store()

    async def estimate(self, route: RoutePlan, profile: MobilityProfile, user_id: str | None = None) -> RoutePlan:
        pace, source = profile.preferred_pace_mps, "profile"
        if user_id:
            try:
                item = await self._store.get(f"PACE#{user_id}", "PACE#current")
                candidate = float(item.get("average_pace_mps")) if item else None
                if candidate is not None and 0.1 <= candidate <= 3:
                    pace, source = candidate, "dynamodb"
            except Exception as exc:
                logger.warning("ETA pace lookup failed: %s", exc)
        walking_distance = route.walking_distance_m or sum((wp.distance_to_next_m or 0) for wp in route.waypoints if wp.travel_mode_to_next == "WALKING")
        transit_duration = route.transit_duration_s or sum((wp.duration_to_next_s or 0) for wp in route.waypoints if wp.travel_mode_to_next == "TRANSIT")
        transit_legs = sum(wp.travel_mode_to_next == "TRANSIT" for wp in route.waypoints)
        walking_s = walking_distance / pace
        fine_motor_s = sum(wp.fine_motor_required for wp in route.waypoints) * self._settings.fine_motor_buffer_s
        transfer_s = max(0, transit_legs - 1) * self._settings.transit_transfer_buffer_s
        calculated_s = int(round(walking_s + transit_duration + fine_motor_s + transfer_s))
        provider_s = route.provider_duration_s or 0
        total_s = max(calculated_s, provider_s) if provider_s else calculated_s
        logger.info("ETA route=%s walking=%.0fm transit=%ss provider=%ss personalised=%ss", route.label, walking_distance, transit_duration, provider_s, total_s)
        return route.model_copy(update={"walking_distance_m": walking_distance, "transit_duration_s": transit_duration, "estimated_duration_s": total_s})

    async def estimate_all(self, routes: list[RoutePlan], profile: MobilityProfile, user_id: str | None = None) -> list[RoutePlan]:
        return [await self.estimate(route, profile, user_id) for route in routes]

    @staticmethod
    def format_duration(seconds: int) -> str:
        minutes, remainder = divmod(max(0, seconds), 60)
        if minutes < 1:
            return f"about {remainder} seconds"
        if remainder == 0:
            return f"about {minutes} minute{'s' if minutes != 1 else ''}"
        return f"about {minutes} minute{'s' if minutes != 1 else ''} and {remainder} seconds"