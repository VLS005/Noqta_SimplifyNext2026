"""
Estimates personalized journey duration for a RoutePlan.
Accounts for fine-motor requirements using the user's mobility profile.
Looks up average walking pace from DynamoDB when available.
"""
from __future__ import annotations

import logging

from backend.routing_agent.config import get_settings
from backend.routing_agent.models import RoutePlan, MobilityProfile
from backend.shared.vector_store import VectorStore, get_vector_store

logger = logging.getLogger(__name__)


class ETAAgent:
    """
    Estimates personalised journey duration for a RoutePlan.
    Returns duration in whole seconds.
    Looks up user's average pace from DynamoDB; falls back to profile default.
    """

    def __init__(self, store: VectorStore | None = None) -> None:
        self._settings = get_settings()
        self._store = store or get_vector_store()

    async def estimate(
        self,
        route: RoutePlan,
        profile: MobilityProfile,
        user_id: str | None = None,
    ) -> RoutePlan:
        """
        Compute ETA and return an updated RoutePlan with estimated_duration_s set.
        If user_id is provided, looks up average pace from DynamoDB first.
        """
        # Determine pace to use
        pace_used = profile.preferred_pace_mps
        pace_source = "profile"

        if user_id:
            try:
                item = await self._store.get(f"PACE#{user_id}", "PACE#current")
                if item and "average_pace_mps" in item:
                    stored_pace = float(item["average_pace_mps"])
                    if 0.1 <= stored_pace <= 3.0:
                        pace_used = stored_pace
                        pace_source = "dynamodb"
                        print(
                            f"[ETAAgent] Found stored pace for user '{user_id}': "
                            f"{stored_pace:.2f} m/s"
                        )
            except Exception as exc:
                logger.warning(f"ETAAgent: pace lookup failed, using profile default: {exc}")
                print(f"[ETAAgent] Pace lookup failed for user '{user_id}': {exc}")

        total_distance_m = sum(
            wp.distance_to_next_m
            for wp in route.waypoints
            if wp.distance_to_next_m is not None
        )
        base_s = total_distance_m / pace_used

        fine_motor_count = sum(
            1 for wp in route.waypoints if wp.fine_motor_required
        )
        buffer_s = fine_motor_count * self._settings.fine_motor_buffer_s

        total_s = int(round(base_s + buffer_s))

        print(
            f"[ETAAgent] Route '{route.label}' | distance={total_distance_m:.1f}m | "
            f"pace={pace_used:.2f} m/s (source: {pace_source}) | "
            f"base_s={base_s:.0f} | fine_motor_buffer={buffer_s}s | total_eta={total_s}s"
        )

        logger.debug(
            "ETAAgent: computed",
            extra={
                "route_id": route.id,
                "distance_m": total_distance_m,
                "pace_mps": pace_used,
                "pace_source": pace_source,
                "base_s": base_s,
                "fine_motor_count": fine_motor_count,
                "buffer_s": buffer_s,
                "total_s": total_s,
            },
        )
        return route.model_copy(update={"estimated_duration_s": total_s})

    async def estimate_all(
        self,
        routes: list[RoutePlan],
        profile: MobilityProfile,
        user_id: str | None = None,
    ) -> list[RoutePlan]:
        """Estimate ETA for all candidates."""
        return [await self.estimate(r, profile, user_id) for r in routes]

    @staticmethod
    def format_duration(seconds: int) -> str:
        """
        Return a human-readable duration string suitable for TTS.
        e.g. 90 → 'about 1 minute 30 seconds', 420 → 'about 7 minutes'
        """
        if seconds < 60:
            return f"about {seconds} seconds"
        minutes = seconds // 60
        remainder = seconds % 60
        if remainder == 0:
            return f"about {minutes} minute{'s' if minutes != 1 else ''}"
        return f"about {minutes} minute{'s' if minutes != 1 else ''} and {remainder} seconds"
