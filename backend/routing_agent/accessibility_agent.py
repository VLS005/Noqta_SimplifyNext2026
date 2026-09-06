"""
Computes and injects a composite accessibility score into each candidate RoutePlan.
Personalizes scores using the user's MobilityProfile.
"""
from __future__ import annotations

import logging

from backend.routing_agent.models import (
    AccessibilityScore,
    LightingQuality,
    MobilityWeights,
    RoutePlan,
    MobilityProfile,
)

logger = logging.getLogger(__name__)

_LIGHTING_SCORES: dict[LightingQuality, float] = {
    LightingQuality.BRIGHT: 1.0,
    LightingQuality.ADEQUATE: 0.6,
    LightingQuality.DIM: 0.2,
    LightingQuality.UNKNOWN: 0.4,
}


class AccessibilityAgent:
    """
    Computes and injects the composite_score into each RoutePlan's AccessibilityScore.
    Scores are personalised using the user's MobilityProfile.
    Weights are derived dynamically from profile flags (not hardcoded).
    """

    def score(self, route: RoutePlan, profile: MobilityProfile) -> RoutePlan:
        """
        Compute the accessibility composite score for one route.
        Returns the same RoutePlan with accessibility_score.composite_score populated.
        """
        weights = self._compute_weights_from_profile(profile)
        print(
            f"[AccessibilityAgent] Computed weights from profile: "
            f"step_free={weights.step_free}, tactile={weights.tactile}, "
            f"lighting={weights.lighting}, crowding={weights.crowding}, "
            f"obstruction={weights.obstruction}"
        )

        acc = route.accessibility_score
        raw_score = self._compute_raw(acc, weights)
        penalised = self._apply_profile_penalties(raw_score, acc, profile)
        final = round(min(max(penalised, 0.0), 1.0), 4)

        print(
            f"[AccessibilityAgent] Route '{route.label}' | "
            f"raw_score={raw_score:.4f} | penalised_final={final}"
        )

        # Return an updated route with the computed score
        updated_acc = acc.model_copy(update={"composite_score": final})
        updated_route = route.model_copy(update={"accessibility_score": updated_acc})

        logger.debug(
            "AccessibilityAgent: scored",
            extra={
                "route_id": route.id,
                "label": route.label,
                "raw_score": raw_score,
                "final_score": final,
            },
        )
        return updated_route

    def score_all(
        self, routes: list[RoutePlan], profile: MobilityProfile
    ) -> list[RoutePlan]:
        """Score all candidates, sort by composite_score descending, and assign ranks."""
        scored = [self.score(r, profile) for r in routes]
        scored_sorted = sorted(
            scored, key=lambda r: r.accessibility_score.composite_score, reverse=True
        )
        # Assign ranks: rank 1 = most accessible/safe
        ranked = []
        for i, r in enumerate(scored_sorted, start=1):
            r_ranked = r.model_copy(update={"rank": i})
            ranked.append(r_ranked)
            print(
                f"[AccessibilityAgent] RANKED: Route '{r_ranked.label}' -> "
                f"rank={r_ranked.rank}, composite_score={r_ranked.accessibility_score.composite_score}"
            )
        return ranked

    # ─── Private helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _compute_weights_from_profile(profile: MobilityProfile) -> MobilityWeights:
        """
        Compute accessibility scoring weights dynamically from the user's
        MobilityProfile flags. If the profile has explicit mobility_weights
        set, use those directly. Otherwise derive from preferences.
        """
        if profile.mobility_weights is not None:
            return profile.mobility_weights

        # Base weights
        w_step_free = 0.25
        w_tactile = 0.15
        w_lighting = 0.15
        w_crowding = 0.15
        w_obstruction = 0.10

        # Boost based on user preferences
        if profile.prefers_step_free:
            w_step_free += 0.10
        if profile.prefers_tactile_paving:
            w_tactile += 0.05
        if profile.vision_level == "blind":
            # Blind users: much higher weight on tactile, lighting, step-free
            w_tactile += 0.05
            w_lighting += 0.05
            w_step_free += 0.05
        if profile.avoids_crowded_areas:
            w_crowding += 0.10
        if profile.preferred_lighting == "bright":
            w_lighting += 0.05

        # Normalise to sum to 1.0
        total = w_step_free + w_tactile + w_lighting + w_crowding + w_obstruction
        return MobilityWeights(
            step_free=round(w_step_free / total, 4),
            tactile=round(w_tactile / total, 4),
            lighting=round(w_lighting / total, 4),
            crowding=round(w_crowding / total, 4),
            obstruction=round(w_obstruction / total, 4),
        )

    @staticmethod
    def _compute_raw(acc: AccessibilityScore, weights: MobilityWeights) -> float:
        """Compute the weighted raw accessibility score."""
        step_free_s = 1.0 if acc.step_free else 0.0
        tactile_s = 1.0 if acc.tactile_paving else 0.0
        lighting_s = _LIGHTING_SCORES.get(acc.lighting_quality, 0.4)
        crowding_s = 1.0 - acc.crowding_estimate       # inverted
        obstruction_s = 1.0 - acc.obstruction_risk     # inverted

        return (
            weights.step_free * step_free_s
            + weights.tactile * tactile_s
            + weights.lighting * lighting_s
            + weights.crowding * crowding_s
            + weights.obstruction * obstruction_s
        )

    @staticmethod
    def _apply_profile_penalties(
        raw_score: float,
        acc: AccessibilityScore,
        profile: MobilityProfile,
    ) -> float:
        score = raw_score
        # Strong preference for step-free routes
        if profile.prefers_step_free and not acc.step_free:
            score *= 0.80
            print("[AccessibilityAgent] Penalty applied: step-free preference violated")
            logger.debug("AccessibilityAgent: step-free penalty applied")
        # Crowding aversion
        if (
            profile.avoids_crowded_areas
            and acc.crowding_estimate > profile.max_crowding_tolerance
        ):
            score *= 0.85
            print("[AccessibilityAgent] Penalty applied: crowding threshold exceeded")
            logger.debug("AccessibilityAgent: crowding penalty applied")
        return score
