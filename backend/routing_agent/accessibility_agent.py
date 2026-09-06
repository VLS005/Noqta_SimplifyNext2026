"""
Computes and injects a composite accessibility score into each candidate RoutePlan.
Personalizes scores using the user's MobilityProfile.
"""
from __future__ import annotations

import logging

import json
from backend.routing_agent.models import (
    AccessibilityScore,
    LightingQuality,
    MobilityWeights,
    RoutePlan,
    MobilityProfile,
)
from backend.routing_agent.bedrock_client import BedrockClient
from dotenv import load_dotenv

load_dotenv()

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
    Weights are derived dynamically from profile flags via an LLM.
    """

    def __init__(self, bedrock: BedrockClient) -> None:
        self._bedrock = bedrock

    async def score(self, route: RoutePlan, profile: MobilityProfile) -> RoutePlan:
        """
        Compute the accessibility composite score for one route.
        Returns the same RoutePlan with accessibility_score.composite_score populated.
        """
        weights = await self._compute_weights_from_profile(profile)
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

    async def score_all(
        self, routes: list[RoutePlan], profile: MobilityProfile
    ) -> list[RoutePlan]:
        """Score all candidates, sort by composite_score descending, and assign ranks."""
        import asyncio
        scored = list(await asyncio.gather(*(self.score(r, profile) for r in routes)))
        scored_sorted = sorted(
            scored, key=lambda r: r.accessibility_score.composite_score, reverse=True
        )
        # Assign ranks: rank 1 = most accessible/safe
        ranked = []
        for i, r in enumerate(scored_sorted, start=1):
            r_ranked = r.model_copy(update={"rank": i})
            ranked.append(r_ranked)
            
        return ranked

    # ─── Private helpers ──────────────────────────────────────────────────────

    async def _compute_weights_from_profile(self, profile: MobilityProfile) -> MobilityWeights:
        """
        Compute accessibility scoring weights dynamically from the user's
        MobilityProfile flags using the LLM.
        """
        if profile.mobility_weights is not None:
            return profile.mobility_weights

        system_prompt = (
            "You are an expert accessibility AI. Given a user's mobility profile, "
            "determine the optimal weights (summing to exactly 1.0) for 5 accessibility factors: "
            "step_free, tactile, lighting, crowding, obstruction.\n"
            "Return ONLY a valid JSON object with these 5 keys and float values, e.g.\n"
            "{\"step_free\": 0.3, \"tactile\": 0.2, \"lighting\": 0.2, \"crowding\": 0.15, \"obstruction\": 0.15}"
        )
        user_prompt = (
            f"User profile: vision_level={profile.vision_level}, "
            f"prefers_step_free={profile.prefers_step_free}, prefers_tactile_paving={profile.prefers_tactile_paving}, "
            f"avoids_crowded_areas={profile.avoids_crowded_areas}, preferred_lighting={profile.preferred_lighting}"
        )

        try:
            raw_response = await self._bedrock.invoke_model(system_prompt, user_prompt, model_tier="nova-lite")
            
            # Extract JSON block
            text = raw_response
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0]
            elif "```" in text:
                text = text.split("```")[1].split("```")[0]
            
            data = json.loads(text.strip())
            
            w_step_free = float(data.get("step_free", 0.25))
            w_tactile = float(data.get("tactile", 0.15))
            w_lighting = float(data.get("lighting", 0.15))
            w_crowding = float(data.get("crowding", 0.15))
            w_obstruction = float(data.get("obstruction", 0.10))
            
            total = w_step_free + w_tactile + w_lighting + w_crowding + w_obstruction
            if total == 0:
                total = 1.0
                
            return MobilityWeights(
                step_free=round(w_step_free / total, 4),
                tactile=round(w_tactile / total, 4),
                lighting=round(w_lighting / total, 4),
                crowding=round(w_crowding / total, 4),
                obstruction=round(w_obstruction / total, 4),
            )
        except Exception as e:
            logger.warning(f"AccessibilityAgent: failed to get weights from LLM ({e}). Using defaults.")
            return MobilityWeights()

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
