"""Implements the decision tree from the spec:
    1. Bad weather?         -> reroute (safe/sheltered path)
    2. Visible obstruction? -> reroute (around obstruction)
    3. Otherwise            -> likely disoriented -> minimally audible prompt
                               -> find anchor point -> reroute if confirmed
"""

from typing import Optional

from domain.models import SensorReading, IrregularPattern, RerouteCause
from integrations.weather_client import get_weather
from integrations.vision_client import check_obstruction, find_anchor_point
from integrations.haptic_audio import prompt_user_minimally_audible


class DecisionResult:
    def __init__(self, cause: RerouteCause, detail: str):
        self.cause = cause
        self.detail = detail


class DecisionEngine:
    def decide(self, pattern: IrregularPattern, reading: SensorReading) -> Optional[DecisionResult]:
        """Returns a DecisionResult if a reroute is warranted, or None if it
        turned out to be a false alarm (e.g. user says they're fine)."""

        weather = get_weather(reading.position.lat, reading.position.lon)
        if weather.get("raining"):
            return DecisionResult(RerouteCause.BAD_WEATHER, weather["condition"])

        obstruction = check_obstruction(reading.position)
        if obstruction:
            return DecisionResult(RerouteCause.OBSTRUCTION, obstruction)

        return self._handle_disorientation(pattern, reading)

    def _handle_disorientation(self, pattern: IrregularPattern, reading: SensorReading) -> Optional[DecisionResult]:
        anchor = find_anchor_point(reading.position)
        if anchor:
            message = (
                f"It seems you might be disoriented. I see a {anchor['object']} "
                f"to your {anchor['direction']}, about {anchor['distance_m']} steps away."
            )
        else:
            message = "It seems you might be lost. Tap the screen, or say 'help' to confirm."

        confirmation = prompt_user_minimally_audible(message)
        if confirmation.lower() in ("yes", "help"):
            return DecisionResult(RerouteCause.DISORIENTED, pattern.value)

        # False alarm - stay quiet, keep monitoring, don't nag the user
        return None
