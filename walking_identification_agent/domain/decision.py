"""Implements the decision tree from the spec:
    1. Bad weather?         -> reroute (safe/sheltered path)
    2. Visible obstruction? -> reroute (around obstruction)
    3. Otherwise            -> likely disoriented -> minimally audible prompt
                               -> find anchor point -> reroute if confirmed

NOTE on Routing Agent compatibility (checked against the actual
routing-orchestrator backend, not assumed): its /api/inbound/{agent_type}
endpoint only accepts agent_type in {"obstruction", "pace", "landmark",
"vision_response"}. There is currently NO accepted type for "bad_weather" or
a general "disoriented" signal - only "obstruction" has a real home there.
DecisionResult.detail is typed as `dict | str` because the two paths carry
different shapes: obstruction needs a dict with 'description' + 'severity'
(required separately by the Routing Agent's schema), while the other two
causes remain simple strings until/unless the team adds matching endpoints.
"""

from typing import Optional, Union

from domain.models import SensorReading, IrregularPattern, RerouteCause
from integrations.weather_client import get_weather
from integrations.vision_client import check_obstruction, find_anchor_point
from integrations.haptic_audio import prompt_user_minimally_audible


class DecisionResult:
    def __init__(self, cause: RerouteCause, detail: Union[dict, str]):
        self.cause = cause
        self.detail = detail  # dict for OBSTRUCTION (description+severity), str otherwise


class DecisionEngine:
    def decide(self, pattern: IrregularPattern, reading: SensorReading) -> Optional[DecisionResult]:
        """Returns a DecisionResult if a reroute is warranted, or None if it
        turned out to be a false alarm (e.g. user says they're fine)."""

        weather = get_weather(reading.position.lat, reading.position.lon)
        if weather.get("raining"):
            # Pass the full dict (condition + severity) through, mirroring
            # the obstruction shape - agent.py builds the placeholder
            # WeatherAlertPayload from this.
            return DecisionResult(RerouteCause.BAD_WEATHER, weather)

        obstruction = check_obstruction(reading.position)
        if obstruction:
            # obstruction is now a dict: {"description": ..., "severity": ...}
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
