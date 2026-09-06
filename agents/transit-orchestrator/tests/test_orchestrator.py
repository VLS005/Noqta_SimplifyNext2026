import pytest

from app.config import Settings
from app.models import Action, CheckRequest, JourneyRequest
from app.orchestrator import TransitOrchestrator


def request(scenario: str) -> CheckRequest:
    return CheckRequest(
        journey=JourneyRequest(
            bus_stop_code="27211",
            expected_bus="199",
            destination="Boon Lay MRT",
            original_eta_minutes=3,
        ),
        demo_scenario=scenario,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("scenario,action", [
    ("on_time", Action.BOARD),
    ("late", Action.REROUTE),
    ("wrong_bus", Action.IGNORE_BUS),
    ("off_service", Action.REROUTE),
    ("conflict", Action.CHECK_CAMERA),
])
async def test_demo_scenarios(scenario, action):
    agent = TransitOrchestrator(Settings(demo_mode=True))
    decision = await agent.run(request(scenario))
    assert decision.action == action
    assert decision.trace
    assert decision.haptic_command.actuators


@pytest.mark.asyncio
async def test_reroute_contains_accessibility_contract():
    decision = await TransitOrchestrator(Settings(demo_mode=True)).run(request("late"))
    assert decision.routing_handoff is not None
    assert decision.routing_handoff.accessibility_preferences.prefer_tactile_paving
    assert decision.routing_handoff.event == "REROUTE_REQUIRED"
