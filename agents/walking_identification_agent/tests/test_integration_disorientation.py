"""
Integration tests for the full pipeline: sensor readings -> PatternDetector
-> DecisionEngine -> haptic prompt -> Routing Agent handoff.

Updated to match the Routing Agent's ACTUAL contract (verified against their
real backend code):
  - start_journey() now requires session_id (their schema rejects messages
    without one) and accepts an optional waypoint_id.
  - Only RerouteCause.OBSTRUCTION has a matching Routing Agent endpoint
    (POST /api/inbound/obstruction). bad_weather and disoriented are logged
    locally via report_unsupported_cause() instead - there is no HTTP call
    to assert on for those two causes, only a local log message.
  - Obstruction detail is now a dict {"description": ..., "severity": ...},
    not a bare string, since their schema requires severity separately.

Run with: python tests/test_integration_disorientation.py
"""

import sys
import os
import math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import agent as agent_module
import domain.decision as decision_module
from config.settings import AgentConfig
from domain.models import GPSPoint, SensorReading, RerouteCause
from agent import StateSafetyAgent


def _install_fakes(anchor_found=True, user_confirms="yes", obstruction=None, raining=False):
    """Swap the real (mocked-but-still-external) integration calls for
    in-memory recorders, so tests can assert on what actually happened.

    obstruction: None (path clear) or a dict {"description": ..., "severity": ...}
    """
    sent_obstruction_payloads = []
    sent_weather_payloads = []
    unsupported_calls = []
    prompts = []

    def fake_send_obstruction_report(payload):
        sent_obstruction_payloads.append(payload.to_dict())

    def fake_send_weather_report(payload):
        sent_weather_payloads.append(payload.to_dict())

    def fake_report_unsupported_cause(cause, detail, lat, lon):
        unsupported_calls.append({"cause": cause.value, "detail": detail, "lat": lat, "lon": lon})

    def fake_prompt(message):
        prompts.append(message)
        return user_confirms

    def fake_find_anchor(position):
        if anchor_found:
            return {"object": "handrail", "direction": "right", "distance_m": 2}
        return None

    def fake_check_obstruction(position):
        return obstruction  # None = path clear; a dict = obstruction detected

    def fake_get_weather(lat, lon):
        return {"raining": raining, "condition": "raining" if raining else "clear", "severity": "medium" if raining else "low"}

    # Patch the names as looked up inside each module's own namespace -
    # this is what actually gets called at runtime, not the original definition.
    agent_module.send_obstruction_report = fake_send_obstruction_report
    agent_module.send_weather_report = fake_send_weather_report
    agent_module.report_unsupported_cause = fake_report_unsupported_cause
    decision_module.prompt_user_minimally_audible = fake_prompt
    decision_module.find_anchor_point = fake_find_anchor
    decision_module.check_obstruction = fake_check_obstruction
    decision_module.get_weather = fake_get_weather

    return sent_obstruction_payloads, sent_weather_payloads, unsupported_calls, prompts


def _circling_readings(n=12, base_pace=95):
    """Positions that jitter around one spot - should trigger CIRCLING,
    not TOO_SLOW (pace is kept at baseline on purpose)."""
    base_lat, base_lon = 1.3000, 103.8000
    readings = []
    for i in range(n):
        jitter = 0.00002 * math.sin(i)
        readings.append(SensorReading(
            position=GPSPoint(base_lat + jitter, base_lon + jitter, i),
            heading_deg=(90 + i * 30) % 360,
            steps_per_minute=base_pace,
        ))
    return readings


def _normal_walking_readings(n=12, base_pace=95):
    """Steady forward progress - should NOT trigger any pattern."""
    base_lat, base_lon = 1.3000, 103.8000
    readings = []
    for i in range(n):
        readings.append(SensorReading(
            position=GPSPoint(base_lat + i * 0.0005, base_lon, i),
            heading_deg=90,
            steps_per_minute=base_pace,
        ))
    return readings


def _new_agent(config=None, baseline_pace_spm=95, session_id="test-session-1", waypoint_id="wp-1"):
    config = config or AgentConfig(position_window=8, alert_cooldown_sec=999)
    a = StateSafetyAgent(config=config, baseline_pace_spm=baseline_pace_spm)
    a.start_journey(session_id=session_id, waypoint_id=waypoint_id)
    return a


def test_start_journey_requires_session_id():
    """The Routing Agent's schema rejects messages without a session_id -
    this agent should fail loudly, not silently proceed without one."""
    config = AgentConfig()
    a = StateSafetyAgent(config=config, baseline_pace_spm=95)
    try:
        a.start_journey(session_id="")
        assert False, "Expected ValueError for empty session_id"
    except ValueError:
        pass  # expected


def test_circling_triggers_prompt_and_local_log_not_http():
    """disoriented has NO matching Routing Agent endpoint - confirm it goes
    through report_unsupported_cause, and NOT send_obstruction_report."""
    sent_obstruction, sent_weather, unsupported, prompts = _install_fakes(anchor_found=True, user_confirms="yes")
    agentx = _new_agent()

    for reading in _circling_readings(n=10):
        agentx.ingest(reading)

    assert len(prompts) >= 1, "Expected the disorientation prompt to fire at least once"
    assert "handrail" in prompts[0], "Expected the anchor-point message when one is found"
    assert len(unsupported) >= 1, "Expected disoriented to be logged via report_unsupported_cause"
    assert unsupported[0]["cause"] == "disoriented"
    assert len(sent_obstruction) == 0, "disoriented must NOT go through the obstruction HTTP path"


def test_normal_walking_never_prompts():
    sent_obstruction, sent_weather, unsupported, prompts = _install_fakes()
    agentx = _new_agent()

    for reading in _normal_walking_readings(n=10):
        agentx.ingest(reading)

    assert len(prompts) == 0, "Normal walking should never trigger a prompt"
    assert len(unsupported) == 0
    assert len(sent_obstruction) == 0


def test_false_alarm_does_not_reroute():
    sent_obstruction, sent_weather, unsupported, prompts = _install_fakes(anchor_found=False, user_confirms="no")
    agentx = _new_agent()

    for reading in _circling_readings(n=10):
        agentx.ingest(reading)

    assert len(prompts) >= 1, "Prompt should still fire even if it turns out to be a false alarm"
    assert "lost" in prompts[0].lower()
    assert len(unsupported) == 0, "A 'no' response should NOT trigger any reroute/log"
    assert len(sent_obstruction) == 0


def test_obstruction_sends_real_payload_matching_routing_agent_schema():
    """The one cause that DOES have a real endpoint - verify the exact
    field shape matches the Routing Agent's ObstructionReportMessage."""
    obstruction = {"description": "scaffolding pole at head height", "severity": "high"}
    sent_obstruction, sent_weather, unsupported, prompts = _install_fakes(obstruction=obstruction)
    agentx = _new_agent(waypoint_id="wp-42")

    for reading in _circling_readings(n=10):
        agentx.ingest(reading)

    assert len(sent_obstruction) >= 1, "Expected an obstruction report to be sent"
    payload = sent_obstruction[0]
    assert payload["session_id"] == "test-session-1"
    assert payload["waypoint_id"] == "wp-42"
    assert payload["severity"] == "high"
    assert payload["description"] == "scaffolding pole at head height"
    assert payload["message_type"] == "obstruction_report"
    assert "message_id" in payload and "timestamp" in payload
    assert len(prompts) == 0, "Obstruction should short-circuit BEFORE the disorientation prompt"
    assert len(unsupported) == 0, "Obstruction IS supported - should not hit the unsupported-cause path"


def test_bad_weather_defaults_to_local_log_not_http():
    """SAFE DEFAULT: attempt_weather_placeholder is False by default, so
    bad_weather behaves like disoriented - no risk of an unexplained
    'FAILED' line appearing during a live demo unless explicitly enabled."""
    sent_obstruction, sent_weather, unsupported, prompts = _install_fakes(raining=True)
    agentx = _new_agent()  # default config: attempt_weather_placeholder=False

    for reading in _circling_readings(n=10):
        agentx.ingest(reading)

    assert len(sent_weather) == 0, "Should NOT attempt the HTTP placeholder by default"
    assert len(unsupported) >= 1
    assert unsupported[0]["cause"] == "bad_weather"
    assert len(prompts) == 0


def test_bad_weather_sends_placeholder_payload_when_explicitly_enabled():
    """When attempt_weather_placeholder=True is explicitly set, bad_weather
    DOES attempt the placeholder endpoint."""
    sent_obstruction, sent_weather, unsupported, prompts = _install_fakes(raining=True)
    config = AgentConfig(position_window=8, alert_cooldown_sec=999, attempt_weather_placeholder=True)
    agentx = _new_agent(config=config)

    for reading in _circling_readings(n=10):
        agentx.ingest(reading)

    assert len(sent_weather) >= 1, "Expected a weather report to be attempted when explicitly enabled"
    payload = sent_weather[0]
    assert payload["session_id"] == "test-session-1"
    assert payload["condition"] == "raining"
    assert payload["severity"] == "medium"
    assert payload["message_type"] == "weather_alert"
    assert len(sent_obstruction) == 0
    assert len(unsupported) == 0
    assert len(prompts) == 0


def test_cooldown_prevents_repeat_prompts():
    """With a short cooldown, repeated circling readings should only prompt
    once per cooldown window, not on every single tick."""
    sent_obstruction, sent_weather, unsupported, prompts = _install_fakes(user_confirms="yes")
    config = AgentConfig(position_window=4, alert_cooldown_sec=100)  # long cooldown
    agentx = _new_agent(config=config)

    for reading in _circling_readings(n=20):
        agentx.ingest(reading)

    assert len(prompts) == 1, f"Expected exactly 1 prompt due to cooldown, got {len(prompts)}"


if __name__ == "__main__":
    test_start_journey_requires_session_id()
    print("PASS: start_journey requires a real session_id")

    test_circling_triggers_prompt_and_local_log_not_http()
    print("PASS: circling triggers prompt + local log (disoriented has no HTTP endpoint)")

    test_normal_walking_never_prompts()
    print("PASS: normal walking never prompts")

    test_false_alarm_does_not_reroute()
    print("PASS: false alarm does not reroute")

    test_obstruction_sends_real_payload_matching_routing_agent_schema()
    print("PASS: obstruction sends a real payload matching the Routing Agent's schema")

    test_bad_weather_defaults_to_local_log_not_http()
    print("PASS: bad weather defaults to local log, no HTTP attempt (demo-safe)")

    test_bad_weather_sends_placeholder_payload_when_explicitly_enabled()
    print("PASS: bad weather attempts placeholder when explicitly enabled")

    test_cooldown_prevents_repeat_prompts()
    print("PASS: cooldown prevents repeat prompts")

    print("\nAll integration tests passed.")
