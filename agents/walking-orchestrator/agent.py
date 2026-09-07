"""
StateSafetyAgent (Walking Identification Agent)
------------------------------------------------
Orchestrates: sensor ingestion -> pattern detection -> decision tree -> reroute handoff.

Lifecycle:
  1. Routing Agent pings start_journey() when navigation begins.
  2. ingest() is called on every sensor tick (polled, not streamed).
  3. On an irregular pattern, the decision engine runs weather -> obstruction
     -> disorientation checks and returns a reroute cause (or None / false alarm).
  4. A reroute payload is sent to the Routing Agent, respecting a cooldown so
     the user isn't nagged every tick while a condition persists.
"""

import time

from config.settings import AgentConfig
from domain.models import SensorReading, IrregularPattern
from domain.detection import PatternDetector
from domain.decision import DecisionEngine
from api_schemas.routing_payload import RerouteRequest
from integrations.routing_client import send_to_routing_agent


class StateSafetyAgent:
    def __init__(self, config: AgentConfig, baseline_pace_spm: float):
        self.config = config
        self.detector = PatternDetector(config, baseline_pace_spm)
        self.decision_engine = DecisionEngine()
        self.active = False
        self._last_alert_at: dict = {}  # cause.value -> last timestamp fired

    # -- lifecycle -----------------------------------------------------

    def start_journey(self):
        self.active = True
        self.detector.reset()
        self._last_alert_at.clear()
        print("[State & Safety Agent] Journey started. Monitoring begins.")

    def stop_journey(self):
        self.active = False
        print("[State & Safety Agent] Journey ended. Monitoring stopped.")

    # -- main loop -------------------------------------------------------

    def run(self, sensor_stream):
        """sensor_stream: iterable/generator of SensorReading, roughly one per
        config.poll_interval_sec. In production this is an async callback fed
        by the phone's sensor pipeline, not a blocking sleep loop."""
        for reading in sensor_stream:
            if not self.active:
                break
            self.ingest(reading)
            time.sleep(self.config.poll_interval_sec)  # remove in real async loop

    def ingest(self, reading: SensorReading):
        pattern = self.detector.evaluate(reading)
        if pattern is IrregularPattern.NONE:
            return

        # Gate BEFORE running the decision engine (which triggers the actual
        # haptic/audio prompt + vision calls). Otherwise the user gets
        # re-prompted every single tick while the condition persists, even
        # if the resulting reroute message is throttled.
        if self._in_cooldown(pattern.value, reading.position.timestamp):
            return

        print(f"[State & Safety Agent] Irregular pattern detected: {pattern.value}")
        self._last_alert_at[pattern.value] = reading.position.timestamp

        result = self.decision_engine.decide(pattern, reading)
        if result is None:
            return  # false alarm - stay quiet, cooldown still applies

        self._reroute(result.cause, result.detail, reading)

    # -- cooldown / anti-nag -------------------------------------------------

    def _in_cooldown(self, cause_key: str, now: float) -> bool:
        last = self._last_alert_at.get(cause_key)
        return last is not None and (now - last) < self.config.alert_cooldown_sec

    # -- reroute handoff -----------------------------------------------------

    def _reroute(self, cause, detail: str, reading: SensorReading):
        request = RerouteRequest.build(
            cause=cause,
            detail=detail,
            lat=reading.position.lat,
            lon=reading.position.lon,
            timestamp=reading.position.timestamp,
        )
        send_to_routing_agent(request)
        # Reset detector state so we don't immediately re-trigger circling/etc.
        # on the new route, but keep the cooldown timer so we still don't nag.
        self.detector.reset()
