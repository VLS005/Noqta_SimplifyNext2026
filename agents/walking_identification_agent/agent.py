"""
StateSafetyAgent (Walking Identification Agent)
------------------------------------------------
Orchestrates: sensor ingestion -> pattern detection -> decision tree -> reroute handoff.

Lifecycle:
  1. Routing Agent pings start_journey(session_id, waypoint_id) when
     navigation begins. session_id and waypoint_id are REQUIRED by the
     Routing Agent's actual API contract (verified against their code) - a
     navigation session must already exist there before this agent can
     report anything back to it.
  2. ingest() is called on every sensor tick (polled, not streamed).
  3. On an irregular pattern, the decision engine runs weather -> obstruction
     -> disorientation checks and returns a reroute cause (or None / false alarm).
  4. A reroute payload is sent to the Routing Agent, respecting a cooldown so
     the user isn't nagged every tick while a condition persists.

     IMPORTANT: only RerouteCause.OBSTRUCTION currently has a matching
     endpoint on the Routing Agent's side (POST /api/inbound/obstruction).
     bad_weather and disoriented are logged locally via
     report_unsupported_cause() instead of attempted over HTTP, since
     sending them would just 404 - see integrations/routing_client.py.
"""

import time
from typing import Optional

from config.settings import AgentConfig
from domain.models import SensorReading, IrregularPattern, RerouteCause
from domain.detection import PatternDetector
from domain.decision import DecisionEngine
from api_schemas.inbound_obstruction import ObstructionReportPayload
from api_schemas.inbound_weather import WeatherAlertPayload
from integrations.routing_client import send_obstruction_report, send_weather_report, report_unsupported_cause


class StateSafetyAgent:
    def __init__(self, config: AgentConfig, baseline_pace_spm: float):
        self.config = config
        self.detector = PatternDetector(config, baseline_pace_spm)
        self.decision_engine = DecisionEngine()
        self.active = False
        self._last_alert_at: dict = {}  # cause.value -> last timestamp fired
        self.session_id: Optional[str] = None
        self.current_waypoint_id: Optional[str] = None

    # -- lifecycle -----------------------------------------------------

    def start_journey(self, session_id: str, waypoint_id: Optional[str] = None):
        """session_id must come from the Routing Agent's own
        POST /api/route-request response (its 'session_id' field) - this
        agent cannot invent one, since the Routing Agent's schema requires
        session_id on every inbound message and presumably ties it to an
        active route. waypoint_id is optional at start (may not know it
        yet) but MUST be set via update_waypoint() before an obstruction
        report can actually be sent - see _reroute()."""
        if not session_id:
            raise ValueError(
                "start_journey() requires a real session_id from the Routing "
                "Agent's /api/route-request response - the Routing Agent's "
                "schema rejects messages without one."
            )
        self.session_id = session_id
        self.current_waypoint_id = waypoint_id
        self.active = True
        self.detector.reset()
        self._last_alert_at.clear()
        print(f"[State & Safety Agent] Journey started (session_id={session_id}). Monitoring begins.")

    def update_waypoint(self, waypoint_id: str):
        """Call this whenever the Routing Agent notifies which waypoint the
        user is currently approaching/on, so obstruction reports can
        reference the correct waypoint_id (required by their schema)."""
        self.current_waypoint_id = waypoint_id

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

    def _reroute(self, cause: RerouteCause, detail, reading: SensorReading):
        if cause is RerouteCause.OBSTRUCTION:
            waypoint_id = self.current_waypoint_id or "unknown"
            if waypoint_id == "unknown":
                print(
                    "[State & Safety Agent] WARNING: sending obstruction report "
                    "with waypoint_id='unknown' - update_waypoint() was never "
                    "called. The Routing Agent may not be able to act on this "
                    "correctly without a real waypoint_id."
                )
            payload = ObstructionReportPayload.build(
                session_id=self.session_id,
                waypoint_id=waypoint_id,
                obstruction=detail,  # {"description": ..., "severity": ...}
                lat=reading.position.lat,
                lon=reading.position.lon,
            )
            send_obstruction_report(payload)
        elif cause is RerouteCause.BAD_WEATHER:
            if not self.config.attempt_weather_placeholder:
                # Safer default: behave like disoriented (local-log only).
                # Flip AgentConfig.attempt_weather_placeholder=True to
                # actually attempt the placeholder HTTP call - see its
                # docstring for why that currently gets a real 400.
                report_unsupported_cause(cause, detail, reading.position.lat, reading.position.lon)
            else:
                payload = WeatherAlertPayload.build(
                    session_id=self.session_id,
                    weather=detail,  # {"raining": ..., "condition": ..., "severity": ...}
                    lat=reading.position.lat,
                    lon=reading.position.lon,
                )
                send_weather_report(payload)
        else:
            # disoriented: no matching Routing Agent endpoint AND no
            # placeholder built for it yet - log locally only.
            report_unsupported_cause(cause, detail, reading.position.lat, reading.position.lon)

        # Reset detector state so we don't immediately re-trigger circling/etc.
        # on the new route, but keep the cooldown timer so we still don't nag.
        self.detector.reset()
