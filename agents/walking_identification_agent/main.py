"""
Demo / test harness for the Walking Identification (State & Safety) Agent.

Run with:  python main.py
"""

import math
import time

from config.settings import AgentConfig
from domain.models import GPSPoint, SensorReading
from agent import StateSafetyAgent


def demo_sensor_stream():
    """Simulates: normal walking for a few ticks, then the user starts
    circling / slowing down near a fixed point (triggering disorientation logic)."""
    base_lat, base_lon = 1.3000, 103.8000
    t = time.time()

    # Normal progression: moving steadily forward
    for i in range(3):
        yield SensorReading(
            position=GPSPoint(base_lat + i * 0.0005, base_lon, t + i),
            heading_deg=90,
            steps_per_minute=100,
        )

    # Simulate circling + slow pace: tiny jitter around one spot
    circle_lat, circle_lon = base_lat + 0.0015, base_lon
    for i in range(20):
        jitter = 0.00002 * math.sin(i)
        yield SensorReading(
            position=GPSPoint(circle_lat + jitter, circle_lon + jitter, t + 3 + i),
            heading_deg=(90 + i * 30) % 360,
            steps_per_minute=40,
        )


if __name__ == "__main__":
    config = AgentConfig(poll_interval_sec=0.1, alert_cooldown_sec=3.0)  # sped up for demo
    agent = StateSafetyAgent(config=config, baseline_pace_spm=95)
    # In production this session_id comes from the Routing Agent's own
    # POST /api/route-request response - hardcoded here only for the
    # standalone demo, since this script doesn't call that endpoint.
    agent.start_journey(session_id="demo-session-001", waypoint_id="wp-1")
    agent.run(demo_sensor_stream())
    agent.stop_journey()
