"""
Configuration for the State & Safety Agent (Walking Identification Agent).

Tune these thresholds during testing on the actual hackathon demo route -
GPS noise and phone sensor jitter will vary by device.

POLL_INTERVAL_SEC and ALERT_COOLDOWN_SEC can be overridden via a .env file
(see .env.example) without touching code - useful for quickly retuning
during live demo rehearsal.
"""

import os
from dataclasses import dataclass, field

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # python-dotenv not installed - fall back to whatever is already in
    # the environment (or the hardcoded defaults below). Run
    # `pip install python-dotenv` to enable .env file support.
    pass


def _env_float(key: str, default: float) -> float:
    value = os.environ.get(key)
    return float(value) if value else default


@dataclass
class AgentConfig:
    # -- Sensor polling --------------------------------------------------
    poll_interval_sec: float = field(default_factory=lambda: _env_float("POLL_INTERVAL_SEC", 5.0))
    # How many recent positions to keep for circling detection
    position_window: int = 12  # ~ last few minutes at default poll interval

    # -- Detection thresholds ---------------------------------------------
    circling_radius_m: float = 15.0
    # If pace < baseline * this ratio -> "too slow"
    slow_pace_ratio: float = 0.5
    wall_hug_distance_m: float = 0.3
    wall_hug_ticks_required: int = 4

    # -- Anti-nagging / debounce --------------------------------------------
    # Once an alert fires, suppress repeat alerts of the SAME cause for this
    # many seconds, so the user isn't buzzed/prompted every single tick while
    # the underlying condition persists.
    alert_cooldown_sec: float = field(default_factory=lambda: _env_float("ALERT_COOLDOWN_SEC", 30.0))
