"""Core data types shared across the agent's modules."""

from dataclasses import dataclass
from enum import Enum
from typing import Optional


@dataclass
class GPSPoint:
    lat: float
    lon: float
    timestamp: float


@dataclass
class SensorReading:
    """One tick of sensor data. In production this comes from the phone's
    accelerometer/gyroscope + step counter + GPS, sampled periodically
    (NOT continuous camera/video - see the hackathon's "Triggered Snapshot"
    architecture decision)."""
    position: GPSPoint
    heading_deg: float                        # compass heading, 0-360
    steps_per_minute: float                   # current pace
    lateral_wall_distance: Optional[float] = None  # meters, from OSM/vision if available


class IrregularPattern(Enum):
    NONE = "none"
    CIRCLING = "circling"
    TOO_SLOW = "too_slow"
    WALL_HUGGING = "wall_hugging"


class RerouteCause(Enum):
    BAD_WEATHER = "bad_weather"
    OBSTRUCTION = "obstruction"
    DISORIENTED = "disoriented"
