from enum import Enum
from pydantic import BaseModel
from typing import Optional

class GPSPoint(BaseModel):
    lat: float
    lon: float
    timestamp: float

class SensorReading(BaseModel):
    position: GPSPoint
    heading_deg: float
    steps_per_minute: float
    timestamp: float

class IrregularPattern(Enum):
    NONE = "none"
    CIRCLING = "circling"
    STOPPED = "stopped"
    ERRATIC = "erratic"
    SLOW = "slow"

class RerouteCause(Enum):
    OBSTRUCTION = "obstruction"
    BAD_WEATHER = "bad_weather"
    DISORIENTED = "disoriented"
