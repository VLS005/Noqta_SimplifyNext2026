"""
Message contract: State & Safety Agent -> Routing Agent.

Share this file (or at least the JSON_SCHEMA dict) with whoever owns the
Routing Agent so both sides agree on the payload shape.
"""

from dataclasses import dataclass, asdict
from domain.models import RerouteCause


@dataclass
class RerouteRequest:
    from_agent: str
    cause: str          # RerouteCause.value - one of: bad_weather | obstruction | disoriented
    detail: str         # human-readable / machine cause detail, e.g. "raining" or "circling"
    lat: float
    lon: float
    timestamp: float

    @classmethod
    def build(cls, cause: RerouteCause, detail: str, lat: float, lon: float, timestamp: float):
        return cls(
            from_agent="state_safety_agent",
            cause=cause.value,
            detail=detail,
            lat=lat,
            lon=lon,
            timestamp=timestamp,
        )

    def to_dict(self) -> dict:
        return asdict(self)


# Reference JSON schema for documentation / validation on the Routing Agent side.
JSON_SCHEMA = {
    "type": "object",
    "required": ["from_agent", "cause", "detail", "lat", "lon", "timestamp"],
    "properties": {
        "from_agent": {"type": "string", "const": "state_safety_agent"},
        "cause": {"type": "string", "enum": ["bad_weather", "obstruction", "disoriented"]},
        "detail": {"type": "string"},
        "lat": {"type": "number"},
        "lon": {"type": "number"},
        "timestamp": {"type": "number"},
    },
}
