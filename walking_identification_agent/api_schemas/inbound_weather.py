"""
PLACEHOLDER / PROPOSED schema - this does NOT exist on the Routing Agent's
side yet.

Their /api/inbound/{agent_type} route currently only accepts:
    {"obstruction", "pace", "landmark", "vision_response"}
"weather" is not in that set. Sending this payload today will get a real
HTTP 400 back from their server ("Unknown agent_type 'weather'") - that's
expected, not a bug in this code. This file exists so the shape is ready to
go the moment the Routing Agent owner adds a matching handler, following the
same BaseMessage pattern as their real ObstructionReportMessage.

Field choices below intentionally mirror ObstructionReportMessage's shape
(severity/description-equivalent + lat/lon) for consistency, since that's
the one real precedent we have for their message design. Confirm the actual
field names with the Routing Agent owner before treating this as final -
this is a starting proposal, not an agreed contract.
"""

import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Optional, Literal


@dataclass
class WeatherAlertPayload:
    session_id: str
    condition: str  # e.g. "raining" - proposed field name, unconfirmed
    severity: Literal["low", "medium", "high"]
    lat: Optional[float] = None
    lon: Optional[float] = None
    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    message_type: str = "weather_alert"  # PROPOSED value - not in their real MessageType enum yet
    source_agent: str = "state_safety_agent"
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @classmethod
    def build(cls, session_id: str, weather: dict, lat: Optional[float] = None, lon: Optional[float] = None) -> "WeatherAlertPayload":
        return cls(
            session_id=session_id,
            condition=weather.get("condition", "unknown"),
            severity=weather.get("severity", "low"),
            lat=lat,
            lon=lon,
        )

    def to_dict(self) -> dict:
        return asdict(self)
