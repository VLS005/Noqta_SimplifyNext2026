"""
Payload matching the Routing Agent's ACTUAL ObstructionReportMessage schema.

Verified directly against their backend code
(backend/routing_agent/models.py + communication_agent.py), not assumed:

    class ObstructionReportMessage(BaseMessage):
        message_type: Literal["obstruction_report"] = "obstruction_report"
        source_agent: str = "obstruction_agent"
        waypoint_id: str
        severity: Literal["low", "medium", "high"]
        description: str
        lat: float | None = None
        lon: float | None = None

    class BaseMessage(BaseModel):
        message_id: str = <auto uuid4>
        message_type: MessageType
        session_id: str
        timestamp: datetime = <auto utcnow>
        source_agent: str = "unknown"

Endpoint: POST /api/inbound/obstruction  (agent_type is a URL path segment,
NOT a field in the JSON body - this differs from our first draft).

IMPORTANT LIMITATION (be upfront about this, don't paper over it):
Their own code comments mark this handler as a no-op for now:
    "Currently a read-only, no-op adapter — logs and discards."
    "The Routing Agent does NOT own obstruction detection."
So even a perfectly-shaped payload won't trigger a live reroute on their end
yet - it will be accepted (200 OK) and logged, nothing more, until they wire
up the TODO on their side.
"""

import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Optional, Literal


@dataclass
class ObstructionReportPayload:
    session_id: str
    waypoint_id: str
    severity: Literal["low", "medium", "high"]
    description: str
    lat: Optional[float] = None
    lon: Optional[float] = None
    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    message_type: str = "obstruction_report"
    source_agent: str = "state_safety_agent"
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @classmethod
    def build(
        cls,
        session_id: str,
        waypoint_id: str,
        obstruction: dict,  # {"description": ..., "severity": ...} from vision_client
        lat: Optional[float] = None,
        lon: Optional[float] = None,
    ) -> "ObstructionReportPayload":
        return cls(
            session_id=session_id,
            waypoint_id=waypoint_id,
            severity=obstruction["severity"],
            description=obstruction["description"],
            lat=lat,
            lon=lon,
        )

    def to_dict(self) -> dict:
        return asdict(self)
