from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class GpsPoint(BaseModel):
    lat: float = Field(ge=-90, le=90)
    lon: float = Field(ge=-180, le=180)


class JourneyStartRequest(BaseModel):
    user_id: str
    route_id: str
    gps: GpsPoint
    timestamp: float


class JourneyStartResponse(BaseModel):
    journey_id: str


class JourneyTickRequest(BaseModel):
    journey_id: str
    steps_since_last_tick: int = Field(ge=0)
    timestamp: float


class JourneyEndRequest(BaseModel):
    journey_id: str


class JourneyEndResponse(BaseModel):
    journey_pace_spm: float
    rolling_avg_pace_spm: float


class BaselinePaceResponse(BaseModel):
    user_id: str
    baseline_pace_spm: float
    is_default: bool


class MemoryQueryRequest(BaseModel):
    lat: float = Field(ge=-90, le=90)
    lon: float = Field(ge=-180, le=180)
    day_of_week: str
    time_of_day: str


class MemoryMatch(BaseModel):
    description: str
    confidence: float = Field(ge=0, le=1)
    lat: float
    lon: float


class MemoryQueryResponse(BaseModel):
    matches: list[MemoryMatch] = Field(default_factory=list)


class MemoryStoreRequest(BaseModel):
    lat: float = Field(ge=-90, le=90)
    lon: float = Field(ge=-180, le=180)
    day_of_week: str
    time_of_day: str
    description: str


class MemoryStoreResponse(BaseModel):
    stored: bool
    memory_id: str


class StartTimerEventMessage(BaseModel):
    """Routing Agent StartTimerEventMessage. Additive; does not change JourneyStartRequest."""

    model_config = ConfigDict(extra="ignore")

    message_id: str
    message_type: str = "start_timer_event"
    session_id: str
    timestamp: datetime
    source_agent: str = "routing_agent"
    user_id: str | None = None
    locked_route_id: str
    estimated_duration_s: float = 0
    destination_label: str = ""
    route_distance_m: float = 0.0


class EndTimerRequest(BaseModel):
    session_id: str


class StartTimerAck(BaseModel):
    status: str = "ok"
