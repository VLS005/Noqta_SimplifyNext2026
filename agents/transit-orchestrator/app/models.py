from datetime import datetime, timezone
from enum import Enum
from pydantic import BaseModel, Field, model_validator


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class Action(str, Enum):
    WAIT = "WAIT"
    CHECK_CAMERA = "CHECK_CAMERA"
    BOARD = "BOARD"
    IGNORE_BUS = "IGNORE_BUS"
    REROUTE = "REROUTE"
    REQUEST_HELP = "REQUEST_HELP"


class AccessibilityPreferences(BaseModel):
    minimise_walking: bool = True
    prefer_sheltered_paths: bool = True
    prefer_tactile_paving: bool = True
    prefer_signalised_crossings: bool = True
    avoid_stairs: bool = True
    maximum_transfers: int = Field(default=1, ge=0, le=5)


class JourneyRequest(BaseModel):
    user_id: str = "demo-user"
    bus_stop_code: str
    expected_bus: str
    destination: str
    latitude: float | None = None
    longitude: float | None = None
    original_eta_minutes: int = Field(default=5, ge=0)
    preferences: AccessibilityPreferences = Field(default_factory=AccessibilityPreferences)


class TransitArrival(BaseModel):
    service_no: str
    eta_minutes: int = Field(ge=0)
    load: str = "UNKNOWN"
    wheelchair_accessible: bool | None = None
    source: str = "mock"
    observed_at: datetime = Field(default_factory=utc_now)


class VisionObservation(BaseModel):
    bus_visible: bool
    service_number: str | None = None
    off_service: bool = False
    confidence: float = Field(default=0.0, ge=0, le=1)
    description: str = ""
    source: str = "mock"
    observed_at: datetime = Field(default_factory=utc_now)


class CaneTelemetry(BaseModel):
    battery_percent: int | None = Field(default=None, ge=0, le=100)
    heading_degrees: float | None = Field(default=None, ge=0, lt=360)
    walking_speed_mps: float | None = Field(default=None, ge=0)
    camera_ready: bool = True
    device_id: str = "cane-demo-01"


class HapticCommand(BaseModel):
    pattern: str
    actuators: list[str]
    priority: str = "normal"
    repeat: int = Field(default=1, ge=1, le=5)


class Evidence(BaseModel):
    transit: TransitArrival
    vision: VisionObservation | None = None


class RoutingHandoff(BaseModel):
    event: str = "REROUTE_REQUIRED"
    reason: str
    expected_bus: str
    current_stop: str
    destination: str
    latest_eta_minutes: int
    accessibility_preferences: AccessibilityPreferences
    excluded_services: list[str] = Field(default_factory=list)


class AgentDecision(BaseModel):
    action: Action
    reason_code: str
    spoken_instruction: str
    haptic_command: HapticCommand
    confidence: float = Field(ge=0, le=1)
    evidence: Evidence
    routing_handoff: RoutingHandoff | None = None
    trace: list[str] = Field(default_factory=list)


class CheckRequest(BaseModel):
    journey: JourneyRequest
    cane_image_base64: str | None = None
    cane_image_media_type: str = "image/jpeg"
    cane_telemetry: CaneTelemetry = Field(default_factory=CaneTelemetry)
    demo_scenario: str | None = Field(
        default=None,
        description="on_time, late, wrong_bus, off_service, no_bus, or conflict",
    )

    @model_validator(mode="after")
    def validate_image_type(self):
        if self.cane_image_media_type not in {"image/jpeg", "image/png", "image/webp"}:
            raise ValueError("cane_image_media_type must be image/jpeg, image/png, or image/webp")
        return self
