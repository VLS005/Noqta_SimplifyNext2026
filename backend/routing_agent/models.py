"""
All data models for the Routing Agent in one file.
- Domain objects (routes, waypoints, accessibility), message contracts, and user profiles.
"""
from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field


# ══════════════════════════════════════════════════════════════════════════════
# ENUMS
# ══════════════════════════════════════════════════════════════════════════════

class LightingQuality(str, Enum):
    BRIGHT = "bright"
    ADEQUATE = "adequate"
    DIM = "dim"
    UNKNOWN = "unknown"


class RouteSelectionState(str, Enum):
    """State machine states for the route-lock flow."""
    UNLOCKED = "unlocked"
    CANDIDATES_PRESENTED = "candidates_presented"
    LOCKED = "locked"


class MessageType(str, Enum):
    # Inbound
    ROUTE_REQUEST = "route_request"
    VOICE_ROUTE_SELECTION = "voice_route_selection"
    OBSTRUCTION_REPORT = "obstruction_report"
    PACE_UPDATE = "pace_update"
    LANDMARK_CONFIRMATION = "landmark_confirmation"
    VISION_RESPONSE = "vision_response"
    # Outbound
    ROUTE_CHOICE_PROMPT = "route_choice_prompt"
    START_TIMER_EVENT = "start_timer_event"
    VISION_REQUEST = "vision_request"
    ROUTE_LOCKED_NOTIFICATION = "route_locked_notification"


# ══════════════════════════════════════════════════════════════════════════════
# DOMAIN OBJECTS — routes, waypoints, scoring
# ══════════════════════════════════════════════════════════════════════════════

class Waypoint(BaseModel):
    """A single navigation checkpoint along a route."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    sequence: int = Field(ge=0, description="0-indexed position in the route")
    lat: float
    lon: float
    label: str = Field(
        description="Human-readable landmark label used in micro-instructions"
    )
    fine_motor_required: bool = Field(
        default=False,
        description=(
            "If True, on-demand vision can be triggered when the user is "
            "within vision_proximity_gate_m of this waypoint"
        ),
    )
    distance_to_next_m: float | None = Field(
        default=None,
        description="Straight-line metres to the next waypoint (None for final waypoint)",
    )
    bearing_to_next_deg: float | None = Field(
        default=None,
        description="Compass bearing (0–360) to the next waypoint",
    )


class AccessibilityScore(BaseModel):
    """Computed accessibility rating for a route."""
    step_free: bool = True
    tactile_paving: bool = False
    lighting_quality: LightingQuality = LightingQuality.UNKNOWN
    crowding_estimate: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="0 = empty, 1 = very crowded",
    )
    obstruction_risk: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="0 = no obstruction risk, 1 = high risk",
    )
    composite_score: float = Field(
        default=0.5, ge=0.0, le=1.0,
        description="Weighted composite — higher is more accessible",
    )


class RoutePlan(BaseModel):
    """A candidate route with full waypoint, accessibility, and instruction data."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str
    label: str = Field(description="Short human label, e.g. 'Step-Free via Station Road'")
    origin_label: str
    destination_label: str
    waypoints: list[Waypoint]
    accessibility_score: AccessibilityScore
    estimated_duration_s: int = Field(
        default=0, description="Personalised ETA in seconds (set by ETAAgent)"
    )
    distance_m: float
    rank: int = Field(
        default=0,
        description="Accessibility rank assigned by AccessibilityAgent. rank=1 is most accessible/safe.",
    )
    micro_instructions: list[str] = Field(
        default_factory=list,
        description="Translated blind-friendly micro-instructions (set by InstructionParsingAgent)",
    )
    raw_instructions: list[str] = Field(
        default_factory=list,
        description="Standard navigation text (input to InstructionParsingAgent)",
    )
    created_at: datetime = Field(default_factory=datetime.utcnow)
    is_stub: bool = Field(
        default=True,
        description=(
            "True for demo/stub data. Set to False when a real mapping API is integrated. "
            "During demo, real-world camera + API data will replace stub content."
        ),
    )


class LockedRoute(BaseModel):
    """
    Immutable snapshot of a RoutePlan after the user confirms via voice.
    Once locked, the route cannot change — only a new session can produce a new route.
    """
    route_plan: RoutePlan
    locked_at: datetime = Field(default_factory=datetime.utcnow)
    session_id: str
    confirmed_by: str = Field(
        default="voice",
        description="Always 'voice' — route selection is always through voice input",
    )


# ══════════════════════════════════════════════════════════════════════════════
# USER PROFILE & SESSION
# ══════════════════════════════════════════════════════════════════════════════

class MobilityWeights(BaseModel):
    """
    Accessibility scoring weights for the 5 metrics.
    Must sum to 1.0. Computed dynamically from MobilityProfile flags
    by AccessibilityAgent, or overridden manually.
    """
    step_free: float = Field(default=0.35, ge=0.0, le=1.0)
    tactile: float = Field(default=0.20, ge=0.0, le=1.0)
    lighting: float = Field(default=0.20, ge=0.0, le=1.0)
    crowding: float = Field(default=0.15, ge=0.0, le=1.0)
    obstruction: float = Field(default=0.10, ge=0.0, le=1.0)


class MobilityProfile(BaseModel):
    """
    User's accessibility and mobility preferences.
    Used to personalise route scoring, ETA estimation, and micro-instructions.
    """
    prefers_step_free: bool = True
    preferred_pace_mps: float = Field(
        default=0.8, ge=0.1, le=3.0,
        description="Preferred walking speed in metres per second (0.8 m/s ≈ comfortable stroll)",
    )
    prefers_tactile_paving: bool = True
    vision_level: Literal["blind", "low_vision"] = "low_vision"
    hearing_level: Literal["full", "hard_of_hearing"] = "full"
    preferred_lighting: Literal["any", "adequate", "bright"] = "bright"
    landmark_preference: Literal["verbal", "tactile", "both"] = "verbal"
    instruction_verbosity: Literal["brief", "standard", "detailed"] = "standard"
    avoids_crowded_areas: bool = False
    max_crowding_tolerance: float = Field(
        default=0.7, ge=0.0, le=1.0,
        description="Routes with crowding_estimate above this threshold are deprioritised",
    )
    mobility_weights: MobilityWeights | None = Field(
        default=None,
        description=(
            "Override accessibility scoring weights. If None, AccessibilityAgent "
            "computes weights dynamically from the profile flags above."
        ),
    )


class UserSession(BaseModel):
    """Active navigation session for a user."""
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    profile: MobilityProfile = Field(default_factory=MobilityProfile)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_active_at: datetime = Field(default_factory=datetime.utcnow)
    active: bool = True

    def touch(self) -> None:
        """Update last_active_at to now."""
        self.last_active_at = datetime.utcnow()


# ══════════════════════════════════════════════════════════════════════════════
# MESSAGES — base + inbound + outbound
# ══════════════════════════════════════════════════════════════════════════════

class BaseMessage(BaseModel):
    message_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    message_type: MessageType
    session_id: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    source_agent: str = "unknown"


# ── Inbound ──────────────────────────────────────────────────────────────────

class RouteRequestMessage(BaseMessage):
    """Received when a user requests navigation from A → B."""
    message_type: Literal[MessageType.ROUTE_REQUEST] = MessageType.ROUTE_REQUEST
    source_agent: str = "ui_agent"
    user_id: str
    origin_label: str
    destination_label: str
    origin_lat: float | None = None
    origin_lon: float | None = None
    destination_lat: float | None = None
    destination_lon: float | None = None


class VoiceRouteSelectionMessage(BaseMessage):
    """Received when the user speaks their route choice (e.g. 'take route 1')."""
    message_type: Literal[MessageType.VOICE_ROUTE_SELECTION] = MessageType.VOICE_ROUTE_SELECTION
    source_agent: str = "voice_agent"
    raw_voice_transcript: str
    resolved_route_id: str | None = Field(
        default=None,
        description="Populated by VoiceAgent after parsing the transcript",
    )


class ObstructionReportMessage(BaseMessage):
    """Stub — from Obstruction Agent. Routing Agent logs and no-ops."""
    message_type: Literal[MessageType.OBSTRUCTION_REPORT] = MessageType.OBSTRUCTION_REPORT
    source_agent: str = "obstruction_agent"
    waypoint_id: str
    severity: Literal["low", "medium", "high"]
    description: str
    lat: float | None = None
    lon: float | None = None


class PaceUpdateMessage(BaseMessage):
    """Stub — from Pace Tracking Agent. Received for awareness only."""
    message_type: Literal[MessageType.PACE_UPDATE] = MessageType.PACE_UPDATE
    source_agent: str = "pace_agent"
    current_pace_mps: float = Field(gt=0)
    deviation_factor: float = Field(
        description="actual_pace / expected_pace — 1.0 = on pace, <1.0 = slower"
    )


class LandmarkConfirmationMessage(BaseMessage):
    """Stub — from Landmark Verification Agent."""
    message_type: Literal[MessageType.LANDMARK_CONFIRMATION] = MessageType.LANDMARK_CONFIRMATION
    source_agent: str = "landmark_agent"
    waypoint_id: str
    confirmed: bool
    landmark_label: str
    confidence: float = Field(ge=0.0, le=1.0, default=1.0)


class VisionResponseMessage(BaseMessage):
    """On-demand response from Vision Agent (only when user is within 5 m of fine-motor waypoint)."""
    message_type: Literal[MessageType.VISION_RESPONSE] = MessageType.VISION_RESPONSE
    source_agent: str = "vision_agent"
    waypoint_id: str
    object_detected: str = Field(description="e.g. 'door_handle', 'elevator_button'")
    confidence: float = Field(ge=0.0, le=1.0)
    guidance_text: str = Field(
        description="e.g. 'Door handle is 30 centimetres to your right'"
    )
    image_captured_at: datetime | None = None


# ── Outbound ─────────────────────────────────────────────────────────────────

class RouteOption(BaseModel):
    """Summary of one candidate route presented to the user."""
    route_id: str
    index: int = Field(description="1-based index used in voice commands (say 'route 1')")
    label: str = Field(description="e.g. 'Step-Free via Station Road'")
    rank: int = Field(description="Accessibility rank (1 = most accessible/safe)")
    duration_s: int
    distance_m: float
    accessibility_composite: float = Field(ge=0.0, le=1.0)
    first_micro_instruction: str = Field(
        description="First micro-instruction read aloud to preview the route"
    )
    highlights: list[str] = Field(
        default_factory=list,
        description="Key features e.g. ['Step-free throughout', 'Tactile paving']",
    )


class RouteChoicePromptMessage(BaseMessage):
    """Sent to TTS/UI Agent to read route options aloud to the user."""
    message_type: Literal[MessageType.ROUTE_CHOICE_PROMPT] = MessageType.ROUTE_CHOICE_PROMPT
    source_agent: str = "routing_agent"
    route_options: list[RouteOption]
    prompt_text: str = Field(
        description="Full spoken prompt, e.g. 'I found 3 routes. Say route 1, 2, or 3.'"
    )


class StartTimerEventMessage(BaseMessage):
    """Sent to Personalization Agent once a route is locked via voice."""
    message_type: Literal[MessageType.START_TIMER_EVENT] = MessageType.START_TIMER_EVENT
    source_agent: str = "routing_agent"
    user_id: str
    locked_route_id: str
    estimated_duration_s: int
    destination_label: str
    route_distance_m: float


class VisionRequestMessage(BaseMessage):
    """On-demand vision request — sent only when user is ≤ 5 m from a fine-motor waypoint."""
    message_type: Literal[MessageType.VISION_REQUEST] = MessageType.VISION_REQUEST
    source_agent: str = "routing_agent"
    waypoint_id: str
    user_distance_m: float = Field(
        le=5.0, description="Must be ≤ 5 m — gate enforced by VisionAgent"
    )
    context: str = Field(description="e.g. 'Locate door handle', 'Find lift button'")


class LockedRouteNotificationMessage(BaseMessage):
    """Broadcast once a route is locked so the UI can transition to navigation mode."""
    message_type: Literal[MessageType.ROUTE_LOCKED_NOTIFICATION] = MessageType.ROUTE_LOCKED_NOTIFICATION
    source_agent: str = "routing_agent"
    locked_route_id: str
    first_micro_instruction: str
    total_waypoints: int
    estimated_duration_s: int


# ── Discriminated union for inbound message parsing ──────────────────────────

InboundMessage = Annotated[
    RouteRequestMessage
    | VoiceRouteSelectionMessage
    | ObstructionReportMessage
    | PaceUpdateMessage
    | LandmarkConfirmationMessage
    | VisionResponseMessage,
    Field(discriminator="message_type"),
]
