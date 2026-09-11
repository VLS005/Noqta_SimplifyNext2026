"""Core Personalisation Agent orchestration, separate from the HTTP layer."""

from __future__ import annotations

import logging
import threading
import uuid
from dataclasses import dataclass, field

import httpx
from fastapi import HTTPException

from app.aws_credentials import is_aws_credential_error
from app.config import Settings
from app.models import (
    BaselinePaceResponse,
    EndTimerRequest,
    JourneyEndResponse,
    JourneyStartRequest,
    JourneyStartResponse,
    MemoryQueryRequest,
    MemoryQueryResponse,
    MemoryStoreRequest,
    MemoryStoreResponse,
    StartTimerAck,
    StartTimerEventMessage,
)
from app.tools.embeddings import EmbeddingClient
from app.tools.memory_store import MemoryStore
from app.tools.pace_store import PaceStore
from app.tools.units import spm_to_mps

logger = logging.getLogger(__name__)


@dataclass
class JourneyState:
    user_id: str
    route_id: str
    start_time: float
    last_timestamp: float
    steps: int = 0
    gps_lat: float | None = None
    gps_lon: float | None = None
    session_id: str | None = None
    extra: dict = field(default_factory=dict)


def query_text(lat: float, lon: float, day_of_week: str, time_of_day: str) -> str:
    return f"Location near {lat},{lon} on {day_of_week} during {time_of_day}"


def store_text(
    description: str,
    lat: float,
    lon: float,
    day_of_week: str,
    time_of_day: str,
) -> str:
    return f"{description}. {query_text(lat, lon, day_of_week, time_of_day)}"


class PersonalisationAgent:
    def __init__(
        self,
        settings: Settings,
        pace_store: PaceStore | None = None,
        memory_store: MemoryStore | None = None,
        embeddings: EmbeddingClient | None = None,
    ):
        self.settings = settings
        self.pace = pace_store or PaceStore(settings)
        self.memory = memory_store or MemoryStore(settings)
        self.embeddings = embeddings or EmbeddingClient(settings)
        self._journeys: dict[str, JourneyState] = {}
        self._session_to_journey: dict[str, str] = {}
        self._lock = threading.Lock()

    def start_journey(self, request: JourneyStartRequest) -> JourneyStartResponse:
        journey_id = str(uuid.uuid4())
        state = JourneyState(
            user_id=request.user_id,
            route_id=request.route_id,
            start_time=request.timestamp,
            last_timestamp=request.timestamp,
            gps_lat=request.gps.lat,
            gps_lon=request.gps.lon,
        )
        with self._lock:
            self._journeys[journey_id] = state
        return JourneyStartResponse(journey_id=journey_id)

    def start_timer(self, message: StartTimerEventMessage) -> StartTimerAck:
        # Routing currently puts session_id into StartTimerEventMessage.user_id
        # (locked.route_plan.session_id). session_id is the stable identifier
        # until that bug is fixed.
        # TODO: switch to message.user_id once Lakshmi's Routing Agent sends the real user id.
        user_id = message.session_id
        start_time = message.timestamp.timestamp()
        state = JourneyState(
            user_id=user_id,
            route_id=message.locked_route_id,
            start_time=start_time,
            last_timestamp=start_time,
            gps_lat=None,
            gps_lon=None,
            session_id=message.session_id,
        )
        journey_id = message.message_id
        with self._lock:
            self._journeys[journey_id] = state
            self._session_to_journey[message.session_id] = journey_id
        return StartTimerAck(status="ok")

    def tick(self, journey_id: str, steps_since_last_tick: int, timestamp: float) -> None:
        with self._lock:
            state = self._journeys.get(journey_id)
            if state is None:
                raise HTTPException(status_code=404, detail="journey_id not found")
            state.steps += steps_since_last_tick
            state.last_timestamp = timestamp

    def end_timer(self, request: EndTimerRequest) -> JourneyEndResponse:
        with self._lock:
            journey_id = self._session_to_journey.get(request.session_id)
        if not journey_id:
            raise HTTPException(status_code=404, detail="session_id not found")
        return self.end_journey(journey_id)

    def end_journey(self, journey_id: str) -> JourneyEndResponse:
        with self._lock:
            state = self._journeys.get(journey_id)
        if state is None:
            raise HTTPException(status_code=404, detail="journey_id not found")

        elapsed_minutes = max(
            (state.last_timestamp - state.start_time) / 60.0,
            self.settings.min_elapsed_minutes,
        )
        journey_pace_spm = state.steps / elapsed_minutes
        end_timestamp = state.last_timestamp

        self.pace.write_journey_pace(
            user_id=state.user_id,
            timestamp=end_timestamp,
            steps=state.steps,
            elapsed_minutes=elapsed_minutes,
            pace_spm=journey_pace_spm,
            route_id=state.route_id,
        )
        recent = self.pace.recent_paces(state.user_id, self.settings.pace_window)
        rolling_avg = sum(recent) / len(recent) if recent else journey_pace_spm
        self.pace.upsert_profile(state.user_id, rolling_avg)
        self._publish_routing_pace(state, rolling_avg)
        with self._lock:
            self._journeys.pop(journey_id, None)
            if state.session_id:
                self._session_to_journey.pop(state.session_id, None)
        return JourneyEndResponse(
            journey_pace_spm=journey_pace_spm,
            rolling_avg_pace_spm=rolling_avg,
        )

    def _publish_routing_pace(self, state: JourneyState, rolling_avg_spm: float) -> None:
        pace_mps = spm_to_mps(rolling_avg_spm, self.settings.default_stride_length_m)
        try:
            self.pace.upsert_routing_pace_current(state.user_id, pace_mps)
        except Exception as exc:
            if is_aws_credential_error(exc):
                raise
            logger.exception("Failed to write Routing-compatible PACE#current item")
        try:
            payload = {
                "message_type": "pace_update",
                "source_agent": "personalisation_agent",
                "current_pace_mps": pace_mps,
                "deviation_factor": 1.0,
                "user_id": state.user_id,
                "session_id": state.session_id or state.user_id,
            }
            with httpx.Client(timeout=1.0) as client:
                client.post(self.settings.routing_agent_pace_url, json=payload)
        except Exception:
            logger.warning(
                "Routing Agent pace inbound unreachable at %s — continuing",
                self.settings.routing_agent_pace_url,
            )

    def baseline_pace(self, user_id: str) -> BaselinePaceResponse:
        stored = self.pace.get_profile(user_id)
        if stored is None:
            return BaselinePaceResponse(
                user_id=user_id,
                baseline_pace_spm=self.settings.default_baseline_pace_spm,
                is_default=True,
            )
        return BaselinePaceResponse(
            user_id=user_id,
            baseline_pace_spm=stored,
            is_default=False,
        )

    def query_memory(self, request: MemoryQueryRequest) -> MemoryQueryResponse:
        text = query_text(request.lat, request.lon, request.day_of_week, request.time_of_day)
        try:
            vector = self.embeddings.embed(text)
        except Exception as exc:
            if is_aws_credential_error(exc):
                raise
            return MemoryQueryResponse(matches=[])
        matches = self.memory.search(vector, request.lat, request.lon)
        return MemoryQueryResponse(matches=matches)

    def store_memory(self, request: MemoryStoreRequest) -> MemoryStoreResponse:
        text = store_text(
            request.description,
            request.lat,
            request.lon,
            request.day_of_week,
            request.time_of_day,
        )
        vector = self.embeddings.embed(text)
        memory_id = self.memory.store(
            lat=request.lat,
            lon=request.lon,
            day_of_week=request.day_of_week,
            time_of_day=request.time_of_day,
            description=request.description,
            embedding=vector,
        )
        return MemoryStoreResponse(stored=True, memory_id=memory_id)
