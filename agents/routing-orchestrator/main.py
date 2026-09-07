"""
main.py — Single entry point for the SimplifyNext backend.
Run: uvicorn main:app --reload --port 8000
Or:  docker-compose up

This file is the ONLY integration point for the frontend.
All /api/* routes delegate to the RoutingAgent orchestrator.
WebSocket at /ws/{session_id} delivers real-time events.
"""
from __future__ import annotations

import asyncio
import logging
import sys
from functools import lru_cache
from typing import Annotated, Any
from contextlib import asynccontextmanager

import structlog
from fastapi import Depends, FastAPI, HTTPException, WebSocket, WebSocketDisconnect, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from backend.routing_agent.routing_agent import RoutingAgent
from backend.routing_agent.config import get_settings
from backend.routing_agent.models import MobilityProfile

from backend.routing_agent.bedrock_client import get_bedrock_client
from backend.shared.vector_store import get_vector_store

# ─── Logging ──────────────────────────────────────────────────────────────────
settings = get_settings()
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    stream=sys.stdout,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
structlog.configure(
    wrapper_class=structlog.make_filtering_bound_logger(
        getattr(logging, settings.log_level.upper(), logging.INFO)
    )
)
logger = logging.getLogger(__name__)


# ─── Dependency injection ─────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def get_routing_agent() -> RoutingAgent:
    return RoutingAgent(store=get_vector_store(), bedrock=get_bedrock_client())


RoutingAgentDep = Annotated[RoutingAgent, Depends(get_routing_agent)]


# ─── WebSocket connection manager ─────────────────────────────────────────────

class ConnectionManager:
    """Manages active WebSocket connections per session."""

    def __init__(self) -> None:
        self._connections: dict[str, list[WebSocket]] = {}

    async def connect(self, session_id: str, ws: WebSocket) -> None:
        await ws.accept()
        self._connections.setdefault(session_id, []).append(ws)

    def disconnect(self, session_id: str, ws: WebSocket) -> None:
        conns = self._connections.get(session_id, [])
        if ws in conns:
            conns.remove(ws)

    async def broadcast(self, session_id: str, data: dict) -> None:
        conns = self._connections.get(session_id, [])
        disconnected = []
        for ws in conns:
            try:
                await ws.send_json(data)
            except Exception:
                disconnected.append(ws)
        for ws in disconnected:
            self.disconnect(session_id, ws)


ws_manager = ConnectionManager()


# ─── FastAPI app ──────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("SimplifyNext Orchestrator starting — frontend integration at /api/*")
    yield
    logger.info("SimplifyNext Orchestrator shutting down")

app = FastAPI(
    title="Routing Agent",
    description="Single entry point for the SimplifyNext multi-agent navigation system.",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # TODO: restrict to frontend domain in production
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Request/Response schemas ─────────────────────────────────────────────────




class RouteRequestReq(BaseModel):
    origin_label: str
    destination_label: str
    user_id: str = "anonymous"
    origin_lat: float | None = None
    origin_lon: float | None = None
    destination_lat: float | None = None
    destination_lon: float | None = None




class VisionAssistReq(BaseModel):
    waypoint_id: str
    user_distance_m: float
    context: str = "Locate the fine-motor control"


# ─── Health ───────────────────────────────────────────────────────────────────

@app.get("/health", tags=["Meta"])
async def health() -> dict:
    return {"status": "ok", "service": "simplify-next-orchestrator", "version": "0.1.0"}





# ─── Route planning ───────────────────────────────────────────────────────────

@app.post("/api/route-request", tags=["Routing"])
async def route_request(body: RouteRequestReq, agent: RoutingAgentDep) -> dict:
    """Triggers the full routing pipeline and returns route options."""
    from backend.routing_agent.models import RouteRequestMessage
    import uuid
    session_id = str(uuid.uuid4())
    print(f"\n[API] POST /api/route-request")
    print(f"[API] Origin: {body.origin_label} | Destination: {body.destination_label}")
    print(f"[API] User: {body.user_id} | Session: {session_id}")
    try:
        msg = RouteRequestMessage(
            session_id=session_id, user_id=body.user_id,
            origin_label=body.origin_label, destination_label=body.destination_label,
            origin_lat=body.origin_lat, origin_lon=body.origin_lon,
            destination_lat=body.destination_lat, destination_lon=body.destination_lon,
        )
        locked, notification = await agent.handle_route_request(msg)
        result = {
            "status": "locked",
            "session_id": session_id,
            "locked_route_id": locked.route_plan.id,
            "route_label": locked.route_plan.label,
            "first_instruction": notification.first_micro_instruction,
            "estimated_duration_s": locked.route_plan.estimated_duration_s,
            "total_waypoints": notification.total_waypoints,
        }
        print(f"\n[API] Route request SUCCESS — auto-locked best route: '{result['route_label']}'")
        await ws_manager.broadcast(session_id, {"event": "route_locked", "data": result})
        return result
    except Exception as exc:
        print(f"[API] Route request FAILED: {exc}")
        logger.exception("route-request failed")
        raise HTTPException(status_code=500, detail=str(exc))





# ─── On-demand vision ─────────────────────────────────────────────────────────

@app.post("/api/sessions/{session_id}/vision-assist", tags=["Vision"])
async def vision_assist(session_id: str, body: VisionAssistReq, agent: RoutingAgentDep) -> dict:
    result = await agent.request_vision_assist(
        session_id=session_id, waypoint_id=body.waypoint_id,
        user_distance_m=body.user_distance_m, context=body.context,
    )
    if result.get("status") == "gate_rejected":
        raise HTTPException(status_code=403, detail=result["reason"])
    return result


# ─── Inbound from external agents ────────────────────────────────────────────

@app.post("/api/inbound/{agent_type}", tags=["Inbound"])
async def inbound(agent_type: str, payload: dict[str, Any], agent: RoutingAgentDep) -> dict:
    allowed = {"obstruction", "pace", "landmark", "vision_response"}
    if agent_type not in allowed:
        raise HTTPException(status_code=400, detail=f"Unknown agent_type '{agent_type}'")
    return await agent.handle_inbound(agent_type, payload)


# ─── WebSocket event stream ───────────────────────────────────────────────────

@app.websocket("/ws/{session_id}")
async def websocket_events(session_id: str, websocket: WebSocket) -> None:
    """Real-time events: route_choice_prompt, route_locked, keepalive."""
    await ws_manager.connect(session_id, websocket)
    try:
        while True:
            data = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
            if data == "ping":
                await websocket.send_text("pong")
    except asyncio.TimeoutError:
        try:
            await websocket.send_json({"event": "keepalive"})
        except Exception:
            pass
    except WebSocketDisconnect:
        pass
    finally:
        ws_manager.disconnect(session_id, websocket)


# ─── Lifecycle ────────────────────────────────────────────────────────────────
# Moved to lifespan context manager above

