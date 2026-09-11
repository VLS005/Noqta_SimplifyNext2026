import asyncio
import logging
import json
from typing import Dict, Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from config.settings import AgentConfig
from domain.models import GPSPoint, SensorReading
from agent import StateSafetyAgent
import uvicorn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Walking Identification Agent")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, list[WebSocket]] = {}

    async def connect(self, session_id: str, websocket: WebSocket):
        await websocket.accept()
        if session_id not in self.active_connections:
            self.active_connections[session_id] = []
        self.active_connections[session_id].append(websocket)

    def disconnect(self, session_id: str, websocket: WebSocket):
        if session_id in self.active_connections:
            if websocket in self.active_connections[session_id]:
                self.active_connections[session_id].remove(websocket)
            if not self.active_connections[session_id]:
                del self.active_connections[session_id]

    async def broadcast(self, session_id: str, message: dict):
        if session_id in self.active_connections:
            for connection in self.active_connections[session_id]:
                try:
                    await connection.send_json(message)
                except Exception as e:
                    logger.error(f"Failed to send WS message: {e}")

manager = ConnectionManager()
active_agents: Dict[str, StateSafetyAgent] = {}
config = AgentConfig(poll_interval_sec=0.1, alert_cooldown_sec=3.0)

class StartJourneyReq(BaseModel):
    session_id: str
    waypoint_id: str | None = None

class SensorTickReq(BaseModel):
    session_id: str
    lat: float
    lon: float
    heading_deg: float = 0.0
    steps_per_minute: float = 95.0
    timestamp: float

@app.post("/journey/start")
async def start_journey(req: StartJourneyReq):
    logger.info(f"Starting journey for {req.session_id}")
    agent = StateSafetyAgent(config=config, baseline_pace_spm=95)
    # Hook the WebSocket broadcast into the agent so it can send alerts
    agent.ws_broadcast = lambda msg: asyncio.create_task(manager.broadcast(req.session_id, msg))
    agent.start_journey(session_id=req.session_id, waypoint_id=req.waypoint_id)
    active_agents[req.session_id] = agent
    return {"status": "started", "session_id": req.session_id}

@app.post("/journey/sensor-tick")
async def sensor_tick(req: SensorTickReq):
    if req.session_id not in active_agents:
        return {"status": "ignored", "reason": "session not active"}
    
    agent = active_agents[req.session_id]
    reading = SensorReading(
        position=GPSPoint(lat=req.lat, lon=req.lon, timestamp=req.timestamp),
        heading_deg=req.heading_deg,
        steps_per_minute=req.steps_per_minute,
        timestamp=req.timestamp
    )
    # Ingest the reading
    agent.ingest(reading)
    return {"status": "ok"}

@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    await manager.connect(session_id, websocket)
    try:
        while True:
            data = await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(session_id, websocket)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8003)
