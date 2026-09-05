from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pathlib import Path

from app.config import get_settings
from app.models import AgentDecision, CheckRequest
from app.orchestrator import TransitOrchestrator


settings = get_settings()
app = FastAPI(title=settings.app_name, version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)
agent = TransitOrchestrator(settings)
STATIC_DIR = Path(__file__).parent / "static"


@app.get("/demo", include_in_schema=False)
async def demo():
    return FileResponse(STATIC_DIR / "demo.html")


@app.get("/health")
async def health():
    return {"status": "ok", "agent": "transit-orchestrator", "demo_mode": settings.demo_mode}


@app.post("/agent/check-bus", response_model=AgentDecision)
async def check_bus(request: CheckRequest):
    try:
        return await agent.run(request)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Agent tool failure: {exc}") from exc
