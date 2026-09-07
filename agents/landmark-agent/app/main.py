import logging

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.agent import LandmarkAgent
from app.config import Settings, get_settings
from app.models import (
    LandmarkClearRequest,
    LandmarkClearResponse,
    LandmarkScanRequest,
    LandmarkScanResponse,
    LandmarkSetRequest,
    LandmarkSetResponse,
)
from app.tools.vision import VisionTool

logger = logging.getLogger(__name__)


def create_app(
    settings: Settings | None = None,
    agent: LandmarkAgent | None = None,
) -> FastAPI:
    settings = settings or get_settings()
    agent = agent or LandmarkAgent(settings, vision=VisionTool(settings))

    app = FastAPI(title=settings.app_name, version="0.1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.state.agent = agent
    app.state.settings = settings

    source = settings.credentials_source
    if source == "settings":
        logger.warning(
            "AWS credentials: using explicit keys from Landmark Agent settings/.env "
            "(not ~/.aws/credentials)"
        )
    else:
        logger.warning(
            "AWS credentials: no AWS_ACCESS_KEY_ID in Landmark settings; "
            "boto3 will use the default chain (~/.aws/credentials or process env)"
        )

    @app.get("/health")
    async def health():
        return {
            "status": "ok",
            "agent": "landmark-agent",
            "demo_mode": settings.demo_mode,
            "credentials_source": settings.credentials_source,
        }

    @app.post("/landmark/set", response_model=LandmarkSetResponse)
    async def landmark_set(request: LandmarkSetRequest):
        agent.set_landmark(request.session_id, request.description)
        return LandmarkSetResponse(session_id=request.session_id)

    @app.post("/landmark/scan", response_model=LandmarkScanResponse)
    async def landmark_scan(request: LandmarkScanRequest):
        try:
            return agent.scan(
                session_id=request.session_id,
                image_base64=request.image_base64,
                image_media_type=request.image_media_type,
                demo_result=request.demo_result,
            )
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"Vision tool failure: {exc}") from exc

    @app.post("/landmark/clear", response_model=LandmarkClearResponse)
    async def landmark_clear(request: LandmarkClearRequest):
        agent.clear_landmark(request.session_id)
        return LandmarkClearResponse()

    return app


app = create_app()
