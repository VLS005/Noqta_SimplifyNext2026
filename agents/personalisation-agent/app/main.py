import logging

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

from app.agent import PersonalisationAgent
from app.aws_credentials import (
    check_aws_credentials,
    http_error_for_store_failure,
    should_check_aws_credentials,
)
from app.config import Settings, get_settings
from app.models import (
    BaselinePaceResponse,
    EndTimerRequest,
    JourneyEndRequest,
    JourneyEndResponse,
    JourneyStartRequest,
    JourneyStartResponse,
    JourneyTickRequest,
    MemoryQueryRequest,
    MemoryQueryResponse,
    MemoryStoreRequest,
    MemoryStoreResponse,
    StartTimerAck,
    StartTimerEventMessage,
)

logger = logging.getLogger(__name__)


def create_app(
    settings: Settings | None = None,
    *,
    sts_client=None,
    check_credentials: bool | None = None,
) -> FastAPI:
    settings = settings or get_settings()
    agent = PersonalisationAgent(settings)
    if settings.create_vector_index_if_missing:
        try:
            status = agent.memory.ensure_vector_index()
            logger.info("ensure_vector_index returned status=%s", status)
        except Exception:
            logger.exception(
                "Vector index create/check failed; server will start anyway. "
                "/memory/query will return empty matches until the index is ACTIVE."
            )

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
    run_check = should_check_aws_credentials() if check_credentials is None else check_credentials
    if run_check:
        app.state.aws_credentials_valid = check_aws_credentials(
            settings.aws_region, sts_client=sts_client
        )
    else:
        app.state.aws_credentials_valid = True

    @app.get("/health")
    async def health():
        return {
            "status": "ok",
            "agent": "personalisation-agent",
            "demo_mode": settings.demo_mode,
            "table": settings.dynamodb_table_name,
        }

    @app.post("/journey/start", response_model=JourneyStartResponse)
    async def journey_start(request: JourneyStartRequest):
        return agent.start_journey(request)

    @app.post("/journey/tick")
    async def journey_tick(request: JourneyTickRequest):
        agent.tick(request.journey_id, request.steps_since_last_tick, request.timestamp)
        return Response(status_code=200)

    @app.post("/journey/end", response_model=JourneyEndResponse)
    async def journey_end(request: JourneyEndRequest):
        try:
            return agent.end_journey(request.journey_id)
        except HTTPException:
            raise
        except Exception as exc:
            raise http_error_for_store_failure(exc, "Pace store failure") from exc

    @app.get("/users/{user_id}/baseline-pace", response_model=BaselinePaceResponse)
    async def baseline_pace(user_id: str):
        try:
            return agent.baseline_pace(user_id)
        except HTTPException:
            raise
        except Exception as exc:
            raise http_error_for_store_failure(exc, "Pace store failure") from exc

    @app.post("/memory/query", response_model=MemoryQueryResponse)
    async def memory_query(request: MemoryQueryRequest):
        try:
            return agent.query_memory(request)
        except HTTPException:
            raise
        except Exception as exc:
            raise http_error_for_store_failure(exc, "Memory query failure") from exc

    @app.post("/memory/store", response_model=MemoryStoreResponse)
    async def memory_store(request: MemoryStoreRequest):
        try:
            return agent.store_memory(request)
        except HTTPException:
            raise
        except Exception as exc:
            raise http_error_for_store_failure(exc, "Memory store failure") from exc

    @app.post("/inbound/start-timer", response_model=StartTimerAck)
    async def inbound_start_timer(message: StartTimerEventMessage):
        return agent.start_timer(message)

    @app.post("/inbound/end-timer", response_model=JourneyEndResponse)
    async def inbound_end_timer(request: EndTimerRequest):
        try:
            return agent.end_timer(request)
        except HTTPException:
            raise
        except Exception as exc:
            raise http_error_for_store_failure(exc, "Pace store failure") from exc

    return app


app = create_app()
