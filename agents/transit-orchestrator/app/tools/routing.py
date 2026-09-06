import httpx

from app.config import Settings
from app.models import RoutingHandoff


class RoutingTool:
    def __init__(self, settings: Settings):
        self.settings = settings

    async def request_alternatives(self, handoff: RoutingHandoff) -> dict:
        if not self.settings.routing_agent_url:
            return {
                "status": "queued-demo",
                "recommended_route": "Walk 40 metres via the sheltered tactile path to the adjacent stop; take bus 199.",
                "estimated_minutes": 14,
            }
        async with httpx.AsyncClient(timeout=12) as client:
            response = await client.post(
                f"{self.settings.routing_agent_url.rstrip('/')}/routes/accessible",
                json=handoff.model_dump(mode="json"),
            )
            response.raise_for_status()
            return response.json()

