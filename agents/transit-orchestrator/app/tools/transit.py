from datetime import datetime, timezone
import httpx

from app.config import Settings
from app.models import TransitArrival


DEMO_ETAS = {
    "on_time": 1,
    "late": 18,
    "wrong_bus": 2,
    "off_service": 1,
    "no_bus": 12,
    "conflict": 0,
}


class TransitTool:
    """Retrieves the expected service ETA from LTA DataMall or deterministic demo data."""

    def __init__(self, settings: Settings):
        self.settings = settings

    async def get_arrival(
        self, bus_stop_code: str, service_no: str, demo_scenario: str | None = None
    ) -> TransitArrival:
        if self.settings.demo_mode or not self.settings.lta_account_key:
            eta = DEMO_ETAS.get(demo_scenario or "on_time", 5)
            return TransitArrival(service_no=service_no, eta_minutes=eta, source="demo")

        headers = {"AccountKey": self.settings.lta_account_key, "accept": "application/json"}
        params = {"BusStopCode": bus_stop_code, "ServiceNo": service_no}
        async with httpx.AsyncClient(timeout=8) as client:
            response = await client.get(
                "https://datamall2.mytransport.sg/ltaodataservice/v3/BusArrival",
                headers=headers,
                params=params,
            )
            response.raise_for_status()
            services = response.json().get("Services", [])

        if not services:
            return TransitArrival(service_no=service_no, eta_minutes=99, source="lta")
        bus = services[0].get("NextBus", {})
        estimated = bus.get("EstimatedArrival")
        eta = 99
        if estimated:
            arrival = datetime.fromisoformat(estimated)
            now = datetime.now(arrival.tzinfo or timezone.utc)
            eta = max(0, round((arrival - now).total_seconds() / 60))
        return TransitArrival(
            service_no=service_no,
            eta_minutes=eta,
            load=bus.get("Load", "UNKNOWN"),
            wheelchair_accessible=bus.get("Feature") == "WAB",
            source="lta",
        )

