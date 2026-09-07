"""
Sends messages to the Routing Agent over HTTP, matching their REAL API
contract (verified against their backend code, not assumed):

    POST /api/inbound/{agent_type}
    where agent_type is a URL PATH SEGMENT, not a JSON body field.
    Accepted agent_type values (per their code): obstruction, pace,
    landmark, vision_response. There is currently NO "bad_weather" or
    general "disoriented" type on their side.

Configure ROUTING_AGENT_BASE_URL in your .env (see .env.example), e.g.:
    ROUTING_AGENT_BASE_URL=http://localhost:8010

If ROUTING_AGENT_BASE_URL is not set (or is unreachable), falls back to
printing locally instead of crashing.
"""

import os
import requests

from api_schemas.inbound_obstruction import ObstructionReportPayload
from api_schemas.inbound_weather import WeatherAlertPayload
from domain.models import RerouteCause

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

DEFAULT_TIMEOUT_SEC = 3.0

# Only causes with an actual matching agent_type on the Routing Agent's side.
# bad_weather and disoriented are deliberately absent - see module docstring.
_SUPPORTED_AGENT_TYPES = {
    RerouteCause.OBSTRUCTION: "obstruction",
}


def send_obstruction_report(payload: ObstructionReportPayload) -> bool:
    """POSTs to /api/inbound/obstruction. Returns True if delivered (or
    printed locally as a fallback), False if a real base URL was configured
    but the call failed."""
    base_url = os.environ.get("ROUTING_AGENT_BASE_URL")

    if not base_url:
        print(f"[-> ROUTING AGENT] (no ROUTING_AGENT_BASE_URL set, printing locally) {payload.to_dict()}")
        return True

    url = f"{base_url.rstrip('/')}/api/inbound/obstruction"
    try:
        response = requests.post(url, json=payload.to_dict(), timeout=DEFAULT_TIMEOUT_SEC)
        response.raise_for_status()
        print(f"[-> ROUTING AGENT] POST {url} -> HTTP {response.status_code} : {response.json()}")
        return True
    except requests.exceptions.RequestException as e:
        print(f"[-> ROUTING AGENT] FAILED to reach {url}: {e}")
        return False


def send_weather_report(payload: WeatherAlertPayload) -> bool:
    """POSTs to /api/inbound/weather - a PLACEHOLDER endpoint that does not
    exist on the Routing Agent's side yet (their allowed agent_type set is
    {"obstruction", "pace", "landmark", "vision_response"}). Sending this
    today will get a real HTTP 400 back ("Unknown agent_type 'weather'") -
    that failure is expected and reported cleanly, not a bug. This function
    exists so the moment they add a matching handler, this starts working
    with no further code changes needed here."""
    base_url = os.environ.get("ROUTING_AGENT_BASE_URL")

    if not base_url:
        print(f"[-> ROUTING AGENT] (no ROUTING_AGENT_BASE_URL set, printing locally) {payload.to_dict()}")
        return True

    url = f"{base_url.rstrip('/')}/api/inbound/weather"
    try:
        response = requests.post(url, json=payload.to_dict(), timeout=DEFAULT_TIMEOUT_SEC)
        response.raise_for_status()
        print(f"[-> ROUTING AGENT] POST {url} -> HTTP {response.status_code} : {response.json()}")
        return True
    except requests.exceptions.RequestException as e:
        print(
            f"[-> ROUTING AGENT] FAILED to reach {url}: {e} "
            f"(expected until the Routing Agent adds a 'weather' agent_type handler)"
        )
        return False


def report_unsupported_cause(cause: RerouteCause, detail, lat: float, lon: float) -> None:
    """For causes with NO matching endpoint AND no placeholder built yet
    (currently just 'disoriented'). Deliberately does NOT attempt an HTTP
    call - sending to a nonexistent route would just 404 during a live demo.
    Logs locally and clearly, so the gap is visible rather than silently eaten."""
    print(
        f"[ROUTING AGENT - NOT YET SUPPORTED] cause='{cause.value}' has no matching "
        f"Routing Agent endpoint and no placeholder built for it yet. "
        f"detail={detail!r} lat={lat} lon={lon}. Handling locally only - "
        f"raise this with the Routing Agent owner if live rerouting for this "
        f"cause is needed before the demo."
    )
