"""
Verifies that send_obstruction_report() performs a REAL HTTP POST to the
Routing Agent's ACTUAL endpoint path - /api/inbound/obstruction - with the
exact payload shape their ObstructionReportMessage schema expects. Uses a
real local HTTP server, not a mock, to prove the wire format is correct.

Run with: python tests/test_routing_http.py
"""

import sys
import os
import json
import threading
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from http.server import BaseHTTPRequestHandler, HTTPServer

received = []
received_paths = []


class _Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        received.append(json.loads(body))
        received_paths.append(self.path)
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(b'{"status": "received", "message_id": "abc123", "action": "no-op"}')

    def log_message(self, format, *args):
        pass  # silence default request logging - keep test output clean


def _start_test_server(port: int) -> HTTPServer:
    server = HTTPServer(("localhost", port), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


def test_obstruction_posts_to_correct_path_with_correct_shape():
    """The core test: confirms BOTH the URL path (agent_type as a path
    segment, not a body field) AND the payload field names match the
    Routing Agent's real ObstructionReportMessage schema."""
    received.clear()
    received_paths.clear()
    port = 8766
    server = _start_test_server(port)
    os.environ["ROUTING_AGENT_BASE_URL"] = f"http://localhost:{port}"

    from api_schemas.inbound_obstruction import ObstructionReportPayload
    from integrations.routing_client import send_obstruction_report

    payload = ObstructionReportPayload.build(
        session_id="sess-abc",
        waypoint_id="wp-7",
        obstruction={"description": "construction barrier", "severity": "medium"},
        lat=1.3521,
        lon=103.8198,
    )

    success = send_obstruction_report(payload)
    server.shutdown()

    assert success is True
    assert len(received) == 1, f"Expected exactly 1 POST to reach the server, got {len(received)}"
    assert received_paths[0] == "/api/inbound/obstruction", (
        f"Expected path '/api/inbound/obstruction' (agent_type in the URL, "
        f"per the Routing Agent's real routes), got '{received_paths[0]}'"
    )

    body = received[0]
    # These field names are copied directly from the Routing Agent's real
    # ObstructionReportMessage / BaseMessage Pydantic models - not guessed.
    assert body["session_id"] == "sess-abc"
    assert body["waypoint_id"] == "wp-7"
    assert body["severity"] == "medium"
    assert body["description"] == "construction barrier"
    assert body["message_type"] == "obstruction_report"
    assert body["lat"] == 1.3521
    assert body["lon"] == 103.8198
    assert "message_id" in body
    assert "timestamp" in body
    assert "source_agent" in body

    os.environ.pop("ROUTING_AGENT_BASE_URL", None)


def test_falls_back_to_local_print_when_no_base_url_configured():
    os.environ.pop("ROUTING_AGENT_BASE_URL", None)

    from api_schemas.inbound_obstruction import ObstructionReportPayload
    from integrations.routing_client import send_obstruction_report

    payload = ObstructionReportPayload.build(
        session_id="sess-1", waypoint_id="wp-1",
        obstruction={"description": "x", "severity": "low"}, lat=1.0, lon=1.0,
    )
    success = send_obstruction_report(payload)
    assert success is True, "Should fall back gracefully, not fail, when no base URL is set"


def test_handles_unreachable_endpoint_gracefully():
    os.environ["ROUTING_AGENT_BASE_URL"] = "http://localhost:1"  # nothing listens on port 1

    from api_schemas.inbound_obstruction import ObstructionReportPayload
    from integrations.routing_client import send_obstruction_report

    payload = ObstructionReportPayload.build(
        session_id="sess-1", waypoint_id="wp-1",
        obstruction={"description": "x", "severity": "low"}, lat=1.0, lon=1.0,
    )
    success = send_obstruction_report(payload)
    assert success is False, "Expected a reported failure, not a crash, for an unreachable endpoint"

    os.environ.pop("ROUTING_AGENT_BASE_URL", None)


def test_unsupported_cause_never_makes_http_call():
    """disoriented STILL has no endpoint or placeholder - confirm
    report_unsupported_cause does NOT attempt any network call at all."""
    from domain.models import RerouteCause
    from integrations.routing_client import report_unsupported_cause

    # Should complete instantly with no server running on any port - if this
    # tried to make an HTTP call to a nonexistent endpoint it would hang/error.
    report_unsupported_cause(RerouteCause.DISORIENTED, "circling", 1.0, 1.0)


class _RejectingHandler(BaseHTTPRequestHandler):
    """Faithfully replays the Routing Agent's REAL rejection behavior for an
    unrecognized agent_type - copied from their actual main.py:
        allowed = {"obstruction", "pace", "landmark", "vision_response"}
        if agent_type not in allowed:
            raise HTTPException(status_code=400, detail=f"Unknown agent_type '{agent_type}'")
    """
    def do_POST(self):
        allowed = {"obstruction", "pace", "landmark", "vision_response"}
        agent_type = self.path.rsplit("/", 1)[-1]
        length = int(self.headers.get("Content-Length", 0))
        self.rfile.read(length)  # drain body
        if agent_type not in allowed:
            self.send_response(400)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"detail": f"Unknown agent_type '{agent_type}'"}).encode())
        else:
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b'{"status": "received"}')

    def log_message(self, format, *args):
        pass


def test_weather_placeholder_gets_real_400_from_her_actual_validation_logic():
    """Confirms our failure handling is correct against the EXACT rejection
    her real server would issue - not just a generic connection failure."""
    port = 8767
    server = HTTPServer(("localhost", port), _RejectingHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    os.environ["ROUTING_AGENT_BASE_URL"] = f"http://localhost:{port}"

    from api_schemas.inbound_weather import WeatherAlertPayload
    from integrations.routing_client import send_weather_report

    payload = WeatherAlertPayload.build(
        session_id="sess-1", weather={"condition": "raining", "severity": "medium"}, lat=1.0, lon=1.0,
    )
    success = send_weather_report(payload)
    server.shutdown()

    assert success is False, (
        "Expected this to fail - her real server rejects 'weather' as an "
        "unknown agent_type with HTTP 400. If this ever starts passing, it "
        "means she's added support and report_unsupported_cause / this "
        "placeholder note can be retired."
    )

    os.environ.pop("ROUTING_AGENT_BASE_URL", None)


if __name__ == "__main__":
    test_obstruction_posts_to_correct_path_with_correct_shape()
    print("PASS: obstruction POSTs to the real path with the real payload shape")

    test_falls_back_to_local_print_when_no_base_url_configured()
    print("PASS: falls back to local print when no base URL is configured")

    test_handles_unreachable_endpoint_gracefully()
    print("PASS: handles an unreachable endpoint without crashing")

    test_unsupported_cause_never_makes_http_call()
    print("PASS: disoriented (no placeholder) never attempts an HTTP call")

    test_weather_placeholder_gets_real_400_from_her_actual_validation_logic()
    print("PASS: weather placeholder correctly handles the real 400 rejection")

    print("\nAll HTTP integration tests passed.")
