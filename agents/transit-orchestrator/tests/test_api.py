from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_health_and_demo_page():
    assert client.get("/health").json()["status"] == "ok"
    assert "Transit Orchestrator" in client.get("/demo").text


def test_check_bus_endpoint():
    response = client.post("/agent/check-bus", json={
        "journey": {
            "bus_stop_code": "27211",
            "expected_bus": "199",
            "destination": "Boon Lay MRT",
            "original_eta_minutes": 3,
        },
        "demo_scenario": "off_service",
    })
    assert response.status_code == 200
    assert response.json()["action"] == "REROUTE"
    assert response.json()["routing_handoff"]["reason"] == "BUS_OFF_SERVICE"
    assert response.json()["haptic_command"]["actuators"] == ["CENTER"]
