from unittest.mock import MagicMock

from fastapi.testclient import TestClient

from app.agent import LandmarkAgent
from app.config import Settings
from app.main import create_app
from app.tools.vision import VisionTool

TINY_JPEG_B64 = "aGVsbG8="  # b"hello"


def _client(demo_mode: bool = True, vision: VisionTool | None = None) -> TestClient:
    settings = Settings(demo_mode=demo_mode)
    agent = LandmarkAgent(settings, vision=vision or VisionTool(settings))
    return TestClient(create_app(settings, agent))


def test_health():
    body = _client().get("/health").json()
    assert body["status"] == "ok"
    assert body["agent"] == "landmark-agent"
    assert body["demo_mode"] is True
    assert body["credentials_source"] == "default_chain"


def test_set_stores_and_scan_finds_session():
    client = _client()
    response = client.post(
        "/landmark/set",
        json={"session_id": "sess-1", "description": "red pillar by the entrance"},
    )
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "session_id": "sess-1"}

    scan = client.post(
        "/landmark/scan",
        json={"session_id": "sess-1", "image_base64": TINY_JPEG_B64, "demo_result": "found"},
    )
    assert scan.status_code == 200
    assert scan.json()["visible"] is True


def test_scan_without_set_returns_404():
    response = _client().post(
        "/landmark/scan",
        json={"session_id": "missing", "image_base64": TINY_JPEG_B64},
    )
    assert response.status_code == 404
    assert "/landmark/set" in response.json()["detail"]


def test_scan_demo_mode_does_not_call_bedrock():
    vision = MagicMock()
    client = _client(demo_mode=True, vision=vision)
    client.post("/landmark/set", json={"session_id": "s", "description": "red pillar"})
    response = client.post(
        "/landmark/scan",
        json={"session_id": "s", "image_base64": TINY_JPEG_B64, "demo_result": "not_found"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["visible"] is False
    assert body["guidance"]
    assert body["confidence"]
    vision.inspect.assert_not_called()


def test_scan_live_calls_bedrock_and_parses_json():
    settings = Settings(demo_mode=False, bedrock_vision_model_id="apac.amazon.nova-pro-v1:0")
    bedrock = MagicMock()
    bedrock.converse.return_value = {
        "output": {
            "message": {
                "content": [{
                    "text": '{"visible": true, "position": "left", "distance": "near", "confidence": "high"}',
                }],
            },
        },
    }
    vision = VisionTool(settings, bedrock_client=bedrock)
    client = _client(demo_mode=False, vision=vision)
    client.post("/landmark/set", json={"session_id": "s", "description": "red pillar"})
    response = client.post(
        "/landmark/scan",
        json={"session_id": "s", "image_base64": TINY_JPEG_B64, "image_media_type": "image/jpeg"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["visible"] is True
    assert body["confidence"] == "high"
    assert "left" in body["guidance"].lower()

    bedrock.converse.assert_called_once()
    kwargs = bedrock.converse.call_args.kwargs
    assert kwargs["modelId"] == "apac.amazon.nova-pro-v1:0"
    content = kwargs["messages"][0]["content"]
    assert content[0]["image"]["format"] == "jpeg"
    assert content[0]["image"]["source"]["bytes"] == b"hello"
    assert "red pillar" in content[1]["text"]


def test_scan_malformed_model_response_is_not_visible():
    settings = Settings(demo_mode=False)
    bedrock = MagicMock()
    bedrock.converse.return_value = {
        "output": {"message": {"content": [{"text": "I cannot tell from this photo."}]}},
    }
    client = _client(demo_mode=False, vision=VisionTool(settings, bedrock_client=bedrock))
    client.post("/landmark/set", json={"session_id": "s", "description": "red pillar"})
    response = client.post(
        "/landmark/scan",
        json={"session_id": "s", "image_base64": TINY_JPEG_B64},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["visible"] is False
    assert body["guidance"] == "Not visible yet, keep moving"
    assert body["confidence"] == "low"


def test_clear_removes_session():
    client = _client()
    client.post("/landmark/set", json={"session_id": "s", "description": "red pillar"})
    cleared = client.post("/landmark/clear", json={"session_id": "s"})
    assert cleared.status_code == 200
    assert cleared.json() == {"status": "cleared"}
    missing = client.post(
        "/landmark/scan",
        json={"session_id": "s", "image_base64": TINY_JPEG_B64},
    )
    assert missing.status_code == 404
