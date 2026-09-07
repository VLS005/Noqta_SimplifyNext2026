from app.config import Settings
from app.models import VisionVerdict
from app.tools.vision import guidance_from, parse_verdict


def test_parse_valid_json():
    verdict = parse_verdict(
        'Here you go: {"visible": true, "position": "center", "distance": "far", "confidence": "medium"}'
    )
    assert verdict.visible is True
    assert verdict.position == "center"
    assert verdict.distance == "far"
    assert verdict.confidence == "medium"


def test_parse_malformed_returns_not_visible():
    verdict = parse_verdict("not json at all")
    assert verdict.visible is False
    assert verdict.position is None
    assert verdict.confidence == "low"
    assert "not json" in verdict.raw_text


def test_parse_broken_json_object():
    verdict = parse_verdict("{visible: true, nope}")
    assert verdict.visible is False
    assert verdict.confidence == "low"


def test_guidance_not_visible():
    assert guidance_from(VisionVerdict(visible=False)) == "Not visible yet, keep moving"


def test_guidance_left_and_ahead():
    assert "left" in guidance_from(VisionVerdict(visible=True, position="left")).lower()
    assert "10 steps" in guidance_from(
        VisionVerdict(visible=True, position="center", distance="near")
    )


def test_explicit_settings_credentials_are_detected():
    empty = Settings(demo_mode=True)
    assert empty.credentials_source == "default_chain"
    filled = Settings(
        demo_mode=True,
        aws_access_key_id="AKIAEXAMPLE",
        aws_secret_access_key="secret",
        aws_session_token="token",
    )
    assert filled.credentials_source == "settings"
    kwargs = filled.boto3_credential_kwargs()
    assert kwargs["aws_access_key_id"] == "AKIAEXAMPLE"
    assert kwargs["aws_session_token"] == "token"
