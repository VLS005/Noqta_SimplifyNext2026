"""Routing Agent compatibility: start-timer / end-timer and PACE#current mirror."""

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import boto3
import pytest
from fastapi.testclient import TestClient
from moto import mock_aws

from app.config import Settings
from app.main import create_app
from app.tools.embeddings import EmbeddingClient
from app.tools.memory_store import MemoryStore
from app.tools.pace_store import PaceStore
from app.tools.units import spm_to_mps

ROUTING_START = {
    "message_id": "11111111-1111-1111-1111-111111111111",
    "message_type": "start_timer_event",
    "session_id": "sess-abc",
    "timestamp": "2026-09-06T12:00:00+00:00",
    "source_agent": "routing_agent",
    "user_id": "do-not-use-this-buggy-field",
    "locked_route_id": "route-locked-1",
    "estimated_duration_s": 0,
    "destination_label": "Jewel Changi",
    "route_distance_m": 0.0,
}


def _table():
    client = boto3.client("dynamodb", region_name="us-east-1")
    client.create_table(
        TableName="DoraDB",
        KeySchema=[
            {"AttributeName": "PK", "KeyType": "HASH"},
            {"AttributeName": "SK", "KeyType": "RANGE"},
        ],
        AttributeDefinitions=[
            {"AttributeName": "PK", "AttributeType": "S"},
            {"AttributeName": "SK", "AttributeType": "S"},
        ],
        BillingMode="PAY_PER_REQUEST",
    )
    return client


def _client(dynamodb) -> TestClient:
    settings = Settings(
        demo_mode=True,
        aws_region="us-east-1",
        dynamodb_table_name="DoraDB",
        create_vector_index_if_missing=False,
        default_stride_length_m=0.7,
        routing_agent_pace_url="http://localhost:8000/api/inbound/pace",
    )
    app = create_app(settings)
    app.state.agent.pace = PaceStore(settings, dynamodb_client=dynamodb)
    app.state.agent.memory = MemoryStore(settings, dynamodb_client=dynamodb)
    app.state.agent.embeddings = EmbeddingClient(settings)
    return TestClient(app)


@mock_aws
def test_start_timer_accepts_routing_payload_without_gps():
    dynamodb = _table()
    client = _client(dynamodb)
    response = client.post("/inbound/start-timer", json=ROUTING_START)
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

    state = client.app.state.agent._journeys[ROUTING_START["message_id"]]
    assert state.user_id == "sess-abc"
    assert state.route_id == "route-locked-1"
    assert state.gps_lat is None
    assert state.gps_lon is None
    expected_ts = datetime(2026, 9, 6, 12, 0, tzinfo=timezone.utc).timestamp()
    assert state.start_time == pytest.approx(expected_ts)


@mock_aws
def test_start_timer_ignores_unknown_extra_fields():
    dynamodb = _table()
    client = _client(dynamodb)
    payload = {**ROUTING_START, "unexpected_field": True, "nested": {"x": 1}}
    assert client.post("/inbound/start-timer", json=payload).status_code == 200


@mock_aws
@patch("app.agent.httpx.Client")
def test_start_timer_then_end_timer_writes_routing_pace_item(mock_httpx_cls):
    mock_httpx_cls.return_value.__enter__.return_value.post.return_value = MagicMock(status_code=200)
    dynamodb = _table()
    client = _client(dynamodb)

    assert client.post("/inbound/start-timer", json=ROUTING_START).status_code == 200
    start_ts = datetime(2026, 9, 6, 12, 0, tzinfo=timezone.utc).timestamp()
    tick = client.post(
        "/journey/tick",
        json={
            "journey_id": ROUTING_START["message_id"],
            "steps_since_last_tick": 90,
            "timestamp": start_ts + 60,
        },
    )
    assert tick.status_code == 200

    ended = client.post("/inbound/end-timer", json={"session_id": "sess-abc"})
    assert ended.status_code == 200
    assert ended.json()["journey_pace_spm"] == pytest.approx(90.0)

    item = dynamodb.get_item(
        TableName="DoraDB",
        Key={"PK": {"S": "PACE#sess-abc"}, "SK": {"S": "PACE#current"}},
    )["Item"]
    assert float(item["average_pace_mps"]["N"]) == pytest.approx(spm_to_mps(90.0, 0.7))
    assert item["user_id"]["S"] == "sess-abc"
    assert item["source"]["S"] == "personalisation_agent"

    post = mock_httpx_cls.return_value.__enter__.return_value.post
    post.assert_called_once()
    args, kwargs = post.call_args
    assert args[0] == "http://localhost:8000/api/inbound/pace"
    body = kwargs["json"]
    assert body["message_type"] == "pace_update"
    assert body["current_pace_mps"] == pytest.approx(spm_to_mps(90.0, 0.7))
    assert body["deviation_factor"] == 1.0
    assert body["user_id"] == "sess-abc"
    assert body["session_id"] == "sess-abc"


@mock_aws
def test_end_timer_unknown_session_is_404():
    dynamodb = _table()
    client = _client(dynamodb)
    response = client.post("/inbound/end-timer", json={"session_id": "missing"})
    assert response.status_code == 404
