"""HTTP contract tests. Pace paths use moto; memory paths mock vector/embedding calls."""

import boto3
import pytest
from fastapi.testclient import TestClient
from moto import mock_aws

from app.config import Settings
from app.main import create_app
from app.tools.embeddings import EmbeddingClient
from app.tools.memory_store import MemoryStore
from app.tools.pace_store import PaceStore


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


def _client(dynamodb, demo_mode: bool = True) -> TestClient:
    settings = Settings(
        demo_mode=demo_mode,
        aws_region="us-east-1",
        dynamodb_table_name="DoraDB",
        create_vector_index_if_missing=False,
    )
    app = create_app(settings)
    app.state.agent.pace = PaceStore(settings, dynamodb_client=dynamodb)
    app.state.agent.memory = MemoryStore(settings, dynamodb_client=dynamodb)
    app.state.agent.embeddings = EmbeddingClient(settings)
    return TestClient(app)


@mock_aws
def test_health():
    dynamodb = _table()
    client = _client(dynamodb)
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["agent"] == "personalisation-agent"
    assert body["table"] == "DoraDB"


@mock_aws
def test_journey_lifecycle_and_baseline():
    dynamodb = _table()
    client = _client(dynamodb)

    default = client.get("/users/demo-user/baseline-pace")
    assert default.status_code == 200
    assert default.json()["is_default"] is True
    assert default.json()["baseline_pace_spm"] == 95.0

    started = client.post(
        "/journey/start",
        json={
            "user_id": "demo-user",
            "route_id": "route-1",
            "gps": {"lat": 1.3, "lon": 103.8},
            "timestamp": 1_000.0,
        },
    )
    assert started.status_code == 200
    journey_id = started.json()["journey_id"]

    tick = client.post(
        "/journey/tick",
        json={"journey_id": journey_id, "steps_since_last_tick": 190, "timestamp": 1_120.0},
    )
    assert tick.status_code == 200
    assert tick.content in {b"", b"null"}

    ended = client.post("/journey/end", json={"journey_id": journey_id})
    assert ended.status_code == 200
    body = ended.json()
    assert body["journey_pace_spm"] == pytest.approx(95.0)
    assert body["rolling_avg_pace_spm"] == pytest.approx(95.0)

    baseline = client.get("/users/demo-user/baseline-pace")
    assert baseline.json()["is_default"] is False
    assert baseline.json()["baseline_pace_spm"] == pytest.approx(95.0)
    assert baseline.json()["user_id"] == "demo-user"


@mock_aws
def test_tick_unknown_journey_is_404():
    dynamodb = _table()
    client = _client(dynamodb)
    response = client.post(
        "/journey/tick",
        json={"journey_id": "missing", "steps_since_last_tick": 10, "timestamp": 1.0},
    )
    assert response.status_code == 404


@mock_aws
def test_memory_query_returns_empty_when_search_unavailable():
    dynamodb = _table()
    client = _client(dynamodb)
    response = client.post(
        "/memory/query",
        json={"lat": 1.3, "lon": 103.8, "day_of_week": "Monday", "time_of_day": "morning"},
    )
    assert response.status_code == 200
    assert response.json() == {"matches": []}
