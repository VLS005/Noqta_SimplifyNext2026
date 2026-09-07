"""Pace store tests against a moto-backed DoraDB table."""

import boto3
import pytest
from moto import mock_aws

from app.config import Settings
from app.agent import PersonalisationAgent
from app.models import JourneyStartRequest, GpsPoint
from app.tools.pace_store import PaceStore, pace_sk, user_pk


def _settings() -> Settings:
    return Settings(
        demo_mode=True,
        aws_region="us-east-1",
        dynamodb_table_name="DoraDB",
        create_vector_index_if_missing=False,
    )


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


@mock_aws
def test_write_query_and_profile_round_trip():
    dynamodb = _table()
    store = PaceStore(_settings(), dynamodb_client=dynamodb)
    store.write_journey_pace("u1", 100.0, steps=80, elapsed_minutes=1.0, pace_spm=80.0, route_id="r1")
    store.write_journey_pace("u1", 200.0, steps=100, elapsed_minutes=1.0, pace_spm=100.0, route_id="r1")
    paces = store.recent_paces("u1", limit=10)
    assert paces == [100.0, 80.0]
    store.upsert_profile("u1", 90.0)
    assert store.get_profile("u1") == 90.0
    assert store.get_profile("unknown") is None

    item = dynamodb.get_item(
        TableName="DoraDB",
        Key={"PK": {"S": user_pk("u1")}, "SK": {"S": pace_sk(200.0)}},
    )["Item"]
    assert item["route_id"]["S"] == "r1"
    assert item["SK"]["S"].startswith("PACE#")


@mock_aws
def test_end_journey_rolling_average_and_zero_elapsed_guard():
    dynamodb = _table()
    settings = _settings()
    agent = PersonalisationAgent(settings, pace_store=PaceStore(settings, dynamodb_client=dynamodb))

    first = agent.start_journey(
        JourneyStartRequest(
            user_id="walker",
            route_id="n1",
            gps=GpsPoint(lat=1.29, lon=103.85),
            timestamp=0.0,
        )
    )
    agent.tick(first.journey_id, 90, timestamp=60.0)
    result = agent.end_journey(first.journey_id)
    assert result.journey_pace_spm == pytest.approx(90.0)
    assert result.rolling_avg_pace_spm == pytest.approx(90.0)

    second = agent.start_journey(
        JourneyStartRequest(
            user_id="walker",
            route_id="n1",
            gps=GpsPoint(lat=1.29, lon=103.85),
            timestamp=1_000.0,
        )
    )
    agent.tick(second.journey_id, 110, timestamp=1_060.0)
    result = agent.end_journey(second.journey_id)
    assert result.journey_pace_spm == pytest.approx(110.0)
    assert result.rolling_avg_pace_spm == pytest.approx(100.0)

    instant = agent.start_journey(
        JourneyStartRequest(
            user_id="walker",
            route_id="n1",
            gps=GpsPoint(lat=1.29, lon=103.85),
            timestamp=2_000.0,
        )
    )
    # No ticks: elapsed floors at 0.1 minutes so we never divide by zero.
    result = agent.end_journey(instant.journey_id)
    assert result.journey_pace_spm == pytest.approx(0.0)
