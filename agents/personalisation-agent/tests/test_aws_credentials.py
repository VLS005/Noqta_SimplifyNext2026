"""Expired/invalid AWS credentials: startup warning and clear 502 responses."""

from unittest.mock import MagicMock

import boto3
from botocore.exceptions import ClientError
from fastapi.testclient import TestClient
from moto import mock_aws

from app.aws_credentials import (
    AWS_CREDENTIALS_INVALID_DETAIL,
    STARTUP_CREDENTIAL_WARNING,
    check_aws_credentials,
)
from app.config import Settings
from app.main import create_app
from app.tools.embeddings import EmbeddingClient
from app.tools.memory_store import MemoryStore
from app.tools.pace_store import PaceStore


def _expired_token(operation: str = "PutItem") -> ClientError:
    return ClientError(
        {
            "Error": {
                "Code": "ExpiredTokenException",
                "Message": "The security token included in the request is expired",
            }
        },
        operation,
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


def _client(dynamodb) -> TestClient:
    settings = Settings(
        demo_mode=True,
        aws_region="us-east-1",
        dynamodb_table_name="DoraDB",
        create_vector_index_if_missing=False,
    )
    app = create_app(settings)
    app.state.agent.pace = PaceStore(settings, dynamodb_client=dynamodb)
    app.state.agent.memory = MemoryStore(settings, dynamodb_client=dynamodb)
    app.state.agent.embeddings = EmbeddingClient(settings)
    return TestClient(app)


def test_check_aws_credentials_expired_token_warns_and_returns_false(caplog, capsys):
    sts = MagicMock()
    sts.get_caller_identity.side_effect = _expired_token("GetCallerIdentity")
    with caplog.at_level("WARNING"):
        valid = check_aws_credentials("us-east-1", sts_client=sts)
    assert valid is False
    assert STARTUP_CREDENTIAL_WARNING in caplog.text
    assert STARTUP_CREDENTIAL_WARNING in capsys.readouterr().out


def test_startup_stores_aws_credentials_valid_false_without_crashing():
    sts = MagicMock()
    sts.get_caller_identity.side_effect = _expired_token("GetCallerIdentity")
    settings = Settings(
        demo_mode=True,
        aws_region="us-east-1",
        dynamodb_table_name="DoraDB",
        create_vector_index_if_missing=False,
    )
    app = create_app(settings, sts_client=sts, check_credentials=True)
    assert app.state.aws_credentials_valid is False


@mock_aws
def test_journey_end_expired_token_returns_clear_502():
    dynamodb = _table()
    client = _client(dynamodb)

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
        json={"journey_id": journey_id, "steps_since_last_tick": 60, "timestamp": 1_060.0},
    )
    assert tick.status_code == 200

    dynamodb.put_item = MagicMock(side_effect=_expired_token("PutItem"))
    client.app.state.agent.pace.dynamodb = dynamodb

    ended = client.post("/journey/end", json={"journey_id": journey_id})
    assert ended.status_code == 502
    assert ended.json()["detail"] == AWS_CREDENTIALS_INVALID_DETAIL
    assert "Refresh your access keys" in ended.json()["detail"]
