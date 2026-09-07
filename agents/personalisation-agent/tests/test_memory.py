"""Memory store tests.

moto does not implement DynamoDB SearchVectors / vector indexes (the feature
went GA in August 2026). These tests mock boto3 client calls directly instead
of relying on moto for the vector path. PutItem for obstacle records is still
exercised via moto.
"""

from unittest.mock import MagicMock

import boto3
from moto import mock_aws

from app.config import Settings
from app.agent import PersonalisationAgent
from app.models import MemoryQueryRequest, MemoryStoreRequest
from app.tools.embeddings import EmbeddingClient
from app.tools.memory_store import (
    MemoryStore,
    cosine_score_to_confidence,
    encode_geohash,
    memory_pk,
    search_vector_attr,
)


def _settings(**overrides) -> Settings:
    values = dict(
        demo_mode=True,
        aws_region="us-east-1",
        dynamodb_table_name="DoraDB",
        create_vector_index_if_missing=False,
        embedding_dimensions=8,
    )
    values.update(overrides)
    return Settings(**values)


def test_geohash_precision_and_stability():
    first = encode_geohash(1.3000, 103.8000, precision=7)
    second = encode_geohash(1.3000, 103.8000, precision=7)
    nearby = encode_geohash(1.30001, 103.80001, precision=7)
    far = encode_geohash(1.4000, 103.9000, precision=7)
    assert first == second
    assert len(first) == 7
    assert nearby == first
    assert far != first
    assert memory_pk(first).startswith("MEMORY#")


def test_cosine_confidence_mapping():
    assert cosine_score_to_confidence(0.0) == 1.0
    assert cosine_score_to_confidence(2.0) == 0.0
    assert cosine_score_to_confidence(1.0) == 0.5


def test_search_returns_matches_from_search_vectors():
    settings = _settings()
    fake = MagicMock()
    fake.search_vectors.return_value = {
        "SearchResults": [
            {
                "Score": 0.2,
                "Item": {
                    "description": {"S": "Uneven paving on the left"},
                    "lat": {"N": "1.3"},
                    "lon": {"N": "103.8"},
                },
            }
        ]
    }
    store = MemoryStore(settings, dynamodb_client=fake)
    matches = store.search([0.1] * 8, lat=1.3, lon=103.8, top_k=5)
    assert len(matches) == 1
    assert matches[0].description == "Uneven paving on the left"
    assert matches[0].confidence == cosine_score_to_confidence(0.2)
    kwargs = fake.search_vectors.call_args.kwargs
    assert kwargs["IndexName"] == "obstacle-memory-index"
    assert kwargs["SearchVector"] == search_vector_attr([0.1] * 8)
    assert kwargs["ExpressionAttributeValues"][":pk"]["S"].startswith("MEMORY#")


def test_search_returns_empty_when_index_missing():
    settings = _settings()
    fake = MagicMock()
    fake.search_vectors.side_effect = Exception("The table does not have the specified index")
    store = MemoryStore(settings, dynamodb_client=fake)
    assert store.search([0.1] * 8, lat=1.3, lon=103.8) == []


def test_ensure_vector_index_skips_when_creating_or_active():
    settings = _settings()
    fake = MagicMock()
    fake.describe_table.return_value = {
        "Table": {
            "VectorIndexes": [
                {
                    "IndexName": "obstacle-memory-index",
                    "IndexStatus": "CREATING",
                    "Backfilling": True,
                }
            ]
        }
    }
    store = MemoryStore(settings, dynamodb_client=fake)
    assert store.ensure_vector_index() == "CREATING"
    fake.update_table.assert_not_called()

    fake.describe_table.return_value = {
        "Table": {
            "VectorIndexes": [
                {"IndexName": "obstacle-memory-index", "IndexStatus": "ACTIVE"}
            ]
        }
    }
    assert store.ensure_vector_index() == "ACTIVE"
    fake.update_table.assert_not_called()


def test_ensure_vector_index_sends_search_schema_attribute_definitions():
    settings = _settings()
    fake = MagicMock()
    fake.describe_table.return_value = {"Table": {"VectorIndexes": []}}
    fake.update_table.return_value = {}
    store = MemoryStore(settings, dynamodb_client=fake)
    store.ensure_vector_index()
    kwargs = fake.update_table.call_args.kwargs
    assert kwargs["AttributeDefinitions"] == [
        {"AttributeName": "PK", "AttributeType": "S"},
    ]
    create = kwargs["VectorIndexUpdates"][0]["Create"]
    assert create["SearchSchema"] == [
        {"AttributeName": "PK", "SearchSchemaElementType": "HASH"},
    ]
    assert create["VectorAttribute"] == {"AttributeName": "embedding"}
    assert "embedding" not in {
        item["AttributeName"] for item in kwargs["AttributeDefinitions"]
    }


def test_search_returns_empty_when_api_absent():
    settings = _settings()
    fake = MagicMock(spec=["put_item", "describe_table"])
    store = MemoryStore(settings, dynamodb_client=fake)
    assert store.search([0.1] * 8, lat=1.3, lon=103.8) == []


@mock_aws
def test_store_writes_memory_keys_and_embedding():
    dynamodb = boto3.client("dynamodb", region_name="us-east-1")
    dynamodb.create_table(
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
    settings = _settings()
    store = MemoryStore(settings, dynamodb_client=dynamodb)
    memory_id = store.store(
        lat=1.3,
        lon=103.8,
        day_of_week="Monday",
        time_of_day="morning",
        description="Low hanging branch",
        embedding=[0.1, 0.2],
        timestamp=1_700_000_000.0,
    )
    geohash = encode_geohash(1.3, 103.8, 7)
    assert memory_id == f"MEMORY#{geohash}|OBSTACLE#1700000000.0"
    item = dynamodb.get_item(
        TableName="DoraDB",
        Key={"PK": {"S": f"MEMORY#{geohash}"}, "SK": {"S": "OBSTACLE#1700000000.0"}},
    )["Item"]
    assert item["description"]["S"] == "Low hanging branch"
    assert item["embedding"]["L"][0]["N"] == "0.1"


def test_agent_query_and_store_use_embeddings():
    settings = _settings()
    fake_ddb = MagicMock()
    fake_ddb.search_vectors.return_value = {"SearchResults": []}
    embeddings = MagicMock(spec=EmbeddingClient)
    embeddings.embed.return_value = [0.5] * 8
    agent = PersonalisationAgent(
        settings,
        memory_store=MemoryStore(settings, dynamodb_client=fake_ddb),
        embeddings=embeddings,
    )
    query = agent.query_memory(
        MemoryQueryRequest(lat=1.3, lon=103.8, day_of_week="Tuesday", time_of_day="evening")
    )
    assert query.matches == []
    embeddings.embed.assert_called()

    fake_ddb.put_item.return_value = {}
    stored = agent.store_memory(
        MemoryStoreRequest(
            lat=1.3,
            lon=103.8,
            day_of_week="Tuesday",
            time_of_day="evening",
            description="Open drain without railing",
        )
    )
    assert stored.stored is True
    assert stored.memory_id.startswith("MEMORY#")
    fake_ddb.put_item.assert_called_once()
