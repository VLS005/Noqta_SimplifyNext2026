"""DynamoDB obstacle memory + native vector search on the shared DoraDB table.

PK/SK scheme (must not collide with other agents):
  PK = MEMORY#<geohash>   SK = OBSTACLE#<timestamp>

Vector index (created once via CREATE_VECTOR_INDEX_IF_MISSING or README steps):
  IndexName:        obstacle-memory-index
  VectorAttribute:  embedding
  Dimensions:       1024  (Titan Text Embeddings V2 default)
  DistanceFunction: COSINE  (score 0 = identical, 2 = opposite)

moto does not implement DynamoDB SearchVectors (GA Aug 2026). Tests should
mock the boto3 client rather than relying on moto for vector search.
"""

from __future__ import annotations

import json
import logging
import time

import boto3
from botocore.exceptions import ClientError

from app.aws_credentials import is_aws_credential_error
from app.config import Settings
from app.models import MemoryMatch

logger = logging.getLogger(__name__)

_BASE32 = "0123456789bcdefghjkmnpqrstuvwxyz"


def encode_geohash(lat: float, lon: float, precision: int = 7) -> str:
    """Encode lat/lon to a geohash so nearby memories share a partition."""
    lat_range = [-90.0, 90.0]
    lon_range = [-180.0, 180.0]
    bits: list[int] = []
    even = True
    while len(bits) < precision * 5:
        if even:
            mid = (lon_range[0] + lon_range[1]) / 2.0
            if lon >= mid:
                bits.append(1)
                lon_range[0] = mid
            else:
                bits.append(0)
                lon_range[1] = mid
        else:
            mid = (lat_range[0] + lat_range[1]) / 2.0
            if lat >= mid:
                bits.append(1)
                lat_range[0] = mid
            else:
                bits.append(0)
                lat_range[1] = mid
        even = not even

    chars: list[str] = []
    for i in range(0, precision * 5, 5):
        idx = 0
        for bit in bits[i : i + 5]:
            idx = (idx << 1) | bit
        chars.append(_BASE32[idx])
    return "".join(chars)


def memory_pk(geohash: str) -> str:
    return f"MEMORY#{geohash}"


def obstacle_sk(timestamp: float) -> str:
    return f"OBSTACLE#{timestamp}"


def embedding_attr(vector: list[float]) -> dict:
    return {"L": [{"N": str(value)} for value in vector]}


def search_vector_attr(vector: list[float]) -> list[dict]:
    # SearchVector is a plain list of N values, NOT wrapped in an L type.
    return [{"N": str(value)} for value in vector]


def cosine_score_to_confidence(score: float) -> float:
    """COSINE scores run 0 (identical) .. 2 (opposite). Map onto 0..1 confidence."""
    clamped = min(max(float(score), 0.0), 2.0)
    return 1.0 - (clamped / 2.0)


class MemoryStore:
    def __init__(self, settings: Settings, dynamodb_client=None):
        self.settings = settings
        self.dynamodb = dynamodb_client or boto3.client(
            "dynamodb", region_name=settings.aws_region
        )

    def store(
        self,
        lat: float,
        lon: float,
        day_of_week: str,
        time_of_day: str,
        description: str,
        embedding: list[float],
        timestamp: float | None = None,
    ) -> str:
        ts = time.time() if timestamp is None else timestamp
        geohash = encode_geohash(lat, lon, self.settings.geohash_precision)
        pk = memory_pk(geohash)
        sk = obstacle_sk(ts)
        self.dynamodb.put_item(
            TableName=self.settings.dynamodb_table_name,
            Item={
                "PK": {"S": pk},
                "SK": {"S": sk},
                "lat": {"N": str(lat)},
                "lon": {"N": str(lon)},
                "day_of_week": {"S": day_of_week},
                "time_of_day": {"S": time_of_day},
                "description": {"S": description},
                "geohash": {"S": geohash},
                "embedding": embedding_attr(embedding),
            },
        )
        return f"{pk}|{sk}"

    def search(
        self,
        embedding: list[float],
        lat: float,
        lon: float,
        top_k: int | None = None,
    ) -> list[MemoryMatch]:
        geohash = encode_geohash(lat, lon, self.settings.geohash_precision)
        pk = memory_pk(geohash)
        try:
            search_vectors = getattr(self.dynamodb, "search_vectors", None)
            if search_vectors is None:
                logger.warning("Installed boto3 does not expose search_vectors; returning no matches")
                return []
            response = search_vectors(
                TableName=self.settings.dynamodb_table_name,
                IndexName=self.settings.vector_index_name,
                SearchVector=search_vector_attr(embedding),
                TopK=top_k or self.settings.memory_top_k,
                SearchConditionExpression="#pk = :pk",
                ExpressionAttributeNames={"#pk": "PK"},
                ExpressionAttributeValues={":pk": {"S": pk}},
                ProjectionExpression="description, lat, lon",
            )
        except Exception as exc:
            if is_aws_credential_error(exc):
                raise
            # Missing index, old boto3, validation during backfill, network, etc.
            logger.warning("Vector search failed; returning no matches: %s", exc)
            return []

        matches: list[MemoryMatch] = []
        for result in response.get("SearchResults", []):
            item = result.get("Item") or {}
            description = item.get("description", {}).get("S")
            lat_raw = item.get("lat", {}).get("N")
            lon_raw = item.get("lon", {}).get("N")
            if description is None or lat_raw is None or lon_raw is None:
                continue
            matches.append(
                MemoryMatch(
                    description=description,
                    confidence=cosine_score_to_confidence(result.get("Score", 2.0)),
                    lat=float(lat_raw),
                    lon=float(lon_raw),
                )
            )
        return matches[: self.settings.memory_top_k]

    def describe_vector_index(self) -> dict | None:
        table = self.dynamodb.describe_table(TableName=self.settings.dynamodb_table_name)["Table"]
        for index in table.get("VectorIndexes", []):
            if index.get("IndexName") == self.settings.vector_index_name:
                return index
        return None

    def ensure_vector_index(self) -> str:
        """Create the COSINE vector index on DoraDB if it is missing.

        Idempotent: if the index already exists in ACTIVE or CREATING (or any
        other reported status), skip UpdateTable so a second run does not crash.
        Returns the current IndexStatus string (e.g. ACTIVE, CREATING).
        """
        table_name = self.settings.dynamodb_table_name
        index_name = self.settings.vector_index_name
        current = self.describe_vector_index()
        if current is not None:
            status = current.get("IndexStatus") or "UNKNOWN"
            logger.info(
                "Vector index %s already present on %s with status %s (Backfilling=%s) — skipping create",
                index_name,
                table_name,
                status,
                current.get("Backfilling", False),
            )
            return status

        logger.info(
            "Creating vector index %s on %s (COSINE, dimensions=%s, attribute=embedding)",
            index_name,
            table_name,
            self.settings.embedding_dimensions,
        )
        # SearchSchema HASH/INLINE_FILTER attrs must be in this UpdateTable's
        # AttributeDefinitions (same rule as a GSI). AWS docs:
        # "One element in SearchSchema is not defined in attribute definitions".
        #
        # embedding is NOT listed here. boto3 VectorAttributeDefinition has only
        # AttributeName (no type), and ScalarAttributeType is enum S|N|B — there
        # is no vector AttributeType. Vectors are stored as L of N on PutItem and
        # named via VectorAttribute, not AttributeDefinitions.
        attribute_definitions = [
            {"AttributeName": "PK", "AttributeType": "S"},
        ]
        vector_index_create = {
            "IndexName": index_name,
            "VectorAttribute": {"AttributeName": "embedding"},
            # HASH on PK scopes SearchVectors to MEMORY#<geohash>.
            "SearchSchema": [
                {
                    "AttributeName": "PK",
                    "SearchSchemaElementType": "HASH",
                }
            ],
            "Projection": {"ProjectionType": "ALL"},
            # 1024 matches Titan Text Embeddings V2 default output.
            "Dimensions": self.settings.embedding_dimensions,
            "DistanceFunction": "COSINE",
        }
        payload = {
            "TableName": table_name,
            "AttributeDefinitions": attribute_definitions,
            "VectorIndexUpdates": [{"Create": vector_index_create}],
        }
        print("UpdateTable payload:", json.dumps(payload, indent=2), flush=True)
        logger.info("UpdateTable payload: %s", json.dumps(payload))
        try:
            self.dynamodb.update_table(**payload)
        except ClientError as exc:
            error = exc.response.get("Error", {})
            code = error.get("Code", "")
            # A concurrent create, or the index appearing between describe and update.
            if code in {"ResourceInUseException", "LimitExceededException", "ValidationException"}:
                logger.info(
                    "Vector index create skipped (%s: %s); re-checking DescribeTable",
                    code,
                    error.get("Message", exc),
                )
                current = self.describe_vector_index()
                if current is not None:
                    return current.get("IndexStatus") or "UNKNOWN"
            raise

        current = self.describe_vector_index()
        return (current or {}).get("IndexStatus") or "CREATING"

    def wait_until_vector_index_active(
        self, poll_seconds: float = 5.0, timeout_seconds: float = 900.0
    ) -> dict:
        """Poll DescribeTable until the vector index is ACTIVE and not backfilling."""
        deadline = time.time() + timeout_seconds
        while True:
            info = self.describe_vector_index()
            if info is None:
                status, backfilling = "MISSING", False
            else:
                status = info.get("IndexStatus") or "UNKNOWN"
                backfilling = bool(info.get("Backfilling", False))
            logger.info(
                "Vector index %s status: %s | Backfilling: %s",
                self.settings.vector_index_name,
                status,
                backfilling,
            )
            print(
                f"Index status: {status} | Backfilling: {backfilling}",
                flush=True,
            )
            if status == "ACTIVE" and not backfilling:
                print(
                    "Vector index ACTIVE -- safe to stop and restart with "
                    "CREATE_VECTOR_INDEX_IF_MISSING=false",
                    flush=True,
                )
                return info
            if time.time() >= deadline:
                raise TimeoutError(
                    f"Timed out after {timeout_seconds:.0f}s waiting for "
                    f"{self.settings.vector_index_name} to become ACTIVE "
                    f"(last status={status}, Backfilling={backfilling})"
                )
            time.sleep(poll_seconds)
