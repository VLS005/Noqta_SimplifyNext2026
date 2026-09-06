"""
shared/vector_store.py
======================
Centralised persistence layer — shared by ALL agents in this project.

Backends
--------
* DynamoDB (default, primary)  — key-value store with single-table design
* S3 Vectors (placeholder)     — vector similarity search, implement when AWS SDK available

Single-table DynamoDB key schema
---------------------------------
PK  (partition)  = "<ENTITY_TYPE>#<primary_id>"   e.g.  "SESSION#user_001"
SK  (sort)       = "<ENTITY_TYPE>#<secondary_id>"  e.g.  "SESSION#sess_abc"

Any agent imports this module:
    from shared.vector_store import get_vector_store
    store = get_vector_store()
    await store.put("SESSION#u1", "SESSION#s1", {"status": "active"})

All AWS credentials/endpoints are read from environment variables.
TODO markers indicate where real credentials / SDK calls must be substituted.
"""
from __future__ import annotations

import logging
import os
from enum import Enum
from typing import Any

import boto3
from boto3.dynamodb.conditions import Key
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)


# ─── Configuration (all values from env vars) ─────────────────────────────────

class StoreBackend(str, Enum):
    DYNAMODB = "dynamodb"
    S3_VECTORS = "s3_vectors"


# TODO: Replace placeholder values with real credentials via IAM role or Secrets Manager
_AWS_REGION: str = os.getenv("AWS_REGION", "ap-southeast-1")
_DYNAMODB_TABLE_NAME: str = os.getenv("DYNAMODB_TABLE_NAME", "simplify-next-store")
_DYNAMODB_ENDPOINT_URL: str | None = os.getenv("DYNAMODB_ENDPOINT_URL")  # None = real AWS
_S3_VECTORS_BUCKET: str = os.getenv("S3_VECTORS_BUCKET", "simplify-next-vectors")  # TODO
_S3_VECTORS_INDEX: str = os.getenv("S3_VECTORS_INDEX", "simplify-next-index")       # TODO
_STORE_BACKEND: StoreBackend = StoreBackend(
    os.getenv("STORE_BACKEND", StoreBackend.DYNAMODB.value)
)


# ─── VectorStore ──────────────────────────────────────────────────────────────

class VectorStore:
    """
    Centralised key-value / vector store shared across all agents.

    DynamoDB (primary):
        - Single-table design; PK + SK pattern supports all entity types.
        - put / get / delete / query operations.

    S3 Vectors (stub):
        - vector_upsert / vector_search are placeholders.
        - TODO: implement when boto3 adds first-class S3 Vectors support.

    If DynamoDB is unavailable (e.g. no credentials in dev), operations
    degrade gracefully and log warnings — the app continues to run in
    stub/in-memory mode so development is not blocked.
    """

    def __init__(self) -> None:
        self._backend = _STORE_BACKEND
        self._table = None
        self._s3v_client = None
        self._fallback: dict[str, dict] = {}   # in-process fallback when AWS is unreachable
        self._init_clients()

    # ─── Initialisation ───────────────────────────────────────────────────────

    def _init_clients(self) -> None:
        if self._backend == StoreBackend.DYNAMODB:
            self._init_dynamodb()
        elif self._backend == StoreBackend.S3_VECTORS:
            self._init_s3_vectors()

    def _init_dynamodb(self) -> None:
        """
        Initialise DynamoDB resource.
        TODO: Remove aws_access_key_id / aws_secret_access_key arguments and rely
              on IAM instance role when running in AWS (ECS / Lambda / EC2).
        """
        try:
            kwargs: dict[str, Any] = {
                "region_name": _AWS_REGION,
            }
            if _DYNAMODB_ENDPOINT_URL:
                kwargs["endpoint_url"] = _DYNAMODB_ENDPOINT_URL
            # TODO: remove explicit key args — use IAM role in production
            if os.getenv("AWS_ACCESS_KEY_ID"):
                kwargs["aws_access_key_id"] = os.getenv("AWS_ACCESS_KEY_ID")
                kwargs["aws_secret_access_key"] = os.getenv("AWS_SECRET_ACCESS_KEY")

            dynamodb = boto3.resource("dynamodb", **kwargs)
            self._table = dynamodb.Table(_DYNAMODB_TABLE_NAME)
            # Trigger a lightweight call to validate connectivity
            self._table.table_status  # raises if table doesn't exist / no access
            logger.info(
                "VectorStore: DynamoDB connected",
                extra={"table": _DYNAMODB_TABLE_NAME, "endpoint": _DYNAMODB_ENDPOINT_URL},
            )
        except Exception as exc:
            logger.warning(
                f"VectorStore: DynamoDB unavailable — using in-process fallback. "
                f"Reason: {exc}"
            )
            self._table = None

    def _init_s3_vectors(self) -> None:
        """
        TODO: Initialise S3 Vectors client when AWS SDK support is available.
        As of 2025, use the REST API directly or wait for boto3 native support.
        """
        logger.warning(
            "VectorStore: S3 Vectors backend selected but SDK is not yet implemented. "
            "Falling back to DynamoDB init."
        )
        # Attempt DynamoDB as fallback until S3 Vectors SDK ships
        self._backend = StoreBackend.DYNAMODB
        self._init_dynamodb()
        # TODO: replace with:
        # self._s3v_client = boto3.client("s3vectors", region_name=_AWS_REGION)

    # ─── DynamoDB CRUD ────────────────────────────────────────────────────────

    async def put(self, pk: str, sk: str, data: dict[str, Any]) -> bool:
        """
        Write or overwrite an item.
        pk and sk follow "<ENTITY>#<id>" convention — e.g. "SESSION#user1".
        Additional attributes are passed in `data`.
        """
        item = {"PK": pk, "SK": sk, **data}
        if self._table is None:
            self._fallback[f"{pk}#{sk}"] = item
            logger.debug("VectorStore.put: fallback store", extra={"pk": pk, "sk": sk})
            return True
        try:
            self._table.put_item(Item=item)
            return True
        except ClientError as exc:
            logger.error(f"VectorStore.put failed: {exc}", extra={"pk": pk, "sk": sk})
            return False

    async def get(self, pk: str, sk: str) -> dict[str, Any] | None:
        """Retrieve a single item by PK + SK. Returns None if not found."""
        if self._table is None:
            return self._fallback.get(f"{pk}#{sk}")
        try:
            resp = self._table.get_item(Key={"PK": pk, "SK": sk})
            return resp.get("Item")
        except ClientError as exc:
            logger.error(f"VectorStore.get failed: {exc}", extra={"pk": pk, "sk": sk})
            return None

    async def delete(self, pk: str, sk: str) -> bool:
        """Delete an item. Returns True if the operation succeeded (item may not have existed)."""
        if self._table is None:
            self._fallback.pop(f"{pk}#{sk}", None)
            return True
        try:
            self._table.delete_item(Key={"PK": pk, "SK": sk})
            return True
        except ClientError as exc:
            logger.error(f"VectorStore.delete failed: {exc}", extra={"pk": pk, "sk": sk})
            return False

    async def query(self, pk: str) -> list[dict[str, Any]]:
        """
        Return all items with the given partition key.
        Useful for listing all route plans for a session, etc.
        """
        if self._table is None:
            prefix = f"{pk}#"
            return [v for k, v in self._fallback.items() if k.startswith(pk + "#")]
        try:
            resp = self._table.query(
                KeyConditionExpression=Key("PK").eq(pk)
            )
            return resp.get("Items", [])
        except ClientError as exc:
            logger.error(f"VectorStore.query failed: {exc}", extra={"pk": pk})
            return []

    async def update(self, pk: str, sk: str, updates: dict[str, Any]) -> bool:
        """
        Partially update an existing item's attributes.
        Only the keys in `updates` are changed; all other attributes are preserved.
        """
        if self._table is None:
            key = f"{pk}#{sk}"
            if key in self._fallback:
                self._fallback[key].update(updates)
            return True
        try:
            expr_parts = []
            expr_values: dict[str, Any] = {}
            expr_names: dict[str, str] = {}
            for i, (attr, val) in enumerate(updates.items()):
                placeholder = f":v{i}"
                name_alias = f"#n{i}"
                expr_parts.append(f"{name_alias} = {placeholder}")
                expr_values[placeholder] = val
                expr_names[name_alias] = attr
            update_expr = "SET " + ", ".join(expr_parts)
            self._table.update_item(
                Key={"PK": pk, "SK": sk},
                UpdateExpression=update_expr,
                ExpressionAttributeValues=expr_values,
                ExpressionAttributeNames=expr_names,
            )
            return True
        except ClientError as exc:
            logger.error(f"VectorStore.update failed: {exc}", extra={"pk": pk, "sk": sk})
            return False

    # ─── S3 Vectors (placeholder) ─────────────────────────────────────────────

    async def vector_upsert(
        self,
        index: str,
        item_id: str,
        vector: list[float],
        metadata: dict[str, Any],
    ) -> bool:
        """
        Upsert a vector embedding into an S3 Vectors index.
        TODO: implement when AWS S3 Vectors boto3 client is available.
        """
        logger.warning(
            "VectorStore.vector_upsert: S3 Vectors not implemented — placeholder only",
            extra={"index": index, "id": item_id},
        )
        # TODO: self._s3v_client.put_vectors(bucket=_S3_VECTORS_BUCKET, index=index, ...)
        return False

    async def vector_search(
        self,
        index: str,
        query_vector: list[float],
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """
        Search for semantically similar items in an S3 Vectors index.
        TODO: implement when AWS S3 Vectors boto3 client is available.
        """
        logger.warning(
            "VectorStore.vector_search: S3 Vectors not implemented — placeholder only",
            extra={"index": index, "top_k": top_k},
        )
        # TODO: self._s3v_client.query_vectors(bucket=_S3_VECTORS_BUCKET, index=index, ...)
        return []

    # ─── Key helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def session_keys(user_id: str, session_id: str) -> tuple[str, str]:
        return f"SESSION#{user_id}", f"SESSION#{session_id}"

    @staticmethod
    def route_plan_keys(session_id: str, plan_id: str) -> tuple[str, str]:
        return f"ROUTE_PLAN#{session_id}", f"PLAN#{plan_id}"

    @staticmethod
    def locked_route_keys(session_id: str) -> tuple[str, str]:
        return f"LOCKED_ROUTE#{session_id}", "LOCKED#current"

    @staticmethod
    def route_state_keys(session_id: str) -> tuple[str, str]:
        return f"ROUTE_STATE#{session_id}", "STATE#current"


# ─── Singleton accessor ───────────────────────────────────────────────────────

_instance: VectorStore | None = None


def get_vector_store() -> VectorStore:
    """
    Return the process-wide VectorStore singleton.
    Safe to call from any agent module.
    """
    global _instance
    if _instance is None:
        _instance = VectorStore()
    return _instance
