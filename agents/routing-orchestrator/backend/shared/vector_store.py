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


# ─── Configuration ──────────────────────────────────────────────────────────────

from backend.routing_agent.config import get_settings

# ─── VectorStore ──────────────────────────────────────────────────────────────

class VectorStore:
    """
    Centralised key-value store shared across all agents.

    DynamoDB (primary):
        - Single-table design; PK + SK pattern supports all entity types.
        - put / get / delete / query operations.

    If DynamoDB is unavailable (e.g. no credentials in dev), operations
    degrade gracefully and log warnings — the app continues to run in
    stub/in-memory mode so development is not blocked.
    """

    def __init__(self) -> None:
        self._settings = get_settings()
        self._table = None
        self._fallback: dict[str, dict] = {}   # in-process fallback when AWS is unreachable
        self._init_dynamodb()

    def _init_dynamodb(self) -> None:
        """
        Initialise DynamoDB resource.
        TODO: Remove aws_access_key_id / aws_secret_access_key arguments and rely
              on IAM instance role when running in AWS (ECS / Lambda / EC2).
        """
        try:
            kwargs: dict[str, Any] = {
                "region_name": self._settings.aws_region,
            }
            if self._settings.dynamodb_endpoint_url:
                kwargs["endpoint_url"] = self._settings.dynamodb_endpoint_url
            
            # Use specific keys if defined, otherwise boto3 falls back to env vars/IAM roles
            if self._settings.aws_access_key_id and self._settings.aws_secret_access_key:
                kwargs["aws_access_key_id"] = self._settings.aws_access_key_id
                kwargs["aws_secret_access_key"] = self._settings.aws_secret_access_key

            dynamodb = boto3.resource("dynamodb", **kwargs)
            self._table = dynamodb.Table(self._settings.dynamodb_table_name)
            # Trigger a lightweight call to validate connectivity
            self._table.table_status  # raises if table doesn't exist / no access
            logger.info(
                "VectorStore: DynamoDB connected",
                extra={"table": self._settings.dynamodb_table_name, "endpoint": self._settings.dynamodb_endpoint_url},
            )
        except Exception as exc:
            logger.warning(
                f"VectorStore: DynamoDB unavailable — using in-process fallback. "
                f"Reason: {exc}"
            )
            self._table = None

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
