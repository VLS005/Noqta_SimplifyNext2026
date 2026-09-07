"""DynamoDB pace tracking on the shared DoraDB table.

PK/SK scheme (must not collide with other agents):
  journey record:  PK = USER#<user_id>   SK = PACE#<timestamp>
  rolling profile: PK = USER#<user_id>   SK = PROFILE

Compatibility mirror for Routing Agent ETA lookup:
  PK = PACE#<user_id>   SK = PACE#current   attribute average_pace_mps
"""

from __future__ import annotations

import boto3
from botocore.exceptions import ClientError

from app.config import Settings


def user_pk(user_id: str) -> str:
    return f"USER#{user_id}"


def pace_sk(timestamp: float) -> str:
    return f"PACE#{timestamp}"


class PaceStore:
    def __init__(self, settings: Settings, dynamodb_client=None):
        self.settings = settings
        self.dynamodb = dynamodb_client or boto3.client(
            "dynamodb", region_name=settings.aws_region
        )

    def write_journey_pace(
        self,
        user_id: str,
        timestamp: float,
        steps: int,
        elapsed_minutes: float,
        pace_spm: float,
        route_id: str,
    ) -> None:
        self.dynamodb.put_item(
            TableName=self.settings.dynamodb_table_name,
            Item={
                "PK": {"S": user_pk(user_id)},
                "SK": {"S": pace_sk(timestamp)},
                "steps": {"N": str(steps)},
                "elapsed_minutes": {"N": str(elapsed_minutes)},
                "pace_spm": {"N": str(pace_spm)},
                "route_id": {"S": route_id},
            },
        )

    def recent_paces(self, user_id: str, limit: int | None = None) -> list[float]:
        response = self.dynamodb.query(
            TableName=self.settings.dynamodb_table_name,
            KeyConditionExpression="PK = :pk AND begins_with(SK, :sk)",
            ExpressionAttributeValues={
                ":pk": {"S": user_pk(user_id)},
                ":sk": {"S": "PACE#"},
            },
            ScanIndexForward=False,
            Limit=limit or self.settings.pace_window,
        )
        paces: list[float] = []
        for item in response.get("Items", []):
            raw = item.get("pace_spm", {}).get("N")
            if raw is not None:
                paces.append(float(raw))
        return paces

    def upsert_profile(self, user_id: str, rolling_avg_pace_spm: float) -> None:
        self.dynamodb.put_item(
            TableName=self.settings.dynamodb_table_name,
            Item={
                "PK": {"S": user_pk(user_id)},
                "SK": {"S": "PROFILE"},
                "rolling_avg_pace_spm": {"N": str(rolling_avg_pace_spm)},
            },
        )

    def get_profile(self, user_id: str) -> float | None:
        try:
            response = self.dynamodb.get_item(
                TableName=self.settings.dynamodb_table_name,
                Key={
                    "PK": {"S": user_pk(user_id)},
                    "SK": {"S": "PROFILE"},
                },
            )
        except ClientError:
            raise
        item = response.get("Item")
        if not item:
            return None
        raw = item.get("rolling_avg_pace_spm", {}).get("N")
        return float(raw) if raw is not None else None

    def upsert_routing_pace_current(self, user_id: str, average_pace_mps: float) -> None:
        """Write the key Routing's ETAAgent already reads: PACE#{user_id} / PACE#current."""
        self.dynamodb.put_item(
            TableName=self.settings.dynamodb_table_name,
            Item={
                "PK": {"S": f"PACE#{user_id}"},
                "SK": {"S": "PACE#current"},
                "average_pace_mps": {"N": str(average_pace_mps)},
                "user_id": {"S": user_id},
                "source": {"S": "personalisation_agent"},
            },
        )
