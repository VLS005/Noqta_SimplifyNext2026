"""
All configuration for the Routing Agent, loaded from environment variables.
- Pydantic BaseSettings provides automatic env-var parsing, type coercion, and .env file loading.
"""
from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ─── App ──────────────────────────────────────────────────────────────────
    app_env: str = Field(default="development")
    log_level: str = Field(default="INFO")
    routing_agent_port: int = Field(default=8010)

    # ─── AWS Core ─────────────────────────────────────────────────────────────
    aws_region: str = Field(default="ap-southeast-1")
    aws_access_key_id: str | None = Field(default=None)          # TODO: replace with IAM role
    aws_secret_access_key: str | None = Field(default=None)      # TODO: replace with IAM role

    # ─── Amazon Bedrock ───────────────────────────────────────────────────────
    bedrock_region: str = Field(default="ap-southeast-1")
    bedrock_model_id: str = Field(default="anthropic.claude-3-5-sonnet-20241022-v2:0")
    bedrock_endpoint_url: str | None = Field(default=None)

    # ─── Gemini LLM ───────────────────────────────────────────────────────────
    gemini_api_key: str | None = Field(default=None)

    # ─── Google Maps / Routes API ─────────────────────────────────────────────
    google_maps_api_key: str | None = Field(default=None)

    # ─── DynamoDB ─────────────────────────────────────────────────────────────
    dynamodb_table_name: str = Field(default="simplify-next-store")
    dynamodb_endpoint_url: str | None = Field(default=None)      # None = real AWS; set for local

    # ─── S3 Vectors ───────────────────────────────────────────────────────────
    s3_vectors_bucket: str = Field(default="simplify-next-vectors")
    s3_vectors_index: str = Field(default="simplify-next-index")
    store_backend: str = Field(default="dynamodb")

    # ─── External agents ──────────────────────────────────────────────────────
    personalization_agent_url: str = Field(
        default="http://localhost:8001/inbound/start-timer"
    )
    personalization_agent_timeout_s: int = Field(default=5)

    # ─── Business logic ───────────────────────────────────────────────────────
    vision_proximity_gate_m: float = Field(default=5.0)
    max_route_candidates: int = Field(default=3)
    fine_motor_buffer_s: int = Field(default=30)

    @property
    def is_development(self) -> bool:
        return self.app_env.lower() == "development"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached Settings instance. Call this anywhere in the app."""
    return Settings()
