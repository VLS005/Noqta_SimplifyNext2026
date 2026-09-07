"""
All configuration for the Routing Agent, loaded from environment variables.
- Pydantic BaseSettings provides automatic env-var parsing, type coercion, and .env file loading.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from dotenv import load_dotenv

load_dotenv()

# {
#   "origin_label": "Marina Bay Sands",
#   "destination_label": "Jewel Changi",
#   "user_id": "anonymous",
#   "origin_lat": 0,
#   "origin_lon": 0,
#   "destination_lat": 0,
#   "destination_lon": 0
# }

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
    aws_region: str = Field(default="us-east-1")
    aws_access_key_id: str | None = Field(default=None)          # TODO: replace with IAM role
    aws_secret_access_key: str | None = Field(default=None)      # TODO: replace with IAM role
    aws_session_token: str | None = Field(default=None)

    # ─── Amazon Bedrock ───────────────────────────────────────────────────────
    bedrock_region: str = Field(default="us-east-1")
    bedrock_model_id: str = Field(default="anthropic.claude-3-5-sonnet-20241022-v2:0")
    bedrock_endpoint_url: str | None = Field(default=None)

    # ─── Gemini LLM ───────────────────────────────────────────────────────────
    gemini_api_key: str | None = Field(default=None)

    # ─── Google Maps / Routes API ─────────────────────────────────────────────
    google_maps_api_key: str | None = Field(default=None)

    # ─── DynamoDB ─────────────────────────────────────────────────────────────
    dynamodb_table_name: str = Field(default="DoraDB")
    dynamodb_endpoint_url: str | None = Field(default=None)      # None = real AWS; set for local



    # ─── External agents ──────────────────────────────────────────────────────
    personalization_agent_url: str = Field(
        default="http://localhost:8002/inbound/start-timer"
    )
    personalization_agent_timeout_s: int = Field(default=5)
    
    # ─── Business logic ───────────────────────────────────────────────────────
    vision_proximity_gate_m: float = Field(default=5.0)
    max_route_candidates: int = Field(default=3)
    fine_motor_buffer_s: int = Field(default=30)
    transit_transfer_buffer_s: int = Field(default=120)

    @property
    def is_development(self) -> bool:
        return self.app_env.lower() == "development"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached Settings instance. Call this anywhere in the app."""
    return Settings()
