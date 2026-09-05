from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "Transit Orchestrator Agent"
    demo_mode: bool = True
    aws_region: str = "ap-southeast-1"
    bedrock_vision_model_id: str = "apac.amazon.nova-pro-v1:0"
    lta_account_key: str | None = None
    routing_agent_url: str | None = None
    late_threshold_minutes: int = 8
    arrival_window_minutes: int = 2
    max_observation_age_seconds: int = 90

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


@lru_cache
def get_settings() -> Settings:
    return Settings()

