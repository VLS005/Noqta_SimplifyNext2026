from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "Personalisation Agent"
    demo_mode: bool = True
    aws_region: str = "us-east-1"
    dynamodb_table_name: str = "DoraDB"
    # Titan Text Embeddings V2 default output is 1024-d (also supports 512 and 256).
    bedrock_embedding_model_id: str = "amazon.titan-embed-text-v2:0"
    embedding_dimensions: int = 1024
    vector_index_name: str = "obstacle-memory-index"
    create_vector_index_if_missing: bool = False
    default_baseline_pace_spm: float = 95.0
    default_stride_length_m: float = 0.7
    routing_agent_pace_url: str = "http://localhost:8000/api/inbound/pace"
    geohash_precision: int = 7
    memory_top_k: int = 5
    pace_window: int = 10
    min_elapsed_minutes: float = 0.1

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


@lru_cache
def get_settings() -> Settings:
    return Settings()
