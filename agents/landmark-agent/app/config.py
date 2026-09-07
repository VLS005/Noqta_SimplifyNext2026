from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


def _nonempty(value: str | None) -> str | None:
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None


class Settings(BaseSettings):
    app_name: str = "Landmark Verification Agent"
    demo_mode: bool = True
    aws_region: str = "ap-southeast-1"
    bedrock_vision_model_id: str = "apac.amazon.nova-pro-v1:0"
    aws_access_key_id: str | None = None
    aws_secret_access_key: str | None = None
    aws_session_token: str | None = None

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    def boto3_credential_kwargs(self) -> dict[str, str]:
        """Explicit keys for boto3.client(...). Empty if .env has no sandbox creds."""
        access = _nonempty(self.aws_access_key_id)
        secret = _nonempty(self.aws_secret_access_key)
        token = _nonempty(self.aws_session_token)
        if not access or not secret:
            return {}
        kwargs = {
            "aws_access_key_id": access,
            "aws_secret_access_key": secret,
        }
        if token:
            kwargs["aws_session_token"] = token
        return kwargs

    @property
    def credentials_source(self) -> str:
        return "settings" if self.boto3_credential_kwargs() else "default_chain"


@lru_cache
def get_settings() -> Settings:
    return Settings()
