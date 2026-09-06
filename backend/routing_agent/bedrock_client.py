"""
Amazon Bedrock & Gemini Runtime wrapper for LLM inference.
- Fully implements both AWS Bedrock and Google Gemini clients.
- Uses Nova Lite by default for basic reasoning (instruction parsing).
- Can use Nova Pro for advanced reasoning, falling back to Sonnet 3.5 or Gemini.
- No stub mode: if every provider fails, raises/returns a clear connection error.
"""
from __future__ import annotations

import logging
from functools import lru_cache

# --- AWS Bedrock Imports ---
import boto3
from botocore.exceptions import ClientError, NoCredentialsError

# --- Gemini Imports (new google-genai SDK) ---
from google import genai
from google.genai import types

from backend.routing_agent.config import get_settings

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


class BedrockClient:
    """
    Async-compatible wrapper around AWS Bedrock and Gemini.
    Provides tiered model selection and fallback logic across providers.
    """

    def __init__(self) -> None:
        settings = get_settings()

        # --- Bedrock Init ---
        self._bedrock_client = None
        self._init_client_bedrock(settings)

        # --- Gemini Init ---
        self._gemini_client = None
        self._gemini_model_id = "gemini-3.6-flash"
        self._init_client_gemini(settings)

    def _init_client_bedrock(self, settings) -> None:
        kwargs: dict = {"region_name": settings.bedrock_region or "us-east-1"}
        if settings.bedrock_endpoint_url:
            kwargs["endpoint_url"] = settings.bedrock_endpoint_url
        if settings.aws_access_key_id:
            kwargs["aws_access_key_id"] = settings.aws_access_key_id
            kwargs["aws_secret_access_key"] = settings.aws_secret_access_key
            # Temporary/STS credentials (e.g. AWS Academy Learner Lab) require
            # the session token as well — without it you get
            # UnrecognizedClientException: security token invalid.
            session_token = getattr(settings, "aws_session_token", None)
            if session_token:
                kwargs["aws_session_token"] = session_token
            else:
                logger.warning(
                    "BedrockClient: no aws_session_token set. If you're using "
                    "temporary/STS credentials, requests will fail with "
                    "UnrecognizedClientException."
                )

        try:
            self._bedrock_client = boto3.client("bedrock-runtime", **kwargs)
            logger.info("BedrockClient: AWS Bedrock initialised")
        except Exception as exc:
            logger.warning(f"BedrockClient: failed to init boto3 client. Reason: {exc}")

    def _init_client_gemini(self, settings) -> None:
        """Initialize Gemini client using the new google-genai SDK."""
        if not settings.gemini_api_key:
            logger.warning("BedrockClient: GEMINI_API_KEY not set.")
            return

        try:
            self._gemini_client = genai.Client(api_key=settings.gemini_api_key)
            logger.info(
                "BedrockClient: Gemini configured",
                extra={"model_id": self._gemini_model_id},
            )
        except Exception as exc:
            logger.warning(f"BedrockClient: failed to init Gemini client. Reason: {exc}")

    async def invoke_model(self, system_prompt: str, user_prompt: str, model_tier: str = "nova-lite", image_bytes: bytes = None) -> str:
        """
        Call LLM with fallback logic.
        Tiers:
        - 'nova-lite': For tasks requiring least amount of reasoning (e.g. instruction parsing).
        - 'nova-pro': For in-depth reasoning (e.g. multimodal, complex logic).

        Fallback chain: Requested Bedrock Model -> Claude 3.5 Sonnet -> Gemini -> hard error.
        """
        last_error: Exception | None = None

        # Map tier to actual model ID
        if model_tier == "nova-pro":
            model_id = "amazon.nova-pro-v1:0"
        else:
            model_id = "amazon.nova-lite-v1:0"

        # Attempt 1: Requested Bedrock Model
        if self._bedrock_client:
            try:
                response_text = self._invoke_bedrock_converse(model_id, system_prompt, user_prompt, image_bytes)
                print(f"[BedrockClient] AWS {model_id} response received ({len(response_text)} chars)")
                return response_text
            except Exception as e:
                last_error = e
                if "UnrecognizedClientException" in str(e) or "ExpiredTokenException" in str(e):
                    logger.error("BedrockClient: Token invalid/expired. Disabling Bedrock and falling back to Gemini.")
                    self._bedrock_client = None
                else:
                    logger.warning(f"BedrockClient: {model_id} failed ({e}). Falling back to Sonnet 3.5...")
                    print(f"[BedrockClient] WARNING: {model_id} failed ({e}). Falling back...")

                    # Attempt 2: Fallback to Sonnet (if credits/quotas for Nova are out)
                    try:
                        fallback_model = "anthropic.claude-3-5-sonnet-20241022-v2:0"
                        response_text = self._invoke_bedrock_converse(fallback_model, system_prompt, user_prompt, image_bytes)
                        print(f"[BedrockClient] AWS {fallback_model} fallback response received ({len(response_text)} chars)")
                        return response_text
                    except Exception as e2:
                        last_error = e2
                        logger.warning(f"BedrockClient: Sonnet fallback failed ({e2}). Falling back to Gemini...")
                        print(f"[BedrockClient] WARNING: Sonnet fallback failed. Trying Gemini...")
                        if "UnrecognizedClientException" in str(e2) or "ExpiredTokenException" in str(e2):
                            self._bedrock_client = None

        # Attempt 3: Fallback to Gemini
        if self._gemini_client:
            try:
                contents = []
                if image_bytes:
                    contents.append(types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"))
                contents.append(user_prompt)
                
                response = await self._gemini_client.aio.models.generate_content(
                    model=self._gemini_model_id,
                    contents=contents,
                    config=types.GenerateContentConfig(
                        system_instruction=system_prompt,
                        max_output_tokens=2048,
                        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True)
                    ),
                )
                text = response.text
                print(f"[BedrockClient/Gemini] LLM response received ({len(text)} chars)")
                return text
            except Exception as exc:
                last_error = exc
                logger.error(f"BedrockClient: Gemini fallback failed: {exc}")
                print(f"[BedrockClient] ERROR: Gemini fallback failed.")

        # All providers failed — surface a clear, unambiguous error instead of a stub.
        error_msg = f"cannot connect to client... {last_error}"
        logger.error(f"BedrockClient: all providers failed. {error_msg}")
        print(f"[BedrockClient] ERROR: {error_msg}")
        return f"ERROR: {error_msg}"

    def _invoke_bedrock_converse(self, model_id: str, system_prompt: str, user_prompt: str, image_bytes: bytes = None) -> str:
        """Helper to invoke Bedrock using the unified Converse API."""
        content = []
        if image_bytes:
            content.append({
                "image": {
                    "format": "jpeg",
                    "source": {"bytes": image_bytes}
                }
            })
        content.append({"text": user_prompt})

        response = self._bedrock_client.converse(
            modelId=model_id,
            system=[{"text": system_prompt}],
            messages=[{"role": "user", "content": content}],
            inferenceConfig={"maxTokens": 2048}
        )
        return response["output"]["message"]["content"][0]["text"]


@lru_cache(maxsize=1)
def get_bedrock_client() -> BedrockClient:
    """Return a process-wide singleton LLM Client."""
    return BedrockClient()