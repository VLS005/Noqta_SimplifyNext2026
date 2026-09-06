"""
Amazon Bedrock & Gemini Runtime wrapper for LLM inference.
- Fully implements both AWS Bedrock and Google Gemini clients.
- Uses Nova Lite by default for basic reasoning (instruction parsing).
- Can use Nova Pro for advanced reasoning, falling back to Sonnet 3.5 or Gemini.
- Stubs automatically when no credentials are set.
"""
from __future__ import annotations

import json
import logging
from functools import lru_cache

# --- AWS Bedrock Imports ---
import boto3
from botocore.exceptions import ClientError, NoCredentialsError

# --- Gemini Imports (new google-genai SDK) ---
from google import genai
from google.genai import types

from backend.routing_agent.config import get_settings

logger = logging.getLogger(__name__)


class BedrockClient:
    """
    Async-compatible wrapper around AWS Bedrock and Gemini.
    Provides tiered model selection and robust fallback logic to prevent 
    downtime during low credit situations.
    """

    def __init__(self) -> None:
        settings = get_settings()
        self._stub_mode = False

        # --- Bedrock Init ---
        self._bedrock_client = None
        self._init_client_bedrock(settings)

        # --- Gemini Init ---
        self._gemini_client = None
        self._gemini_model_id = "gemini-2.0-flash"
        self._init_client_gemini(settings)

    def _init_client_bedrock(self, settings) -> None:
        kwargs: dict = {"region_name": settings.bedrock_region or "us-east-1"}
        if settings.bedrock_endpoint_url:
            kwargs["endpoint_url"] = settings.bedrock_endpoint_url
        if settings.aws_access_key_id:
            kwargs["aws_access_key_id"] = settings.aws_access_key_id
            kwargs["aws_secret_access_key"] = settings.aws_secret_access_key

        try:
            self._bedrock_client = boto3.client("bedrock-runtime", **kwargs)
            logger.info("BedrockClient: AWS Bedrock initialised")
        except Exception as exc:
            logger.warning(f"BedrockClient: failed to init boto3 client. Reason: {exc}")
            # Don't set full stub mode here, we might still have Gemini

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

    async def invoke_model(self, system_prompt: str, user_prompt: str, model_tier: str = "nova-lite") -> str:
        """
        Call LLM with fallback logic.
        Tiers:
        - 'nova-lite': For tasks requiring least amount of reasoning (e.g. instruction parsing).
        - 'nova-pro': For in-depth reasoning (e.g. multimodal, complex logic).
        
        Fallback chain: Requested Bedrock Model -> Claude 3.5 Sonnet -> Gemini -> Stub.
        """
        if self._stub_mode or (self._bedrock_client is None and self._gemini_client is None):
            return self._stub_response(system_prompt, user_prompt)

        # Map tier to actual model ID
        if model_tier == "nova-pro":
            model_id = "amazon.nova-pro-v1:0"
        else:
            # Default to nova-lite for fast instruction parsing
            model_id = "amazon.nova-lite-v1:0"

        # Attempt 1: Requested Bedrock Model
        if self._bedrock_client:
            try:
                response_text = self._invoke_bedrock_converse(model_id, system_prompt, user_prompt)
                print(f"[BedrockClient] AWS {model_id} response received ({len(response_text)} chars)")
                return response_text
            except Exception as e:
                logger.warning(f"BedrockClient: {model_id} failed ({e}). Falling back to Sonnet 3.5...")
                print(f"[BedrockClient] WARNING: {model_id} failed ({e}). Falling back...")
                
                # Attempt 2: Fallback to Sonnet (if credits/quotas for Nova are out)
                try:
                    fallback_model = "anthropic.claude-3-5-sonnet-20241022-v2:0"
                    response_text = self._invoke_bedrock_converse(fallback_model, system_prompt, user_prompt)
                    print(f"[BedrockClient] AWS {fallback_model} fallback response received ({len(response_text)} chars)")
                    return response_text
                except Exception as e2:
                    logger.warning(f"BedrockClient: Sonnet fallback failed ({e2}). Falling back to Gemini...")
                    print(f"[BedrockClient] WARNING: Sonnet fallback failed. Trying Gemini...")

        # Attempt 3: Fallback to Gemini (our ultimate safety net)
        if self._gemini_client:
            try:
                response = await self._gemini_client.aio.models.generate_content(
                    model=self._gemini_model_id,
                    contents=user_prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=system_prompt,
                        max_output_tokens=2048,
                    ),
                )
                text = response.text
                print(f"[BedrockClient/Gemini] LLM response received ({len(text)} chars)")
                return text
            except Exception as exc:
                logger.error(f"BedrockClient: Gemini fallback failed: {exc}")
                print(f"[BedrockClient] ERROR: Gemini fallback failed. Using stub.")

        # Attempt 4: Stub
        self._stub_mode = True
        return self._stub_response(system_prompt, user_prompt)

    def _invoke_bedrock_converse(self, model_id: str, system_prompt: str, user_prompt: str) -> str:
        """Helper to invoke Bedrock using the unified Converse API."""
        response = self._bedrock_client.converse(
            modelId=model_id,
            system=[{"text": system_prompt}],
            messages=[{"role": "user", "content": [{"text": user_prompt}]}],
            inferenceConfig={"maxTokens": 2048}
        )
        return response["output"]["message"]["content"][0]["text"]

    def _stub_response(self, system_prompt: str, user_prompt: str) -> str:
        """Deterministic stub responses keyed by intent detected in the prompts."""
        combined = (system_prompt + user_prompt).lower()
        logger.debug("BedrockClient: returning stub response")
        print(f"[BedrockClient] STUB mode — returning deterministic response")

        if "translate" in combined or "micro-instruction" in combined or "blind" in combined:
            return json.dumps([
                "Walk straight for 50 metres. Tactile paving is underfoot.",
                "At the corner, turn left. You'll hear the coffee shop on your right.",
                "Continue 30 metres. The entrance is directly ahead — push the door.",
                "You have arrived. The reception desk is 5 metres ahead.",
            ])

        if "voice" in combined or "route" in combined and "select" in combined:
            return json.dumps({"resolved_index": 1, "confidence": 0.95})

        if "score" in combined or "accessib" in combined:
            return "0.82"

        if "eta" in combined or "duration" in combined:
            return "420"

        return "Stub response — LLM not connected. Set API keys to enable real inference."


@lru_cache(maxsize=1)
def get_bedrock_client() -> BedrockClient:
    """Return a process-wide singleton LLM Client."""
    return BedrockClient()
