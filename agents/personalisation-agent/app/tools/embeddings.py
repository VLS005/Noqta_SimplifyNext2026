"""Bedrock Titan Text Embeddings V2 client.

Titan Text Embeddings V2 (`amazon.titan-embed-text-v2:0`) accepts 256, 512, or
1024 dimensions. We pin 1024 because that is the model default and the size of
the DoraDB vector index (`obstacle-memory-index`).
"""

from __future__ import annotations

import hashlib
import json
import math

import boto3

from app.config import Settings

# Titan V2 input limit is 8,192 tokens. 20,000 chars leaves a safe margin.
_MAX_INPUT_CHARS = 20_000


class EmbeddingClient:
    def __init__(self, settings: Settings, bedrock_client=None):
        self.settings = settings
        self.bedrock = bedrock_client or boto3.client(
            "bedrock-runtime", region_name=settings.aws_region
        )

    def embed(self, text: str) -> list[float]:
        if self.settings.demo_mode:
            return self._demo_embedding(text)

        response = self.bedrock.invoke_model(
            modelId=self.settings.bedrock_embedding_model_id,
            contentType="application/json",
            accept="application/json",
            body=json.dumps(
                {
                    "inputText": text[:_MAX_INPUT_CHARS],
                    "dimensions": self.settings.embedding_dimensions,
                    "normalize": True,
                }
            ),
        )
        payload = json.loads(response["body"].read())
        embedding = payload["embedding"]
        if len(embedding) != self.settings.embedding_dimensions:
            raise ValueError(
                f"Titan returned {len(embedding)} dimensions; "
                f"expected {self.settings.embedding_dimensions}"
            )
        return [float(value) for value in embedding]

    def _demo_embedding(self, text: str) -> list[float]:
        """Deterministic unit vector so local demo/tests do not call Bedrock."""
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        dims = self.settings.embedding_dimensions
        values: list[float] = []
        seed = digest
        while len(values) < dims:
            for byte in seed:
                values.append((byte / 127.5) - 1.0)
                if len(values) == dims:
                    break
            seed = hashlib.sha256(seed).digest()
        norm = math.sqrt(sum(v * v for v in values)) or 1.0
        return [v / norm for v in values]
