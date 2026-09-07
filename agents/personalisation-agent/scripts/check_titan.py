"""Live Bedrock Titan Text Embeddings V2 check in us-east-1. Never falls back to demo mode."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.aws_env import load_local_env  # noqa: E402

load_local_env()

import boto3  # noqa: E402
from botocore.exceptions import BotoCoreError, ClientError  # noqa: E402

from app.config import Settings  # noqa: E402


def main() -> int:
    settings = Settings(demo_mode=False)
    model_id = settings.bedrock_embedding_model_id
    region = settings.aws_region
    print(f"Calling bedrock-runtime in {region} model={model_id} input='test'", flush=True)
    try:
        client = boto3.client("bedrock-runtime", region_name=region)
        response = client.invoke_model(
            modelId=model_id,
            contentType="application/json",
            accept="application/json",
            body=json.dumps(
                {
                    "inputText": "test",
                    "dimensions": settings.embedding_dimensions,
                    "normalize": True,
                }
            ),
        )
        payload = json.loads(response["body"].read())
        embedding = payload.get("embedding") or []
        print(
            f"SUCCESS: Titan returned {len(embedding)} dimensions "
            f"(inputTextTokenCount={payload.get('inputTextTokenCount')})",
            flush=True,
        )
        return 0
    except ClientError as exc:
        error = exc.response.get("Error", {})
        print(
            f"FAIL: ClientError code={error.get('Code')} message={error.get('Message')}",
            flush=True,
        )
        print(f"RAW: {exc}", flush=True)
        return 1
    except BotoCoreError as exc:
        print(f"FAIL: BotoCoreError {type(exc).__name__}: {exc}", flush=True)
        return 1
    except Exception as exc:
        print(f"FAIL: {type(exc).__name__}: {exc}", flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
