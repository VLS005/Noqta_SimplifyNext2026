"""AWS credential sanity checks and error classification."""

from __future__ import annotations

import logging
import sys

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError
from fastapi import HTTPException

logger = logging.getLogger(__name__)

AWS_CREDENTIAL_ERROR_CODES = frozenset(
    {
        "ExpiredTokenException",
        "ExpiredToken",
        "UnrecognizedClientException",
        "InvalidClientTokenId",
    }
)

STARTUP_CREDENTIAL_WARNING = (
    "WARNING: AWS credentials appear invalid or expired. DynamoDB/Bedrock "
    "calls will fail until you refresh your access keys from the AWS portal "
    "and update .env, then restart this server."
)

AWS_CREDENTIALS_INVALID_DETAIL = (
    "AWS credentials expired or invalid. Refresh your access keys "
    "from the AWS portal, update .env, and restart the server."
)


def is_aws_credential_error(exc: BaseException) -> bool:
    """True for expired/invalid AWS access-key errors (not generic AWS failures)."""
    if type(exc).__name__ in AWS_CREDENTIAL_ERROR_CODES:
        return True
    if isinstance(exc, ClientError):
        code = exc.response.get("Error", {}).get("Code", "")
        if code in AWS_CREDENTIAL_ERROR_CODES:
            return True
    cause = exc.__cause__ or getattr(exc, "__context__", None)
    if cause is not None and cause is not exc:
        return is_aws_credential_error(cause)
    return False


def should_check_aws_credentials() -> bool:
    """Skip the live STS call under pytest so unit tests never hit AWS."""
    return "pytest" not in sys.modules


def check_aws_credentials(region: str, sts_client=None) -> bool:
    """Cheap STS ping. Never raises; False means DynamoDB/Bedrock will likely fail."""
    client = sts_client or boto3.client(
        "sts",
        region_name=region,
        config=Config(
            connect_timeout=2,
            read_timeout=2,
            retries={"max_attempts": 1},
        ),
    )
    try:
        client.get_caller_identity()
        return True
    except Exception as exc:
        if is_aws_credential_error(exc):
            logger.warning(STARTUP_CREDENTIAL_WARNING)
            print(STARTUP_CREDENTIAL_WARNING, flush=True)
        else:
            logger.warning(
                "WARNING: AWS credential check failed (%s). "
                "DynamoDB/Bedrock calls may fail until credentials are fixed.",
                exc,
            )
        logger.debug("AWS credential check error: %s", exc, exc_info=True)
        return False


def http_error_for_store_failure(exc: Exception, generic_prefix: str) -> HTTPException:
    """Map DynamoDB/Bedrock failures to 502; credential errors get a fixed message."""
    if is_aws_credential_error(exc):
        logger.debug("AWS credential error: %s", exc, exc_info=True)
        return HTTPException(status_code=502, detail=AWS_CREDENTIALS_INVALID_DETAIL)
    return HTTPException(status_code=502, detail=f"{generic_prefix}: {exc}")
