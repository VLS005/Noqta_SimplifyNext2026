"""
Consolidates outbound personalization events and inbound obstruction reports.
Acts as the central communication sub-agent.
"""
from __future__ import annotations

import logging
from typing import Any

import httpx

from backend.routing_agent.config import get_settings
from backend.routing_agent.models import BaseMessage, StartTimerEventMessage, ObstructionReportMessage
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


class PersonalizationAdapter:
    """
    Sends StartTimerEventMessage to the Personalization Agent via HTTP POST.
    Gracefully handles the case where the agent is not yet deployed.
    """

    def agent_name(self) -> str:
        return "personalization_agent"

    async def send(self, message: BaseMessage) -> bool:
        if not isinstance(message, StartTimerEventMessage):
            logger.error(
                "PersonalizationAdapter.send: unexpected message type",
                extra={"type": type(message).__name__},
            )
            return False

        settings = get_settings()
        url = settings.personalization_agent_url
        timeout = settings.personalization_agent_timeout_s

        logger.info(
            "PersonalizationAdapter: dispatching StartTimerEventMessage",
            extra={
                "url": url,
                "session_id": message.session_id,
                "route_id": message.locked_route_id,
                "duration_s": message.estimated_duration_s,
            },
        )

        payload = message.model_dump(mode="json")

        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(url, json=payload)
                response.raise_for_status()
                logger.info(
                    "PersonalizationAdapter: StartTimerEventMessage accepted",
                    extra={"status": response.status_code, "url": url},
                )
                return True
        except httpx.ConnectError:
            # Expected in MVP — PersonalizationAgent not yet deployed
            logger.warning(
                "PersonalizationAdapter: Personalization Agent unreachable — "
                "StartTimerEvent logged locally. Set PERSONALIZATION_AGENT_URL when deployed.",
                extra={"url": url, "session_id": message.session_id},
            )
            self._log_locally(message)
            return False
        except httpx.HTTPStatusError as exc:
            logger.error(
                f"PersonalizationAdapter: HTTP error {exc.response.status_code}",
                extra={"url": url, "session_id": message.session_id},
            )
            return False
        except httpx.TimeoutException:
            logger.warning(
                f"PersonalizationAdapter: request timed out after {timeout}s",
                extra={"url": url},
            )
            return False

    @staticmethod
    def _log_locally(message: StartTimerEventMessage) -> None:
        """
        Persist the event locally when the Personalization Agent is unreachable.
        This ensures no data is silently lost during development.
        """
        logger.info(
            "PersonalizationAdapter [LOCAL LOG] StartTimerEvent",
            extra={
                "event": "start_timer",
                "user_id": message.user_id,
                "session_id": message.session_id,
                "locked_route_id": message.locked_route_id,
                "estimated_duration_s": message.estimated_duration_s,
                "destination_label": message.destination_label,
                "route_distance_m": message.route_distance_m,
                "timestamp": message.timestamp.isoformat(),
            },
        )


class ObstructionAdapter:
    """
    Parses obstruction reports from the Obstruction Agent.
    Currently a read-only, no-op adapter — logs and discards.
    """

    def agent_name(self) -> str:
        return "obstruction_agent"

    def parse(self, raw: dict[str, Any]) -> ObstructionReportMessage:
        """
        Parse and validate an inbound obstruction report payload.
        Raises ValidationError (Pydantic) if the payload is malformed.
        """
        msg = ObstructionReportMessage.model_validate(raw)
        logger.warning(
            "ObstructionAdapter: obstruction report received — no-op (MVP)",
            extra={
                "session_id": msg.session_id,
                "waypoint_id": msg.waypoint_id,
                "severity": msg.severity,
                "description": msg.description,
            },
        )
        # TODO: When live re-routing is implemented, call:
        #   await routing_agent.handle_obstruction(msg)
        # For now: no-op. The Routing Agent does NOT own obstruction detection.
        return msg
