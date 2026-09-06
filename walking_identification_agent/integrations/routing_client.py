"""Swap send_to_routing_agent() for your team's actual inter-agent transport
(direct function call, message queue, or an MCP tool call to the Routing Agent)."""

from api_schemas.routing_payload import RerouteRequest


def send_to_routing_agent(request: RerouteRequest) -> None:
    print(f"[-> ROUTING AGENT] {request.to_dict()}")
