from app.config import Settings
from app.models import (
    Action, AgentDecision, CheckRequest, Evidence, HapticCommand, RoutingHandoff,
    TransitArrival, VisionObservation,
)
from app.tools.routing import RoutingTool
from app.tools.transit import TransitTool
from app.tools.vision import VisionTool


class TransitOrchestrator:
    """Bounded autonomous agent: perceive -> reconcile -> decide -> act.

    The model perceives the scene, while explicit safety policy owns the final boarding
    decision. This prevents an LLM from confidently telling a blind user to board the
    wrong vehicle.
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self.transit = TransitTool(settings)
        self.vision = VisionTool(settings)
        self.routing = RoutingTool(settings)

    async def run(self, request: CheckRequest) -> AgentDecision:
        journey = request.journey
        trace = ["GOAL: safely board expected service or obtain an accessible alternative"]

        arrival = await self.transit.get_arrival(
            journey.bus_stop_code, journey.expected_bus, request.demo_scenario
        )
        trace.append(f"TOOL transit: service={arrival.service_no}, eta={arrival.eta_minutes}m, source={arrival.source}")

        should_look = bool(request.cane_image_base64) or arrival.eta_minutes <= self.settings.arrival_window_minutes or request.demo_scenario is not None
        vision: VisionObservation | None = None
        if should_look and request.cane_telemetry.camera_ready:
            trace.append("ACTION: autonomously trigger cane-mounted camera scan")
            vision = await self.vision.inspect(
                journey.expected_bus,
                request.cane_image_base64,
                request.cane_image_media_type,
                request.demo_scenario,
            )
            trace.append(
                f"TOOL vision: visible={vision.bus_visible}, service={vision.service_number}, "
                f"off_service={vision.off_service}, confidence={vision.confidence:.2f}"
            )
        elif not should_look:
            trace.append("POLICY: camera not triggered because bus is outside arrival window")
        else:
            trace.append("DEVICE: cane camera unavailable; fall back to safe waiting instruction")

        decision = self._decide(request, arrival, vision, trace)
        if decision.routing_handoff:
            route = await self.routing.request_alternatives(decision.routing_handoff)
            trace.append(f"TOOL routing: {route.get('status', 'received')}")
            if route.get("recommended_route"):
                decision.spoken_instruction += f" Alternative found: {route['recommended_route']}"
        decision.trace = trace
        return decision

    def _decide(
        self,
        request: CheckRequest,
        arrival: TransitArrival,
        vision: VisionObservation | None,
        trace: list[str],
    ) -> AgentDecision:
        j = request.journey
        eta = arrival.eta_minutes
        evidence = Evidence(transit=arrival, vision=vision)

        def result(action, code, speech, haptic, confidence, handoff=None):
            trace.append(f"DECISION {action}: {code}")
            haptic_command = HapticCommand(**haptic)
            return AgentDecision(action=action, reason_code=code, spoken_instruction=speech,
                                 haptic_command=haptic_command, confidence=confidence,
                                 evidence=evidence, routing_handoff=handoff, trace=trace)

        def reroute(reason: str, speech: str, confidence: float):
            handoff = RoutingHandoff(
                reason=reason,
                expected_bus=j.expected_bus,
                current_stop=j.bus_stop_code,
                destination=j.destination,
                latest_eta_minutes=eta,
                accessibility_preferences=j.preferences,
                excluded_services=[j.expected_bus] if reason == "BUS_OFF_SERVICE" else [],
            )
            return result(Action.REROUTE, reason, speech,
                          {"pattern": "LONG-SHORT-LONG", "actuators": ["CENTER"], "priority": "high"},
                          confidence, handoff)

        if vision and vision.off_service and vision.confidence >= 0.75:
            return reroute("BUS_OFF_SERVICE", f"Bus {j.expected_bus} is off service. Do not board. I am finding an accessible alternative.", vision.confidence)

        if vision and vision.bus_visible and vision.service_number:
            if vision.service_number == j.expected_bus and vision.confidence >= 0.80:
                return result(Action.BOARD, "EXPECTED_BUS_CONFIRMED",
                              f"Bus {j.expected_bus} is here and confirmed. Prepare to board; keep the boarding edge ahead of you.",
                              {"pattern": "SWEEP-FORWARD", "actuators": ["LEFT", "CENTER", "RIGHT"], "priority": "high", "repeat": 2},
                              vision.confidence)
            if vision.service_number != j.expected_bus and vision.confidence >= 0.75:
                return result(Action.IGNORE_BUS, "WRONG_BUS",
                              f"This is bus {vision.service_number}, not {j.expected_bus}. Do not board; continue waiting.",
                              {"pattern": "ONE-LONG", "actuators": ["CENTER"], "priority": "high"},
                              vision.confidence)

        if eta >= self.settings.late_threshold_minutes:
            return reroute("BUS_LATE", f"Bus {j.expected_bus} is now {eta} minutes away. I am checking faster accessible routes.", 0.90)

        if eta <= self.settings.arrival_window_minutes and (not vision or not vision.bus_visible):
            return result(Action.CHECK_CAMERA, "API_VISION_CONFLICT",
                          f"The arrival feed says bus {j.expected_bus} is due, but I cannot confirm it visually. "
"Stay safely back from the road. Sweep the cane slowly from left to right so I can scan again. "
"You will feel alternating left and right pulses.",
                          {"pattern": "SCAN-AGAIN", "actuators": ["LEFT", "RIGHT"], "priority": "high"}, 0.72)

        return result(Action.WAIT, "BUS_EXPECTED_SOON",
                      f"Bus {j.expected_bus} is expected in {eta} minutes. I will check again near arrival.",
                      {"pattern": "ONE-SHORT", "actuators": ["CENTER"]}, 0.88)
