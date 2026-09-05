import base64
import json
import re
import boto3

from app.config import Settings
from app.models import VisionObservation


DEMO_VISION = {
    "on_time": VisionObservation(bus_visible=True, service_number="EXPECTED", confidence=0.96, description="Expected bus approaching"),
    "late": VisionObservation(bus_visible=False, confidence=0.92, description="No bus visible"),
    "wrong_bus": VisionObservation(bus_visible=True, service_number="179", confidence=0.95, description="A different bus is approaching"),
    "off_service": VisionObservation(bus_visible=True, service_number="EXPECTED", off_service=True, confidence=0.98, description="Display reads Off Service"),
    "no_bus": VisionObservation(bus_visible=False, confidence=0.91, description="No bus visible"),
    "conflict": VisionObservation(bus_visible=False, confidence=0.94, description="API claims arrival but no bus is visible"),
}


class VisionTool:
    """Uses a triggered snapshot; never continuously streams video."""

    def __init__(self, settings: Settings):
        self.settings = settings

    async def inspect(
        self,
        expected_bus: str,
        image_base64: str | None,
        media_type: str,
        demo_scenario: str | None = None,
    ) -> VisionObservation:
        if self.settings.demo_mode or not image_base64:
            item = DEMO_VISION.get(demo_scenario or "on_time", DEMO_VISION["on_time"]).model_copy(deep=True)
            if item.service_number == "EXPECTED":
                item.service_number = expected_bus
            if demo_scenario == "wrong_bus" and expected_bus == "179":
                item.service_number = "199"
            item.source = "demo"
            return item

        raw = base64.b64decode(image_base64, validate=True)
        image_format = media_type.split("/")[1].replace("jpg", "jpeg")
        client = boto3.client("bedrock-runtime", region_name=self.settings.aws_region)
        prompt = f"""Inspect this bus-stop snapshot for a blind traveller expecting bus {expected_bus}.
Return ONLY JSON with: bus_visible (boolean), service_number (string or null),
off_service (boolean), confidence (0 to 1), description (short factual sentence).
Do not infer an unreadable service number. Safety is more important than guessing."""
        response = client.converse(
            modelId=self.settings.bedrock_vision_model_id,
            messages=[{"role": "user", "content": [
                {"image": {"format": image_format, "source": {"bytes": raw}}},
                {"text": prompt},
            ]}],
            inferenceConfig={"maxTokens": 300, "temperature": 0},
        )
        text = "".join(
            block.get("text", "") for block in response["output"]["message"]["content"]
        )
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise ValueError("Bedrock vision response did not contain JSON")
        result = VisionObservation.model_validate(json.loads(match.group(0)))
        result.source = "amazon-bedrock"
        return result

