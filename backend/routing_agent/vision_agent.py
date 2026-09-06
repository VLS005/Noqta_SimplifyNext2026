"""
Merged Vision Agent Adapter.
Enforces the 5m proximity gate, captures real-time frames from OpenCV,
processes them via AWS Rekognition, and returns navigation instructions.
"""
from __future__ import annotations

import logging
import math
from typing import Any
import boto3
import cv2

from backend.routing_agent.config import get_settings
from backend.routing_agent.models import (
    BaseMessage,
    VisionRequestMessage,
    VisionResponseMessage,
)

logger = logging.getLogger(__name__)

class ProximityGateError(Exception):
    """Raised when a vision request is attempted outside the 5 m proximity gate."""

class BudgetExceededError(Exception):
    """Raised when the $10 testing budget limit is hit to prevent overcharges."""


class VisionAgent:
    """
    Unified local Vision Agent.
    1. Enforces the 5m safety gate from the routing agent.
    2. Simulates/Tracks frame sampling & motion.
    3. Captures real frames from webcam (cv2).
    4. Calls AWS Rekognition directly and generates local structural instructions.
    """

    def __init__(self):
        # Initialize the AWS Rekognition Client (uses credentials from env/aws config)
        self.rekognition_client = boto3.client('rekognition', region_name='us-east-1')
        
        # Budget Safeguard Tracking ($10 budget / $0.001 per call = 10,000 max calls)
        self.api_call_count = 0
        self.MAX_ALLOWED_CALLS = 10000 

    def agent_name(self) -> str:
        return "vision_agent"

    # ─── Hardware Interactions ─────────────────────────────────────────────

    def is_moving(self) -> bool:
        """Stub method simulating mobile accelerometer. Always true for testing."""
        return True

    def capture_frame(self) -> bytes:
        """
        Captures a live frame from the webcam using OpenCV, rescales it to 720p,
        and encodes it to JPEG bytes.
        """
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            logger.error("VisionAgent: Failed to open webcam. Returning mock frame.")
            return self._mock_frame()

        ret, frame = cap.read()
        cap.release()

        if not ret or frame is None:
            logger.error("VisionAgent: Failed to read frame from webcam. Returning mock frame.")
            return self._mock_frame()

        # Downscale to 720p to save bandwidth/API payload limits
        height, width = frame.shape[:2]
        if height > 720:
            scale = 720 / float(height)
            frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

        # Encode to JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            logger.error("VisionAgent: Failed to encode frame. Returning mock frame.")
            return self._mock_frame()

        return buffer.tobytes()

    def _mock_frame(self) -> bytes:
        """Fallback 1x1 black GIF if webcam fails."""
        return b'GIF89a\x01\x00\x01\x00\x80\x00\x00\x00\x00\x00\xff\xff\xff!\xf9\x04\x01\x00\x00\x00\x00,\x00\x00\x00\x00\x01\x00\x01\x00\x00\x02\x02D\x01\x00;'

    # ─── Core Spatial Processing Logic ────────────────────────────────────────

    def sectorize_and_guide(self, rekognition_labels: list[dict[str, Any]]) -> str:
        """Parses AWS bounding boxes into Left, Center, and Right sectors to build instructions."""
        sectors = {"left": [], "center": [], "right": []}
        
        for label in rekognition_labels:
            name = label.get('Name', 'object')
            for instance in label.get('Instances', []):
                box = instance.get('BoundingBox', {})
                box_left = box.get('Left', 0.5)  # Default middle if missing
                
                # Split camera viewport into 3 semantic vertical columns
                if box_left < 0.33:
                    sectors["left"].append(name)
                elif box_left > 0.66:
                    sectors["right"].append(name)
                else:
                    sectors["center"].append(name)
                    
        # Generate custom spatial spoken output string based on sector density
        if sectors["center"]:
            primary_obstacle = sectors["center"][0]
            left_count = len(sectors["left"])
            right_count = len(sectors["right"])
            
            # Direct the user safely toward the less crowded path
            if left_count <= right_count:
                direction = "left"
                steps = 3 + (left_count * 2)
            else:
                direction = "right"
                steps = 3 + (right_count * 2)
                
            return (
                f"There is a {primary_obstacle} right in front. I see {left_count} obstacles on your left "
                f"and {right_count} obstacles on your right. Clear pathway is to go around by taking "
                f"{steps} steps to the {direction}, then moving straight."
            )
        
        return "Path is completely clear. Continue moving straight forward."

    # ─── Inbound ──────────────────────────────────────────────────────────────

    def parse(self, raw: dict[str, Any]) -> VisionResponseMessage:
        """Parse and validate an inbound vision response structure."""
        msg = VisionResponseMessage.model_validate(raw)
        logger.info("VisionAgent: Local parsing validated.", extra={"session_id": msg.session_id})
        return msg

    # ─── Outbound ─────────────────────────────────────────────────────────────

    async def send(self, message: BaseMessage) -> bool:
        """Dispatches outbound logic. Always enforces proximity gate first."""
        if not isinstance(message, VisionRequestMessage):
            logger.error(
                "VisionAgent.send: expected VisionRequestMessage",
                extra={"type": type(message).__name__},
            )
            return False
            
        # 1. Enforce the 5-meter proximity safety gate
        settings = get_settings()
        gate_m = settings.vision_proximity_gate_m

        if message.user_distance_m > gate_m:
            raise ProximityGateError(
                f"Vision request rejected: user is {message.user_distance_m:.1f} m from waypoint "
                f"(gate is {gate_m} m). Vision is only allowed within {gate_m} m of fine-motor targets."
            )

        # 2. Enforce the Motion Throttle to protect your $10 budget
        if not self.is_moving():
            logger.info("Vision skipped: User is standing still. Saving API budget.")
            return False

        # 3. Guard against accidental credit overruns
        if self.api_call_count >= self.MAX_ALLOWED_CALLS:
            raise BudgetExceededError("AWS Safety Limit reached! Aborting call to protect your $10 budget.")

        logger.info(
            "VisionAgent: Gate passed. Calling local AWS Rekognition pipeline.",
            extra={"session_id": message.session_id, "waypoint_id": message.waypoint_id}
        )

        try:
            # Capture frame payload data
            frame_bytes = self.capture_frame()
            
            # Send binary image payload directly to AWS
            self.api_call_count += 1
            response = self.rekognition_client.detect_labels(
                Image={'Bytes': frame_bytes},
                MaxLabels=15,
                MinConfidence=70.0
            )
            
            # Process spatial geometric analysis
            labels = response.get('Labels', [])
            guidance_text = self.sectorize_and_guide(labels)
            
            # Print text to terminal (Simulated Text-To-Speech)
            print(f"\n[TTS AUDIO OUTPUT]: {guidance_text}\n")
            
            # Mock generating a valid return response payload to hand back to routing_agent
            return True

        except Exception as e:
            logger.error(f"VisionAgent pipeline failed processing locally: {str(e)}")
            return False
