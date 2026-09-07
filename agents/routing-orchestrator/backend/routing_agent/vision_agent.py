"""
Merged Vision Agent Adapter.
Enforces the 5m proximity gate, captures real-time frames from OpenCV,
processes them via AWS Rekognition, and returns navigation instructions.
"""
from __future__ import annotations

from dotenv import load_dotenv

import logging
import math
from typing import Any
import boto3
import cv2

load_dotenv()

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
        from backend.routing_agent.bedrock_client import BedrockClient
        # Initialize the AWS Rekognition Client (uses credentials from env/aws config)
        self.rekognition_client = boto3.client('rekognition', region_name='us-east-1')
        self._bedrock = BedrockClient()
        
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

    # ─── Burst Screenshots and Bounding Boxes ─────────────────────────────────

    def _draw_bounding_boxes(self, img: np.ndarray, labels: list[dict]) -> np.ndarray:
        height, width = img.shape[:2]
        for label in labels:
            name = label.get('Name', 'object')
            for instance in label.get('Instances', []):
                box = instance.get('BoundingBox', {})
                if not box:
                    continue
                
                x = int(box.get('Left', 0) * width)
                y = int(box.get('Top', 0) * height)
                w = int(box.get('Width', 0) * width)
                h = int(box.get('Height', 0) * height)
                
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(img, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        return img

    # ─── Inbound ──────────────────────────────────────────────────────────────

    def parse(self, raw: dict[str, Any]) -> VisionResponseMessage:
        """Parse and validate an inbound vision response structure."""
        msg = VisionResponseMessage.model_validate(raw)
        logger.info("VisionAgent: Local parsing validated.", extra={"session_id": msg.session_id})
        return msg

    # ─── Outbound ─────────────────────────────────────────────────────────────

    async def send(self, message: BaseMessage, on_obstruction_callback=None) -> bool:
        import time
        import asyncio
        import numpy as np

        if not isinstance(message, VisionRequestMessage):
            return False
            
        settings = get_settings()
        gate_m = settings.vision_proximity_gate_m

        if message.user_distance_m > gate_m:
            raise ProximityGateError(f"Gate {gate_m}m exceeded.")

        if not self.is_moving():
            return False

        if self.api_call_count >= self.MAX_ALLOWED_CALLS:
            raise BudgetExceededError("AWS Safety Limit reached.")

        logger.info("VisionAgent: Starting live session for 30s.")
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            logger.error("VisionAgent: Failed to open webcam.")
            return False

        start_time = time.time()
        last_api_call = 0
        last_labels = []

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                current_time = time.time()
                
                if current_time - start_time > 30:
                    logger.info("VisionAgent: 30s timeout reached.")
                    break
                    
                height, width = frame.shape[:2]
                if height > 720:
                    scale = 720 / float(height)
                    frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

                # Throttle AWS call to 1 every 3 seconds
                if current_time - last_api_call >= 3.0:
                    last_api_call = current_time
                    ret_encode, buffer = cv2.imencode('.jpg', frame)
                    if ret_encode:
                        self.api_call_count += 1
                        try:
                            loop = asyncio.get_running_loop()
                            response = await loop.run_in_executor(
                                None, 
                                lambda: self.rekognition_client.detect_labels(
                                    Image={'Bytes': buffer.tobytes()},
                                    MaxLabels=15,
                                    MinConfidence=70.0
                                )
                            )
                            last_labels = response.get('Labels', [])
                            
                            if last_labels:
                                guidance_text = self.sectorize_and_guide(last_labels)
                                if "completely clear" not in guidance_text and on_obstruction_callback:
                                    # Layer 2: Multimodal analysis
                                    system_prompt = "You are a spatial reasoning AI. Analyze this image and the accompanying Rekognition labels. Describe any safety hazards for a blind person walking forward and provide directions on how to move away from the obstructions."
                                    user_prompt = f"Rekognition Labels: {last_labels}\n\nWhat are the hazards, and what directions should I take to avoid them safely?"
                                    try:
                                        deep_context = await self._bedrock.invoke_model(
                                            system_prompt=system_prompt,
                                            user_prompt=user_prompt,
                                            model_tier="nova-pro",
                                            image_bytes=buffer.tobytes()
                                        )
                                        # Pass the rich multimodal response to the callback (Layer 3)
                                        asyncio.create_task(on_obstruction_callback(deep_context))
                                    except Exception as bedrock_err:
                                        logger.error(f"VisionAgent: Nova Pro failed, falling back to basic guidance: {bedrock_err}")
                                        asyncio.create_task(on_obstruction_callback(guidance_text))
                        except Exception as e:
                            logger.error(f"Rekognition error: {e}")

                annotated_frame = self._draw_bounding_boxes(frame.copy(), last_labels)
                cv2.imshow("Live Vision Feedback", annotated_frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    logger.info("VisionAgent: User quit live session.")
                    break
                
                await asyncio.sleep(0.01)

        finally:
            cap.release()
            cv2.destroyAllWindows()
            # On mac we often need 4 waitKeys to flush events and close the window properly
            for _ in range(4):
                cv2.waitKey(1)
            
        return True
