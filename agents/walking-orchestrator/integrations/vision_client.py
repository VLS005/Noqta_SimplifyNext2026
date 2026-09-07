"""
Camera/vision integration.

Per the hackathon architecture decision, the camera is NEVER streamed
continuously - it only takes a triggered snapshot burst when this agent
calls one of these functions. Swap the bodies below for a real call to
Google Vision / a Bedrock vision model once the happy-path demo works.
"""

from typing import Optional
from domain.models import GPSPoint


def check_obstruction(position: GPSPoint) -> Optional[str]:
    """Trigger a snapshot and ask the vision model if there's a visible
    obstruction in the path. Returns a description, or None if clear."""
    return None


def find_anchor_point(position: GPSPoint) -> Optional[dict]:
    """Trigger a snapshot and ask the vision model to find a nearby physical
    anchor (handrail, wall, doorway, bollard, etc.) to help re-orient the user."""
    return {"object": "handrail", "direction": "right", "distance_m": 2}
