"""
Generates 2-3 candidate RoutePlans for a given origin/destination.
Uses REAL Google Directions API for routes and OpenStreetMap Overpass API
for blind-friendly accessibility features (tactile paving, steps, lighting, etc.).

Pipeline:
1. Call Google Directions API → get 2-3 walking routes with polylines
2. For each route, query OSM Overpass API → find accessibility features
3. Parse combined data into RoutePlan JSON → pass to AccessibilityAgent
"""
from __future__ import annotations

import logging
import uuid
import json
import os
import math

import httpx
import polyline

from backend.routing_agent.models import (
    AccessibilityScore,
    LightingQuality,
    RoutePlan,
    Waypoint,
    MobilityProfile,
)
from backend.routing_agent.config import get_settings
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# Google Directions API endpoint
_GOOGLE_DIRECTIONS_URL = "https://maps.googleapis.com/maps/api/directions/json"

# OSM Overpass API endpoint (free, no key required)
_OVERPASS_URL = "https://overpass-api.de/api/interpreter"


class DirectionFinderAgent:
    """
    Independent route planning sub-agent.
    Calls REAL APIs:
      1. Google Directions API → walking routes with waypoints
      2. OpenStreetMap Overpass API → accessibility features per route
    Returns parsed RoutePlan objects for the AccessibilityAgent.
    """

    def __init__(self, bedrock=None) -> None:
        """bedrock param kept for backward compatibility but is no longer used."""
        self._settings = get_settings()
        self._maps_api_key = self._settings.google_maps_api_key

    async def generate_candidates(
        self,
        session_id: str,
        origin_label: str,
        destination_label: str,
        profile: MobilityProfile,
        origin_lat: float | None = None,
        origin_lon: float | None = None,
        destination_lat: float | None = None,
        destination_lon: float | None = None,
    ) -> list[RoutePlan]:
        """
        Return a list of candidate RoutePlans from REAL API data.
        Accessibility scores and ETAs are NOT set here — they are computed
        downstream by AccessibilityAgent and ETAAgent respectively.
        """
        logger.info(
            "DirectionFinderAgent: generating candidates via Google Directions + OSM",
            extra={
                "session_id": session_id,
                "origin": origin_label,
                "destination": destination_label,
            },
        )
        print(f"\n{'='*70}")
        print(f"[DirectionFinderAgent] Starting route generation")
        print(f"[DirectionFinderAgent] Origin: {origin_label}")
        print(f"[DirectionFinderAgent] Destination: {destination_label}")
        print(f"{'='*70}")

        # Step 1: Call Google Directions API
        google_routes = await self._fetch_google_directions(
            origin_label, destination_label,
            origin_lat, origin_lon, destination_lat, destination_lon,
        )
        print(f"[DirectionFinderAgent] Google Directions returned {len(google_routes)} routes")

        # Step 2 + 3: For each route, query OSM and build RoutePlan
        candidates = []
        for i, route_data in enumerate(google_routes):
            print(f"\n[DirectionFinderAgent] --- Processing route {i+1}/{len(google_routes)} ---")

            # Get decoded polyline points for OSM query
            decoded_points = route_data["decoded_points"]
            print(f"[DirectionFinderAgent] Route has {len(decoded_points)} polyline points")

            # Step 2: Query OSM Overpass for accessibility features
            osm_features = await self._fetch_osm_accessibility(decoded_points)
            print(f"[DirectionFinderAgent] OSM returned: {json.dumps(osm_features, indent=2)}")

            # Step 3: Build RoutePlan from combined data
            route_plan = self._build_route_plan(
                session_id, origin_label, destination_label,
                route_data, osm_features, i,
            )
            candidates.append(route_plan)
            print(
                f"[DirectionFinderAgent] Built RoutePlan: '{route_plan.label}' | "
                f"distance={route_plan.distance_m:.0f}m | "
                f"waypoints={len(route_plan.waypoints)} | "
                f"instructions={len(route_plan.raw_instructions)}"
            )

        logger.info(f"DirectionFinderAgent: {len(candidates)} candidates generated from real APIs")
        print(f"\n[DirectionFinderAgent] Total candidates generated: {len(candidates)}")
        return candidates

    # ═══════════════════════════════════════════════════════════════════════════
    # GOOGLE DIRECTIONS API
    # ═══════════════════════════════════════════════════════════════════════════

    async def _fetch_google_directions(
        self,
        origin_label: str,
        destination_label: str,
        origin_lat: float | None,
        origin_lon: float | None,
        destination_lat: float | None,
        destination_lon: float | None,
    ) -> list[dict]:
        """
        Call Google Directions API for walking routes.
        Returns list of route dicts with decoded polyline points.
        """
        if not self._maps_api_key:
            raise RuntimeError(
                "GOOGLE_MAPS_API_KEY is not set. "
                "Get it from Google Cloud Console → APIs & Services → Credentials. "
                "Enable the Directions API."
            )

        # Geocode the labels if coordinates are not provided
        if not origin_lat or not origin_lon:
            origin_lat, origin_lon = await self._geocode_label(origin_label)
        if not destination_lat or not destination_lon:
            destination_lat, destination_lon = await self._geocode_label(destination_label)

        # Build origin/destination strings using the coordinates
        origin = f"{origin_lat},{origin_lon}"
        destination = f"{destination_lat},{destination_lon}"

        params = {
            "origin": origin,
            "destination": destination,
            "mode": "transit",
            "alternatives": "true",      # Request multiple routes
            "key": self._maps_api_key,
        }

        print(f"[DirectionFinderAgent] Calling Google Directions API (mode=transit)...")
        print(f"[DirectionFinderAgent] Origin: {origin_label} ({origin}) | Destination: {destination_label} ({destination})")

        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.get(_GOOGLE_DIRECTIONS_URL, params=params)
            response.raise_for_status()
            data = response.json()

        print(f"[DirectionFinderAgent] Google API status: {data.get('status')}")

        if data.get("status") != "OK":
            error_msg = data.get("error_message", data.get("status", "Unknown error"))
            print(f"[DirectionFinderAgent] ERROR: Google API returned: {error_msg}")
            raise RuntimeError(f"Google Directions API error: {error_msg}")

        routes = []
        for route in data.get("routes", []):
            leg = route["legs"][0]  # Walking = single leg

            # Decode the overview polyline to get route coordinates
            overview_polyline = route.get("overview_polyline", {}).get("points", "")
            decoded_points = polyline.decode(overview_polyline) if overview_polyline else []

            # Extract step-by-step instructions
            steps = leg.get("steps", [])
            raw_instructions = []
            step_waypoints = []

            for step_idx, step in enumerate(steps):
                travel_mode = step.get("travel_mode", "UNKNOWN")
                
                if travel_mode == "TRANSIT":
                    transit_details = step.get("transit_details", {})
                    line = transit_details.get("line", {}).get("name", "") or transit_details.get("line", {}).get("short_name", "")
                    vehicle = transit_details.get("line", {}).get("vehicle", {}).get("name", "transit")
                    num_stops = transit_details.get("num_stops", "")
                    headsign = transit_details.get("headsign", "")
                    boarding_stop = transit_details.get("departure_stop", {}).get("name", "unknown")
                    alighting_stop = transit_details.get("arrival_stop", {}).get("name", "unknown")
                    text_instr = f"[TRANSIT] Board {vehicle} {line} at {boarding_stop} towards {headsign} for {num_stops} stops. Alight at {alighting_stop}."
                else:
                    html_instr = step.get("html_instructions", "")
                    text_instr = f"[WALKING] {self._strip_html(html_instr)}"

                if "steps" in step:
                    for sub_step in step["steps"]:
                        sub_html = sub_step.get("html_instructions", "")
                        sub_text = f"[{travel_mode}] {self._strip_html(sub_html)}"
                        raw_instructions.append(sub_text)
                        
                        start_loc = sub_step.get("start_location", {})
                        distance_m = sub_step.get("distance", {}).get("value", 0)
                        duration_s = sub_step.get("duration", {}).get("value", 0)
                        step_waypoints.append({
                            "lat": start_loc.get("lat", 0),
                            "lon": start_loc.get("lng", 0),
                            "label": sub_text[:60] if sub_text else f"Step {len(step_waypoints) + 1}",
                            "distance_to_next_m": float(distance_m),
                            "duration_to_next_s": int(duration_s),
                            "travel_mode_to_next": travel_mode,
                        })
                else:
                    raw_instructions.append(text_instr)
                    start_loc = step.get("start_location", {})
                    distance_m = step.get("distance", {}).get("value", 0)
                    duration_s = step.get("duration", {}).get("value", 0)

                    step_waypoints.append({
                        "lat": start_loc.get("lat", 0),
                        "lon": start_loc.get("lng", 0),
                        "label": text_instr[:60] if text_instr else f"Step {len(step_waypoints) + 1}",
                        "distance_to_next_m": float(distance_m),
                        "duration_to_next_s": int(duration_s),
                        "travel_mode_to_next": travel_mode,
                    })

            # Add the final destination as last waypoint
            end_loc = leg.get("end_location", {})
            step_waypoints.append({
                "lat": end_loc.get("lat", 0),
                "lon": end_loc.get("lng", 0),
                "label": destination_label,
                "distance_to_next_m": None,
                "duration_to_next_s": None,
                "travel_mode_to_next": None,
            })

            route_info = {
                "summary": route.get("summary", f"Route via {leg.get('start_address', 'unknown')}"),
                "distance_m": float(leg.get("distance", {}).get("value", 0)),
                "duration_s": int(leg.get("duration", {}).get("value", 0)),
                "decoded_points": decoded_points,
                "raw_instructions": raw_instructions,
                "waypoints": step_waypoints,
            }
            routes.append(route_info)

            print(
                f"[DirectionFinderAgent] Google route: '{route_info['summary']}' | "
                f"distance={route_info['distance_m']:.0f}m | "
                f"duration={route_info['duration_s']}s | "
                f"steps={len(raw_instructions)}"
            )

        return routes[:3]  # Cap at 3 routes

    async def _geocode_label(self, label: str) -> tuple[float, float]:
        """Convert a text label to lat/lon using Google Geocoding API."""
        print(f"[DirectionFinderAgent] Geocoding label: {label}")
        params = {
            "address": label,
            "key": self._maps_api_key,
        }
        url = "https://maps.googleapis.com/maps/api/geocode/json"
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(url, params=params)
            response.raise_for_status()
            data = response.json()
        
        if data.get("status") == "OK" and data.get("results"):
            loc = data["results"][0]["geometry"]["location"]
            return loc["lat"], loc["lng"]
        
        print(f"[DirectionFinderAgent] Geocoding failed for '{label}': {data.get('status')}")
        raise RuntimeError(f"Could not geocode '{label}'. Please provide more specific input.")

    # ═══════════════════════════════════════════════════════════════════════════
    # OPENSTREETMAP OVERPASS API
    # ═══════════════════════════════════════════════════════════════════════════

    async def _fetch_osm_accessibility(self, route_points: list[tuple]) -> dict:
        """
        Query OSM Overpass API for accessibility features near the route.
        Checks for: tactile_paving, kerb, steps, lighting, surface quality.
        Returns aggregated accessibility features dict.
        """
        if not route_points:
            print("[DirectionFinderAgent] No route points for OSM query — using defaults")
            return self._default_osm_features()

        # Sample points along the route to avoid huge queries
        # Take every Nth point to get ~20 sample points max
        n = max(1, len(route_points) // 20)
        sampled = route_points[::n]

        # Build a bounding box around the route with a 50m buffer
        lats = [p[0] for p in sampled]
        lons = [p[1] for p in sampled]
        buffer_deg = 0.0005  # ~50 metres
        min_lat = min(lats) - buffer_deg
        max_lat = max(lats) + buffer_deg
        min_lon = min(lons) - buffer_deg
        max_lon = max(lons) + buffer_deg

        # Overpass QL query for accessibility-relevant features
        query = f"""
[out:json][timeout:10];
(
  // Tactile paving
  way["tactile_paving"="yes"]({min_lat},{min_lon},{max_lat},{max_lon});
  node["tactile_paving"="yes"]({min_lat},{min_lon},{max_lat},{max_lon});
  // Kerb info
  node["kerb"]({min_lat},{min_lon},{max_lat},{max_lon});
  // Steps / stairs
  way["highway"="steps"]({min_lat},{min_lon},{max_lat},{max_lon});
  // Lighting
  way["lit"="yes"]({min_lat},{min_lon},{max_lat},{max_lon});
  way["lit"="no"]({min_lat},{min_lon},{max_lat},{max_lon});
  // Surface quality
  way["surface"]({min_lat},{min_lon},{max_lat},{max_lon});
  // Footways with wheelchair info
  way["wheelchair"]({min_lat},{min_lon},{max_lat},{max_lon});
);
out count;
out body;
"""
        print(f"[DirectionFinderAgent] Querying OSM Overpass API (bbox: {min_lat:.4f},{min_lon:.4f} to {max_lat:.4f},{max_lon:.4f})...")

        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                response = await client.post(
                    _OVERPASS_URL,
                    data={"data": query},
                    headers={
                        "Content-Type": "application/x-www-form-urlencoded",
                        "User-Agent": "SimplifyNext/1.0 (contact@simplifynext.com)"
                    },
                )
                response.raise_for_status()
                osm_data = response.json()

            elements = osm_data.get("elements", [])
            print(f"[DirectionFinderAgent] OSM returned {len(elements)} elements")

            return self._parse_osm_features(elements)

        except Exception as exc:
            print(f"[DirectionFinderAgent] OSM query failed: {exc} — using defaults")
            logger.warning(f"DirectionFinderAgent: OSM query failed: {exc}")
            return self._default_osm_features()

    @staticmethod
    def _parse_osm_features(elements: list[dict]) -> dict:
        """Parse raw OSM Overpass elements into accessibility feature counts."""
        tactile_count = 0
        steps_count = 0
        lit_yes_count = 0
        lit_no_count = 0
        kerb_count = 0
        surface_good_count = 0
        surface_bad_count = 0
        wheelchair_yes_count = 0
        total_ways = 0

        good_surfaces = {"asphalt", "concrete", "paved", "paving_stones", "sett"}
        bad_surfaces = {"gravel", "grass", "dirt", "sand", "mud", "cobblestone", "unpaved"}

        for elem in elements:
            tags = elem.get("tags", {})

            if tags.get("tactile_paving") == "yes":
                tactile_count += 1

            if tags.get("highway") == "steps":
                steps_count += 1

            if tags.get("lit") == "yes":
                lit_yes_count += 1
            elif tags.get("lit") == "no":
                lit_no_count += 1

            if "kerb" in tags:
                kerb_count += 1

            surface = tags.get("surface", "")
            if surface in good_surfaces:
                surface_good_count += 1
            elif surface in bad_surfaces:
                surface_bad_count += 1

            if tags.get("wheelchair") in ("yes", "designated"):
                wheelchair_yes_count += 1

            if elem.get("type") == "way":
                total_ways += 1

        features = {
            "tactile_paving_count": tactile_count,
            "steps_count": steps_count,
            "lit_yes_count": lit_yes_count,
            "lit_no_count": lit_no_count,
            "kerb_count": kerb_count,
            "surface_good_count": surface_good_count,
            "surface_bad_count": surface_bad_count,
            "wheelchair_yes_count": wheelchair_yes_count,
            "total_ways_scanned": total_ways,
            # Derived booleans
            "has_tactile_paving": tactile_count > 0,
            "has_steps": steps_count > 0,
            "is_well_lit": lit_yes_count > lit_no_count,
            "is_step_free": steps_count == 0,
        }

        print(f"[DirectionFinderAgent] OSM parsed features: {json.dumps(features)}")
        return features

    @staticmethod
    def _default_osm_features() -> dict:
        """Default features when OSM is unavailable."""
        return {
            "tactile_paving_count": 0,
            "steps_count": 0,
            "lit_yes_count": 0,
            "lit_no_count": 0,
            "kerb_count": 0,
            "surface_good_count": 0,
            "surface_bad_count": 0,
            "wheelchair_yes_count": 0,
            "total_ways_scanned": 0,
            "has_tactile_paving": False,
            "has_steps": False,
            "is_well_lit": False,
            "is_step_free": True,
        }

    # ═══════════════════════════════════════════════════════════════════════════
    # BUILD ROUTE PLAN FROM API DATA
    # ═══════════════════════════════════════════════════════════════════════════

    @staticmethod
    def _build_route_plan(
        session_id: str,
        origin_label: str,
        destination_label: str,
        google_route: dict,
        osm_features: dict,
        route_index: int,
    ) -> RoutePlan:
        """Build a RoutePlan from Google Directions + OSM Overpass data."""
        # Build waypoints from Google steps
        waypoints = []
        for seq, wp_data in enumerate(google_route["waypoints"]):
            waypoints.append(
                Waypoint(
                    id=str(uuid.uuid4()),
                    sequence=seq,
                    lat=wp_data["lat"],
                    lon=wp_data["lon"],
                    label=wp_data["label"],
                    fine_motor_required=False,
                    distance_to_next_m=wp_data.get("distance_to_next_m"),
                    duration_to_next_s=wp_data.get("duration_to_next_s"),
                    travel_mode_to_next=wp_data.get("travel_mode_to_next"),
                )
            )

        # Determine lighting quality from OSM data
        if osm_features["lit_yes_count"] > 0 and osm_features["lit_no_count"] == 0:
            lighting = LightingQuality.BRIGHT
        elif osm_features["lit_yes_count"] > osm_features["lit_no_count"]:
            lighting = LightingQuality.ADEQUATE
        elif osm_features["lit_no_count"] > 0:
            lighting = LightingQuality.DIM
        else:
            lighting = LightingQuality.UNKNOWN

        # Compute obstruction risk from surface quality
        total_surface = osm_features["surface_good_count"] + osm_features["surface_bad_count"]
        if total_surface > 0:
            obstruction_risk = round(osm_features["surface_bad_count"] / total_surface, 2)
        else:
            obstruction_risk = 0.2  # Unknown surfaces get moderate risk

        # Crowding estimate: 0.3 default (modest), no live data available from OSM
        crowding_estimate = 0.3

        accessibility_score = AccessibilityScore(
            step_free=osm_features["is_step_free"],
            tactile_paving=osm_features["has_tactile_paving"],
            lighting_quality=lighting,
            crowding_estimate=crowding_estimate,
            obstruction_risk=obstruction_risk,
            composite_score=0.0,  # Filled by AccessibilityAgent
        )

        label = google_route.get("summary", f"Route {route_index + 1}")

        return RoutePlan(
            id=str(uuid.uuid4()),
            session_id=session_id,
            label=label,
            origin_label=origin_label,
            destination_label=destination_label,
            waypoints=waypoints,
            accessibility_score=accessibility_score,
            estimated_duration_s=0,  # Filled by ETAAgent
            distance_m=google_route["distance_m"],
            provider_duration_s=google_route["duration_s"],
            raw_instructions=google_route["raw_instructions"],
            micro_instructions=[],  # Filled by InstructionParsingAgent
            is_stub=False,  # This is REAL API data!
        )

    # ═══════════════════════════════════════════════════════════════════════════
    # UTILITIES
    # ═══════════════════════════════════════════════════════════════════════════

    @staticmethod
    def _strip_html(html_str: str) -> str:
        """Remove HTML tags from Google Directions instruction text."""
        import re
        clean = re.sub(r"<[^>]+>", "", html_str)
        # Collapse whitespace
        clean = re.sub(r"\s+", " ", clean).strip()
        return clean
