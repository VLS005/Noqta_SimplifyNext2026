import math
from domain.models import GPSPoint


def haversine_m(p1: GPSPoint, p2: GPSPoint) -> float:
    """Distance in meters between two GPS points."""
    R = 6371000
    phi1, phi2 = math.radians(p1.lat), math.radians(p2.lat)
    dphi = math.radians(p2.lat - p1.lat)
    dlambda = math.radians(p2.lon - p1.lon)
    a = (math.sin(dphi / 2) ** 2
         + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2)
    return 2 * R * math.asin(math.sqrt(a))
