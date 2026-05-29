"""
Approximate city coordinates for quick testing (no geocoding API).
"""

from __future__ import annotations

# name -> (latitude, longitude)
CITIES: dict[str, tuple[float, float]] = {
    "new york": (40.7128, -74.0060),
    "london": (51.5074, -0.1278),
    "paris": (48.8566, 2.3522),
    "tokyo": (35.6762, 139.6503),
    "sydney": (-33.8688, 151.2093),
    "los angeles": (34.0522, -118.2437),
    "chicago": (41.8781, -87.6298),
    "toronto": (43.6532, -79.3832),
    "berlin": (52.5200, 13.4050),
    "mumbai": (19.0760, 72.8777),
    "new_delhi": (28.6139, 77.2090),
    "singapore": (1.3521, 103.8198),
    "denver": (39.7392, -104.9903),
    "seattle": (47.6062, -122.3321),
    "boston": (42.3601, -71.0589),
    "san francisco": (37.7749, -122.4194),
}


def resolve_city(city: str) -> tuple[float, float]:
    """Return (lat, lon) for a known city name (case-insensitive)."""
    key = city.strip().lower()
    if key not in CITIES:
        known = ", ".join(sorted(CITIES))
        raise KeyError(f"Unknown city '{city}'. Known: {known}")
    return CITIES[key]
