"""
Visible solar-system bodies via Skyfield (no heavy ephemeris beyond de421.bsp).
"""

from __future__ import annotations

from datetime import datetime
from functools import lru_cache
from typing import Any

from skyfield.api import load, wgs84

from backend.astronomy.visible_stars import _parse_timestamp, _timescale

# Skyfield ephemeris keys → display names
BODIES: list[tuple[str, str]] = [
    ("mercury", "Mercury"),
    ("venus", "Venus"),
    ("mars barycenter", "Mars"),
    ("jupiter barycenter", "Jupiter"),
    ("saturn barycenter", "Saturn"),
]


@lru_cache(maxsize=1)
def _ephemeris():
    return load("de421.bsp")


def visible_planets(
    latitude: float,
    longitude: float,
    timestamp: datetime | str,
    *,
    min_altitude_deg: float = 5.0,
) -> list[dict[str, Any]]:
    """Return planets above the horizon at observer time/place."""
    ts = _timescale()
    t = ts.from_datetime(_parse_timestamp(timestamp))
    earth = _ephemeris()["earth"]
    location = earth + wgs84.latlon(latitude, longitude)
    eph = _ephemeris()

    out: list[dict[str, Any]] = []
    for key, name in BODIES:
        try:
            body = eph[key]
        except KeyError:
            continue
        alt, az, _ = location.at(t).observe(body).apparent().altaz()
        alt_deg = float(alt.degrees)
        if alt_deg <= min_altitude_deg:
            continue
        out.append({
            "name": name,
            "type": "planet",
            "altitude_deg": round(alt_deg, 3),
            "azimuth_deg": round(float(az.degrees), 3),
        })

    out.sort(key=lambda p: -p["altitude_deg"])
    return out
