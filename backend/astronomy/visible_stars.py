"""
Compute which catalog stars are above the horizon at a given place and time.

Uses Skyfield for Earth rotation, aberration/nutation (apparent place), and
alt/az in the local horizon frame.
"""

from __future__ import annotations

from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

from skyfield.api import Star, load, wgs84

from backend.astronomy.catalog import filter_by_magnitude, load_catalog


def _parse_timestamp(timestamp: datetime | str) -> datetime:
    """Normalize to timezone-aware UTC."""
    if isinstance(timestamp, str):
        text = timestamp.strip()
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        dt = datetime.fromisoformat(text)
    else:
        dt = timestamp

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


@lru_cache(maxsize=1)
def _timescale():
    return load.timescale()


@lru_cache(maxsize=1)
def _earth():
    """Small JPL ephemeris (~16 MB, cached once by Skyfield)."""
    return load("de421.bsp")["earth"]


def visible_stars(
    latitude: float,
    longitude: float,
    timestamp: datetime | str,
    *,
    max_magnitude: float = 3.5,
    min_altitude_deg: float = 0.0,
    azimuth_deg: float | None = None,
    azimuth_tolerance_deg: float = 45.0,
    catalog_path: str | None = None,
) -> list[dict[str, Any]]:
    """
    Stars from the bright catalog that are above the horizon at the given time.

    Parameters
    ----------
    latitude, longitude
        Observer location in degrees (WGS84). East-positive longitude.
    timestamp
        UTC instant (aware datetime or ISO-8601 string).
    max_magnitude
        Faintest V magnitude to include (lower number = brighter).
    min_altitude_deg
        Minimum altitude above the geometric horizon (degrees).
    azimuth_deg, azimuth_tolerance_deg
        When set, restrict to stars within ±tolerance of this compass bearing.

    Returns
    -------
    List of dicts: name, ra_hours, dec_deg, magnitude, altitude_deg, azimuth_deg.
    Sorted by altitude descending.
    """
    cat_path = Path(catalog_path) if catalog_path else None
    catalog = filter_by_magnitude(load_catalog(cat_path), max_magnitude)

    ts = _timescale()
    t = ts.from_datetime(_parse_timestamp(timestamp))
    location = _earth() + wgs84.latlon(latitude, longitude)

    visible: list[dict[str, Any]] = []

    for entry in catalog:
        star = Star(ra_hours=entry.ra_hours, dec_degrees=entry.dec_deg)
        alt, az, _ = location.at(t).observe(star).apparent().altaz()

        alt_deg = float(alt.degrees)
        az_deg = float(az.degrees)

        if alt_deg <= min_altitude_deg:
            continue

        if azimuth_deg is not None:
            if _azimuth_difference(az_deg, azimuth_deg) > azimuth_tolerance_deg:
                continue

        row = entry.to_dict()
        row["altitude_deg"] = round(alt_deg, 3)
        row["azimuth_deg"] = round(az_deg, 3)
        visible.append(row)

    visible.sort(key=lambda s: -s["altitude_deg"])
    return visible


def _azimuth_difference(a: float, b: float) -> float:
    """Smallest angle between two azimuths in degrees (0–180)."""
    diff = abs(a - b) % 360.0
    return diff if diff <= 180.0 else 360.0 - diff


def visible_stars_summary(
    latitude: float,
    longitude: float,
    timestamp: datetime | str,
    **kwargs: Any,
) -> dict[str, Any]:
    """visible_stars with observer metadata for API responses."""
    stars = visible_stars(latitude, longitude, timestamp, **kwargs)
    return {
        "latitude": latitude,
        "longitude": longitude,
        "timestamp": _parse_timestamp(timestamp).isoformat(),
        "count": len(stars),
        "filters": {
            "max_magnitude": kwargs.get("max_magnitude", 3.5),
            "min_altitude_deg": kwargs.get("min_altitude_deg", 0.0),
            "azimuth_deg": kwargs.get("azimuth_deg"),
            "azimuth_tolerance_deg": kwargs.get("azimuth_tolerance_deg", 45.0),
        },
        "stars": stars,
    }
