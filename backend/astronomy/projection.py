"""
Horizon (alt/az) ↔ tangent plane ↔ pixel helpers for wide-field DSLR alignment.

Uses a gnomonic (tangent-plane) projection centered on the approximate field center.
Good enough for kit-lens fields (~20–50°); not a full astrometric solution.
"""

from __future__ import annotations

import math
from typing import Any


def altaz_to_unit_vector(altitude_deg: float, azimuth_deg: float) -> tuple[float, float, float]:
    """
    Horizon unit vector in the local ENU frame.

    x = east, y = north, z = up. Azimuth measured eastward from north.
    """
    alt = math.radians(altitude_deg)
    az = math.radians(azimuth_deg)
    cos_alt = math.cos(alt)
    return (
        cos_alt * math.sin(az),
        cos_alt * math.cos(az),
        math.sin(alt),
    )


def unit_vector_to_altaz(x: float, y: float, z: float) -> tuple[float, float]:
    """ENU unit vector → (altitude_deg, azimuth_deg)."""
    alt = math.degrees(math.asin(max(-1.0, min(1.0, z))))
    az = math.degrees(math.atan2(x, y)) % 360.0
    return alt, az


def angular_separation_deg(
    alt1: float,
    az1: float,
    alt2: float,
    az2: float,
) -> float:
    """
    Great-circle distance between two horizon coordinates (degrees).

    Uses the spherical law of cosines on the celestial sphere. This is the
    true on-sky angle, independent of camera scale — used to compare star
  patterns in the catalog and for geometric matching.
    """
    a1, a2 = math.radians(alt1), math.radians(alt2)
    daz = math.radians(az2 - az1)
    # cos(separation) = sin a1 sin a2 + cos a1 cos a2 cos(delta_az)
    cos_d = math.sin(a1) * math.sin(a2) + math.cos(a1) * math.cos(a2) * math.cos(daz)
    cos_d = max(-1.0, min(1.0, cos_d))
    return math.degrees(math.acos(cos_d))


def altaz_to_tangent(
    altitude_deg: float,
    azimuth_deg: float,
    center_alt_deg: float,
    center_az_deg: float,
) -> tuple[float, float]:
    """
    Gnomonic projection to tangent plane at the field center.

    Returns (xi, eta) in radians: xi = east, eta = north.
    """
    alt0 = math.radians(center_alt_deg)
    az0 = math.radians(center_az_deg)
    alt = math.radians(altitude_deg)
    az = math.radians(azimuth_deg)

    daz = az - az0
    cos_daz = math.cos(daz)
    sin_daz = math.sin(daz)

    sin_theta = math.sqrt(
        (math.cos(alt) * sin_daz) ** 2
        + (math.cos(alt0) * math.sin(alt) - math.sin(alt0) * math.cos(alt) * cos_daz) ** 2
    )
    cos_theta = math.sin(alt0) * math.sin(alt) + math.cos(alt0) * math.cos(alt) * cos_daz
    theta = math.atan2(sin_theta, cos_theta)

    if theta < 1e-12:
        return 0.0, 0.0

    # Position angle PA: direction from field center toward the star, measured
    # from north through east (standard astronomical convention).
    sin_pa = math.cos(alt) * sin_daz / sin_theta
    cos_pa = (math.sin(alt) - math.sin(alt0) * cos_theta) / (math.cos(alt0) * sin_theta)
    pa = math.atan2(sin_pa, cos_pa)

    # Gnomonic projection: project along PA by tan(angular distance).
    # Straight lines in the tangent plane map to great circles on the sky.
    r = math.tan(theta)
    xi = r * math.sin(pa)
    eta = r * math.cos(pa)
    return xi, eta


def tangent_to_altaz(
    xi: float,
    eta: float,
    center_alt_deg: float,
    center_az_deg: float,
) -> tuple[float, float]:
    """Inverse gnomonic: tangent-plane radians → (altitude_deg, azimuth_deg)."""
    alt0 = math.radians(center_alt_deg)
    az0 = math.radians(center_az_deg)

    r = math.sqrt(xi * xi + eta * eta)
    if r < 1e-12:
        return center_alt_deg, center_az_deg

    theta = math.atan(r)
    pa = math.atan2(xi, eta)
    sin_theta = math.sin(theta)
    cos_theta = math.cos(theta)

    sin_alt = math.sin(alt0) * cos_theta + math.cos(alt0) * sin_theta * math.cos(pa)
    sin_alt = max(-1.0, min(1.0, sin_alt))
    alt = math.asin(sin_alt)

    sin_daz = sin_theta * math.sin(pa) / max(math.cos(alt), 1e-12)
    cos_daz = (sin_alt - math.sin(alt0) * cos_theta) / (math.cos(alt0) * max(sin_theta, 1e-12))
    az = az0 + math.atan2(sin_daz, cos_daz)

    return math.degrees(alt), math.degrees(az) % 360.0


def star_to_horizontal_dict(star: dict[str, Any]) -> dict[str, Any]:
    """Attach unit_vector to a visible-star dict."""
    out = dict(star)
    out["unit_vector"] = altaz_to_unit_vector(
        float(star["altitude_deg"]),
        float(star["azimuth_deg"]),
    )
    return out
