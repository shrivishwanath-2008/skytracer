"""
Semi-manual image ↔ sky alignment using one user-identified reference star.

Not plate solving: the user supplies look direction and which detection is a known star.
"""

from __future__ import annotations

import math
from datetime import datetime
from typing import Any

from backend.astronomy.catalog import CatalogStar, load_catalog
from backend.astronomy.projection import angular_separation_deg
from backend.astronomy.visible_stars import visible_stars
from backend.matching.geometry import PixelSkyTransform, fit_transform_from_reference


def find_catalog_star(name: str) -> CatalogStar:
    """Case-insensitive lookup in the bright-star catalog."""
    key = name.strip().lower()
    for star in load_catalog():
        if star.name.lower() == key:
            return star
    raise KeyError(f"Star '{name}' not in the bright-star catalog.")


def _resolve_reference_pixel(
    detections: list[dict[str, Any]],
    *,
    detection_index: int | None,
    pixel_x: float | None,
    pixel_y: float | None,
) -> tuple[float, float]:
    if detection_index is not None:
        if detection_index < 0 or detection_index >= len(detections):
            raise IndexError(f"detection_index {detection_index} out of range (0..{len(detections)-1})")
        d = detections[detection_index]
        return float(d["x"]), float(d["y"])
    if pixel_x is not None and pixel_y is not None:
        return float(pixel_x), float(pixel_y)
    raise ValueError("Provide detection_index or reference_x and reference_y.")


def align_field(
    *,
    latitude: float,
    longitude: float,
    timestamp: datetime | str,
    detections: list[dict[str, Any]],
    image_width: int,
    image_height: int,
    center_azimuth_deg: float,
    center_altitude_deg: float,
    reference_star_name: str,
    detection_index: int | None = None,
    reference_x: float | None = None,
    reference_y: float | None = None,
    fov_deg: float | None = None,
    refine_center: bool = True,
    max_magnitude: float = 4.0,
    suggest_match_radius_deg: float = 2.0,
) -> dict[str, Any]:
    """
    Semi-manual alignment: map image pixels to alt/az using one labeled star.

    Parameters
    ----------
    center_azimuth_deg, center_altitude_deg
        Where the user thinks the image center is pointing.
    reference_star_name
        Catalog name, e.g. "Vega".
    detection_index
        Index into detections[] for that star (from /detect-stars numbering minus 1).
    reference_x, reference_y
        Alternative to detection_index: explicit pixel of the reference star.
    fov_deg
        Optional horizontal field of view; fixes scale, fits rotation only.
    refine_center
        Slightly adjust field center so the reference star matches its pixel exactly.
    """
    catalog_star = find_catalog_star(reference_star_name)
    visible = visible_stars(
        latitude,
        longitude,
        timestamp,
        max_magnitude=max(max_magnitude, catalog_star.magnitude + 0.5),
        min_altitude_deg=-5.0,
    )
    ref_sky = next((s for s in visible if s["name"].lower() == catalog_star.name.lower()), None)
    if ref_sky is None:
        # Star below horizon or filtered — still use catalog coords via Skyfield path
        from backend.astronomy.visible_stars import _parse_timestamp, _timescale, _earth
        from skyfield.api import Star, wgs84

        ts = _timescale()
        t = ts.from_datetime(_parse_timestamp(timestamp))
        loc = _earth() + wgs84.latlon(latitude, longitude)
        sf = Star(ra_hours=catalog_star.ra_hours, dec_degrees=catalog_star.dec_deg)
        alt, az, _ = loc.at(t).observe(sf).apparent().altaz()
        ref_alt, ref_az = float(alt.degrees), float(az.degrees)
        if ref_alt <= 0:
            raise ValueError(
                f"{reference_star_name} is below the horizon at this time/location."
            )
        ref_sky = {
            "name": catalog_star.name,
            "ra_hours": catalog_star.ra_hours,
            "dec_deg": catalog_star.dec_deg,
            "magnitude": catalog_star.magnitude,
            "altitude_deg": ref_alt,
            "azimuth_deg": ref_az,
        }

    ref_px, ref_py = _resolve_reference_pixel(
        detections,
        detection_index=detection_index,
        pixel_x=reference_x,
        pixel_y=reference_y,
    )

    cx, cy = image_width / 2.0, image_height / 2.0
    transform = fit_transform_from_reference(
        center_x=cx,
        center_y=cy,
        center_alt_deg=center_altitude_deg,
        center_az_deg=center_azimuth_deg,
        ref_pixel_x=ref_px,
        ref_pixel_y=ref_py,
        ref_alt_deg=float(ref_sky["altitude_deg"]),
        ref_az_deg=float(ref_sky["azimuth_deg"]),
        fov_deg=fov_deg,
        image_width=float(image_width),
    )

    if refine_center and fov_deg is None:
        transform = _refine_center(transform, ref_px, ref_py, ref_sky)

    # Map each detection to sky coordinates
    mapped_detections: list[dict[str, Any]] = []
    for i, det in enumerate(detections):
        alt, az = transform.pixel_to_altaz(float(det["x"]), float(det["y"]))
        mapped_detections.append({
            **det,
            "index": i,
            "estimated_altitude_deg": round(alt, 3),
            "estimated_azimuth_deg": round(az, 3),
        })

    # Predict pixel positions for visible catalog stars (validation overlay)
    predicted: list[dict[str, Any]] = []
    for star in visible:
        px, py = transform.altaz_to_pixel(
            float(star["altitude_deg"]),
            float(star["azimuth_deg"]),
        )
        if 0 <= px < image_width and 0 <= py < image_height:
            predicted.append({
                "name": star["name"],
                "magnitude": star["magnitude"],
                "altitude_deg": star["altitude_deg"],
                "azimuth_deg": star["azimuth_deg"],
                "pixel_x": round(px, 1),
                "pixel_y": round(py, 1),
            })

    # Suggest names for detections near predicted catalog positions (optional hints)
    suggestions = _suggest_names(mapped_detections, predicted, suggest_match_radius_deg)

    return {
        "mode": "semi-manual (1 reference star)",
        "reference": {
            "name": reference_star_name,
            "pixel_x": ref_px,
            "pixel_y": ref_py,
            "altitude_deg": ref_sky["altitude_deg"],
            "azimuth_deg": ref_sky["azimuth_deg"],
        },
        "field_center": {
            "altitude_deg": round(transform.center_alt_deg, 3),
            "azimuth_deg": round(transform.center_az_deg, 3),
            "pixel_x": transform.center_x,
            "pixel_y": transform.center_y,
        },
        "transform": {
            "scale_pixels_per_degree": round(transform.scale_pixels_per_deg, 2),
            "rotation_deg": round(math.degrees(transform.rotation_rad), 2),
            "estimated_horizontal_fov_deg": round(
                math.degrees(image_width / transform.scale_pixels_per_rad),
                2,
            ),
        },
        "image": {"width": image_width, "height": image_height},
        "detections": mapped_detections,
        "predicted_catalog_stars": predicted,
        "suggested_labels": suggestions,
    }


def _refine_center(
    transform: PixelSkyTransform,
    ref_px: float,
    ref_py: float,
    ref_sky: dict[str, Any],
) -> PixelSkyTransform:
    """Iteratively nudge field center so reference star hits its pixel."""
    alt0 = transform.center_alt_deg
    az0 = transform.center_az_deg
    ref_alt = float(ref_sky["altitude_deg"])
    ref_az = float(ref_sky["azimuth_deg"])

    for _ in range(20):
        px, py = transform.altaz_to_pixel(ref_alt, ref_az)
        err_x = ref_px - px
        err_y = ref_py - py
        if abs(err_x) < 1.0 and abs(err_y) < 1.0:
            break
        deg_per_px = 1.0 / transform.scale_pixels_per_deg
        alt0 += err_y * deg_per_px * 0.3
        az0 = (az0 + err_x * deg_per_px * 0.3) % 360.0
        transform = PixelSkyTransform(
            center_x=transform.center_x,
            center_y=transform.center_y,
            center_alt_deg=alt0,
            center_az_deg=az0,
            scale_pixels_per_rad=transform.scale_pixels_per_rad,
            rotation_rad=transform.rotation_rad,
        )
    return transform


def _suggest_names(
    mapped: list[dict[str, Any]],
    predicted: list[dict[str, Any]],
    radius_deg: float,
) -> list[dict[str, Any]]:
    """Nearest-neighbor in sky angle between detections and catalog predictions."""
    suggestions: list[dict[str, Any]] = []
    used: set[str] = set()

    for det in mapped:
        best_name = None
        best_sep = radius_deg
        for cat in predicted:
            if cat["name"] in used:
                continue
            sep = angular_separation_deg(
                det["estimated_altitude_deg"],
                det["estimated_azimuth_deg"],
                cat["altitude_deg"],
                cat["azimuth_deg"],
            )
            if sep < best_sep:
                best_sep = sep
                best_name = cat["name"]
        if best_name:
            used.add(best_name)
            suggestions.append({
                "detection_index": det["index"],
                "suggested_name": best_name,
                "separation_deg": round(best_sep, 2),
            })
    return suggestions
