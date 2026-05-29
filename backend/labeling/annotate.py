"""
Phase 5: full annotation pipeline — detect, match/align, label, export image.
"""

from __future__ import annotations

from typing import Any

from backend.astronomy.constellations import constellation_label_positions
from backend.astronomy.planets import visible_planets
from backend.astronomy.visible_stars import visible_stars
from backend.labeling.draw import draw_labeled_image
from backend.labeling.placement import place_labels, specs_from_annotations
from backend.matching.geometry import PixelSkyTransform, fit_transform_from_reference
from backend.matching.geometric_matcher import match_field
from backend.matching.solver import align_field, find_catalog_star
from backend.vision.detect_stars import detect_stars
from backend.vision.preprocess import preprocess_from_array


def _label_scale(image_width: int, image_height: int) -> float:
    """Scale fonts for resolution (reference ~1280px wide)."""
    return max(0.6, min(1.4, image_width / 1280.0))


def _sky_to_pixel_items(
    transform: PixelSkyTransform,
    items: list[dict[str, Any]],
    image_width: int,
    image_height: int,
) -> list[dict[str, Any]]:
    projected = []
    for item in items:
        px, py = transform.altaz_to_pixel(
            float(item["altitude_deg"]),
            float(item["azimuth_deg"]),
        )
        if 0 <= px < image_width and 0 <= py < image_height:
            projected.append({**item, "pixel_x": px, "pixel_y": py})
    return projected


def _transform_from_reference(
    *,
    detections: list[dict[str, Any]],
    detection_index: int,
    reference_star_name: str,
    center_azimuth_deg: float,
    center_altitude_deg: float,
    image_width: int,
    image_height: int,
    ref_sky: dict[str, Any],
) -> PixelSkyTransform:
    det = detections[detection_index]
    return fit_transform_from_reference(
        center_x=image_width / 2.0,
        center_y=image_height / 2.0,
        center_alt_deg=center_altitude_deg,
        center_az_deg=center_azimuth_deg,
        ref_pixel_x=float(det["x"]),
        ref_pixel_y=float(det["y"]),
        ref_alt_deg=float(ref_sky["altitude_deg"]),
        ref_az_deg=float(ref_sky["azimuth_deg"]),
    )

from numpy import ndarray
import numpy as np
def annotate_image(
    bgr: np.ndarray,
    *,
    latitude: float,
    longitude: float,
    timestamp: str,
    center_azimuth_deg: float,
    center_altitude_deg: float,
    reference_star_name: str | None = None,
    detection_index: int | None = None,
    show_planets: bool = True,
    show_constellations: bool = True,
    max_magnitude: float = 4.0,
    azimuth_tolerance_deg: float = 60.0,
    min_match_votes: int = 3,
) -> dict[str, Any]:
    """
    End-to-end labeling: detect → match → (optional) reference align → overlay labels.
    """
    h, w = bgr.shape[:2]
    scale = _label_scale(w, h)

    pre = preprocess_from_array(bgr)
    detections = detect_stars(pre["processed"])["stars"]
    if not detections:
        return {"error": "No stars detected", "detections": []}

    match_result = match_field(
        latitude=latitude,
        longitude=longitude,
        timestamp=timestamp,
        detections=detections,
        max_magnitude=max_magnitude,
        min_altitude_deg=5.0,
        azimuth_deg=center_azimuth_deg,
        azimuth_tolerance_deg=azimuth_tolerance_deg,
        min_votes=min_match_votes,
    )

    catalog = visible_stars(
        latitude,
        longitude,
        timestamp,
        max_magnitude=max_magnitude,
        min_altitude_deg=5.0,
        azimuth_deg=center_azimuth_deg,
        azimuth_tolerance_deg=azimuth_tolerance_deg,
    )

    transform: PixelSkyTransform | None = None
    align_meta: dict[str, Any] | None = None

    # Prefer explicit user reference; else best geometric match as reference
    ref_name = reference_star_name
    ref_index = detection_index
    if ref_name is None and match_result.get("matches"):
        best = max(match_result["matches"], key=lambda m: m["votes"])
        ref_name = best["catalog_name"]
        ref_index = best["detection_index"]

    if ref_name is not None and ref_index is not None:
        try:
            cat = find_catalog_star(ref_name)
            ref_sky = next((s for s in catalog if s["name"].lower() == cat.name.lower()), None)
            if ref_sky is None:
                from backend.astronomy.visible_stars import _parse_timestamp, _earth, _timescale
                
                from skyfield.api import Star, wgs84

                ts = _timescale()
                t = ts.from_datetime(_parse_timestamp(timestamp))
                loc = _earth() + wgs84.latlon(latitude, longitude)
                sf = Star(ra_hours=cat.ra_hours, dec_degrees=cat.dec_deg)
                alt, az, _ = loc.at(t).observe(sf).apparent().altaz()
                ref_sky = {
                    "name": cat.name,
                    "altitude_deg": float(alt.degrees),
                    "azimuth_deg": float(az.degrees),
                }
            transform = _transform_from_reference(
                detections=detections,
                detection_index=ref_index,
                reference_star_name=ref_name,
                center_azimuth_deg=center_azimuth_deg,
                center_altitude_deg=center_altitude_deg,
                image_width=w,
                image_height=h,
                ref_sky=ref_sky,
            )
            align_meta = align_field(
                latitude=latitude,
                longitude=longitude,
                timestamp=timestamp,
                detections=detections,
                image_width=w,
                image_height=h,
                center_azimuth_deg=center_azimuth_deg,
                center_altitude_deg=center_altitude_deg,
                reference_star_name=ref_name,
                detection_index=ref_index,
                max_magnitude=max_magnitude,
            )
        except (KeyError, ValueError, IndexError) as exc:
            align_meta = {"error": str(exc)}

    label_items: list[dict[str, Any]] = []

    # Matched stars at detection pixels
    name_by_index = {
        m["detection_index"]: m["catalog_name"] for m in match_result.get("matches", [])
    }
    for i, det in enumerate(detections):
        name = name_by_index.get(i)
        if name:
            label_items.append({
                "name": name,
                "type": "star",
                "pixel_x": float(det["x"]),
                "pixel_y": float(det["y"]),
            })

    if transform is not None:
        if show_planets:
            planets = visible_planets(latitude, longitude, timestamp)
            label_items.extend(
                _sky_to_pixel_items(transform, planets, w, h),
            )

        if show_constellations:
            constellations = constellation_label_positions(catalog)
            label_items.extend(
                _sky_to_pixel_items(transform, constellations, w, h),
            )

        # Extra bright catalog stars not detected (faint cyan) — skip to reduce clutter
    warnings: list[str] = list(match_result.get("warnings") or [])
    if transform is None and (show_planets or show_constellations):
        warnings.append(
            "No sky alignment; only matched stars labeled. "
            "Provide reference_star_name + detection_index for planets/constellations."
        )

    specs = specs_from_annotations(label_items, image_scale=scale)
    placed = place_labels(specs, w, h)
    annotated_bgr = draw_labeled_image(bgr, placed)

    return {
        "star_count": len(detections),
        "match_count": len(match_result.get("matches", [])),
        "label_count": len(placed),
        "detections": detections,
        "matches": match_result.get("matches", []),
        "labels": [
            {
                "text": p.text,
                "x": p.x,
                "y": p.y,
                "kind": p.kind,
                "anchor_x": p.anchor_x,
                "anchor_y": p.anchor_y,
            }
            for p in placed
        ],
        "align": align_meta,
        "match_debug": match_result.get("debug"),
        "warnings": warnings,
        "annotated_bgr": annotated_bgr,
    }
