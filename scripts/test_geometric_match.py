"""Smoke test for Phase 4 triangle / distance / angle matching."""

from __future__ import annotations

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.astronomy.projection import tangent_to_altaz
from backend.matching.geometric_matcher import geometric_match


def _make_square_catalog(side_deg: float = 8.0) -> list[dict]:
    """Four stars forming a square on the tangent plane (~8 deg sides)."""
    center_alt, center_az = 55.0, 200.0
    side_rad = math.radians(side_deg)
    stars = []
    for name, (xi, eta) in [
        ("Star-A", (0.0, 0.0)),
        ("Star-B", (side_rad, 0.0)),
        ("Star-C", (side_rad, side_rad)),
        ("Star-D", (0.0, side_rad)),
    ]:
        alt, az = tangent_to_altaz(xi, eta, center_alt, center_az)
        stars.append({
            "name": name,
            "altitude_deg": alt,
            "azimuth_deg": az,
            "magnitude": 2.0,
        })
    return stars


def _make_square_detections(scale: float = 40.0) -> list[dict]:
    """Same shape in pixels (scaled)."""
    cx, cy = 640, 360
    out = []
    for i, (dx, dy) in enumerate([(0, 0), (scale, 0), (scale, scale), (0, scale)]):
        out.append({
            "x": int(cx + dx),
            "y": int(cy + dy),
            "brightness": 200 - i,
            "radius": 3,
        })
    return out


def test_scaled_square() -> None:
    result = geometric_match(
        _make_square_detections(scale=50),
        _make_square_catalog(side_deg=8.0),
        max_points=4,
        min_votes=2,
        ratio_tol=0.15,
        angle_tol_deg=12.0,
    )
    matches = result["matches"]
    print(f"  square pattern: {len(matches)} matches")
    for m in matches:
        print(f"    det {m['detection_index']} -> {m['catalog_name']} (votes={m['votes']})")
    if len(matches) < 3:
        print("FAIL: expected at least 3 matches on identical shape")
        sys.exit(1)


def test_real_visible_subset() -> None:
    from datetime import datetime, timezone

    from backend.astronomy.visible_stars import visible_stars
    from backend.matching.geometry import fit_transform_from_reference

    lat, lon = 40.7128, -74.0060
    t = datetime(2024, 7, 15, 2, 0, 0, tzinfo=timezone.utc)
    vis = visible_stars(lat, lon, t, max_magnitude=2.5)[:8]
    vega = next(s for s in vis if s["name"] == "Vega")

    w, h = 1280, 720
    center_az = float(vega["azimuth_deg"])
    center_alt = float(vega["altitude_deg"]) - 6
    truth = fit_transform_from_reference(
        center_x=w / 2,
        center_y=h / 2,
        center_alt_deg=center_alt,
        center_az_deg=center_az,
        ref_pixel_x=700,
        ref_pixel_y=250,
        ref_alt_deg=float(vega["altitude_deg"]),
        ref_az_deg=float(vega["azimuth_deg"]),
    )
    dets = []
    for i, s in enumerate(vis):
        px, py = truth.altaz_to_pixel(float(s["altitude_deg"]), float(s["azimuth_deg"]))
        dets.append({"x": int(px), "y": int(py), "brightness": 220 - i, "radius": 3})

    result = geometric_match(dets, vis, max_points=8, min_votes=2)
    names = {m["catalog_name"] for m in result["matches"]}
    print(f"  NYC summer subset: {len(result['matches'])} matches, Vega={'Vega' in names}")
    if "Vega" not in names and len(result["matches"]) < 2:
        print("WARN: weak match on synthetic NYC field")


def main() -> None:
    test_scaled_square()
    test_real_visible_subset()
    print("OK")


if __name__ == "__main__":
    main()
