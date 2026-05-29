"""Smoke test for Phase 3 semi-manual alignment."""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.astronomy.visible_stars import visible_stars
from backend.matching.geometry import PixelSkyTransform, fit_transform_from_reference
from backend.matching.solver import align_field


def test_roundtrip_transform() -> None:
    """Known transform → synthetic pixel → fit recovers scale/rotation."""
    w, h = 1280, 720
    cx, cy = w / 2, h / 2
    truth = PixelSkyTransform(
        center_x=cx,
        center_y=cy,
        center_alt_deg=55.0,
        center_az_deg=90.0,
        scale_pixels_per_rad=1200.0,
        rotation_rad=0.35,
    )
    ref_alt, ref_az = 62.0, 105.0
    ref_px, ref_py = truth.altaz_to_pixel(ref_alt, ref_az)

    est = fit_transform_from_reference(
        center_x=cx,
        center_y=cy,
        center_alt_deg=truth.center_alt_deg,
        center_az_deg=truth.center_az_deg,
        ref_pixel_x=ref_px,
        ref_pixel_y=ref_py,
        ref_alt_deg=ref_alt,
        ref_az_deg=ref_az,
    )
    assert abs(est.scale_pixels_per_rad - truth.scale_pixels_per_rad) < 1.0
    assert abs(est.rotation_rad - truth.rotation_rad) < 0.02
    print("  transform round-trip OK")


def test_align_vega_nyc() -> None:
    lat, lon = 40.7128, -74.0060
    t = datetime(2024, 7, 15, 2, 0, 0, tzinfo=timezone.utc)
    vis = visible_stars(lat, lon, t, max_magnitude=2.0)
    vega = next(s for s in vis if s["name"] == "Vega")

    w, h = 1280, 720
    # User thinks center is a few degrees from Vega
    center_az = float(vega["azimuth_deg"])
    center_alt = float(vega["altitude_deg"]) - 8.0

    truth = fit_transform_from_reference(
        center_x=w / 2,
        center_y=h / 2,
        center_alt_deg=center_alt,
        center_az_deg=center_az,
        ref_pixel_x=640,
        ref_pixel_y=200,
        ref_alt_deg=float(vega["altitude_deg"]),
        ref_az_deg=float(vega["azimuth_deg"]),
    )
    fake_detections = [{"x": 640, "y": 200, "brightness": 200, "radius": 3}]
    for s in vis[:5]:
        px, py = truth.altaz_to_pixel(float(s["altitude_deg"]), float(s["azimuth_deg"]))
        fake_detections.append({
            "x": int(px),
            "y": int(py),
            "brightness": 180,
            "radius": 3,
        })

    result = align_field(
        latitude=lat,
        longitude=lon,
        timestamp=t,
        detections=fake_detections,
        image_width=w,
        image_height=h,
        center_azimuth_deg=center_az,
        center_altitude_deg=center_alt,
        reference_star_name="Vega",
        detection_index=0,
    )
    fov = result["transform"]["estimated_horizontal_fov_deg"]
    print(f"  Vega align: FOV~{fov} deg, suggestions={len(result['suggested_labels'])}")
    if fov < 5 or fov > 120:
        print("FAIL: unrealistic FOV")
        sys.exit(1)
    print("  semi-manual align OK")


def main() -> None:
    test_roundtrip_transform()
    test_align_vega_nyc()
    print("OK")


if __name__ == "__main__":
    main()
