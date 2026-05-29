"""Smoke test for Phase 2 visible-star computation."""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.astronomy.catalog import catalog_stats, load_catalog
from backend.astronomy.visible_stars import visible_stars


def main() -> None:
    stats = catalog_stats()
    print("Catalog:", stats)

    # New York, summer evening
    lat, lon = 40.7128, -74.0060
    t = datetime(2024, 7, 15, 2, 0, 0, tzinfo=timezone.utc)

    stars = visible_stars(lat, lon, t, max_magnitude=3.0, min_altitude_deg=5.0)
    print(f"Visible (mag<=3, alt>5°): {len(stars)}")
    if not stars:
        print("FAIL: expected some bright stars")
        sys.exit(1)

    top = stars[:5]
    for s in top:
        print(
            f"  {s['name']:16s}  alt={s['altitude_deg']:6.1f}°  "
            f"az={s['azimuth_deg']:6.1f}°  mag={s['magnitude']}"
        )

    names = {s["name"] for s in stars}
    if "Sirius" not in names and "Vega" not in names and "Arcturus" not in names:
        print("WARN: none of Sirius/Vega/Arcturus in list (check time/location)")

    print("OK")


if __name__ == "__main__":
    main()
