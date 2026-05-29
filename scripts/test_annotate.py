"""Smoke test for Phase 5 labeling pipeline."""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.labeling.annotate import annotate_image
from scripts.smoke_test import make_synthetic_stars


def main() -> None:
    bgr = make_synthetic_stars()
    # Synthetic has no real sky match — expect few labels, no crash
    result = annotate_image(
        bgr,
        latitude=40.7128,
        longitude=-74.0060,
        timestamp=datetime(2024, 7, 15, 2, 0, 0, tzinfo=timezone.utc),
        center_azimuth_deg=90,
        center_altitude_deg=45,
        show_planets=False,
        show_constellations=False,
    )
    if "error" in result:
        print("detection error:", result["error"])
        sys.exit(1)

    out = ROOT / "backend" / "static" / "output" / "test_labeled.jpg"
    cv2.imwrite(str(out), result["annotated_bgr"])
    print(f"labels={result['label_count']} matches={result['match_count']}")
    print(f"wrote {out}")
    print("OK")


if __name__ == "__main__":
    main()
