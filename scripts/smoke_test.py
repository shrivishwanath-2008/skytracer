"""
Smoke test for Phase 1 without a real DSLR image.

Generates synthetic stars on a dark background, runs preprocess → detect → overlay,
and writes debug artifacts to backend/static/debug/.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.vision.detect_stars import detect_stars, draw_blob_preview
from backend.vision.overlay import overlay_and_save, save_image
from backend.vision.preprocess import preprocess_from_array

DEBUG_DIR = ROOT / "backend" / "static" / "debug"
OUTPUT_DIR = ROOT / "backend" / "static" / "output"


def make_synthetic_stars(width: int = 1280, height: int = 720, n_stars: int = 40) -> np.ndarray:
    rng = np.random.default_rng(42)
    image = np.full((height, width, 3), 12, dtype=np.uint8)
    # faint gradient (horizon glow simulation)
    for y in range(height):
        image[y, :, :] = np.clip(12 + int(y * 0.02), 0, 255)

    for _ in range(n_stars):
        x = int(rng.integers(40, width - 40))
        y = int(rng.integers(40, height - 200))
        brightness = int(rng.integers(180, 255))
        cv2.circle(image, (x, y), 2, (brightness, brightness, brightness), -1)
        # slight bloom
        cv2.circle(image, (x, y), 4, (brightness // 3, brightness // 3, brightness // 3), 1)

    return image


def main() -> None:
    DEBUG_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    bgr = make_synthetic_stars()
    stem = "smoke"

    pre = preprocess_from_array(bgr)
    detection = detect_stars(pre["processed"])
    stars = detection["stars"]

    save_image(detection["mask"], DEBUG_DIR / f"{stem}_threshold_mask.png")
    raw = detection["contour_stars"] + detection["blob_stars"]
    save_image(draw_blob_preview(bgr, raw), DEBUG_DIR / f"{stem}_detected_blobs.png")
    overlay_and_save(bgr, stars, DEBUG_DIR / f"{stem}_final_overlay.png")
    overlay_and_save(bgr, stars, OUTPUT_DIR / f"{stem}_annotated.jpg")

    print(f"Synthetic stars placed: 40")
    print(f"Detected: {len(stars)}")
    if stars:
        print("Sample:", stars[0])
    print(f"Debug images: {DEBUG_DIR}")
    if len(stars) < 10:
        print("WARNING: low detection count — tune detect_stars.py")
        sys.exit(1)
    print("OK")


if __name__ == "__main__":
    main()
