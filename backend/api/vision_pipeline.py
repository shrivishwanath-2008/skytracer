"""Phase 1 detection pipeline used by several HTTP endpoints."""

from __future__ import annotations

from typing import Any

import numpy as np

from backend.api.config import DEBUG_DIR, OUTPUT_DIR
from backend.api.helpers import static_url
from backend.vision.detect_stars import detect_stars, draw_blob_preview
from backend.vision.overlay import overlay_and_save, save_image
from backend.vision.preprocess import preprocess_from_array


def run_detection_with_debug(bgr: np.ndarray, stem: str) -> dict[str, Any]:
    """
    Detect stars and write debug images (mask, blobs, overlay).

    Returns URLs and star list for the API response.
    """
    pre = preprocess_from_array(bgr)
    processed = pre["processed"]
    original = pre["original"]

    detection = detect_stars(processed)
    stars = detection["stars"]

    mask_path = DEBUG_DIR / f"{stem}_threshold_mask.png"
    save_image(detection["mask"], mask_path)

    raw_stars = detection["contour_stars"] + detection["blob_stars"]
    blobs_path = DEBUG_DIR / f"{stem}_detected_blobs.png"
    save_image(draw_blob_preview(original, raw_stars), blobs_path)

    overlay_path = OUTPUT_DIR / f"{stem}_annotated.jpg"
    overlay_and_save(original, stars, overlay_path, show_indices=True)

    final_debug_path = DEBUG_DIR / f"{stem}_final_overlay.png"
    overlay_and_save(original, stars, final_debug_path, show_indices=True)

    return {
        "annotated_image": static_url(overlay_path),
        "star_count": len(stars),
        "stars": stars,
        "debug": {
            "threshold_mask": static_url(mask_path),
            "detected_blobs": static_url(blobs_path),
            "final_overlay": static_url(final_debug_path),
        },
    }
