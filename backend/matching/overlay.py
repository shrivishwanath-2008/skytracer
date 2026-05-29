"""Draw semi-manual alignment overlays on detected images."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np


def draw_alignment_overlay(
    bgr: np.ndarray,
    *,
    predicted_catalog: list[dict[str, Any]],
    suggested_labels: list[dict[str, Any]],
    detections: list[dict[str, Any]],
    reference_name: str | None = None,
    reference_pixel: tuple[float, float] | None = None,
) -> np.ndarray:
    """
    Cyan circles = predicted catalog positions; yellow text = suggested names;
    green ring = user reference star.
    """
    out = bgr.copy()

    for cat in predicted_catalog:
        x, y = int(cat["pixel_x"]), int(cat["pixel_y"])
        cv2.circle(out, (x, y), 10, (255, 200, 0), 1)
        cv2.putText(
            out,
            cat["name"],
            (x + 12, y + 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 200, 0),
            1,
            cv2.LINE_AA,
        )

    label_by_index = {s["detection_index"]: s.get("suggested_name") or s.get("catalog_name") for s in suggested_labels}
    for det in detections:
        idx = det.get("index", 0)
        x, y = int(det["x"]), int(det["y"])
        name = label_by_index.get(idx)
        if name:
            cv2.putText(
                out,
                name,
                (x + 8, y - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )

    if reference_pixel and reference_name:
        rx, ry = int(reference_pixel[0]), int(reference_pixel[1])
        cv2.circle(out, (rx, ry), 14, (0, 255, 0), 2)
        cv2.putText(
            out,
            f"ref: {reference_name}",
            (rx + 16, ry),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )

    return out
