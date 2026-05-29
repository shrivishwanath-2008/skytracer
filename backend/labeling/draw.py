"""Render placed labels and anchor markers on the image."""

from __future__ import annotations

import cv2
import numpy as np

from backend.labeling.placement import PlacedLabel


def draw_labeled_image(
    bgr: np.ndarray,
    labels: list[PlacedLabel],
    *,
    draw_markers: bool = True,
) -> np.ndarray:
    out = bgr.copy()
    h, w = out.shape[:2]
    line_th = max(1, int(round(min(h, w) / 1200)))

    for lab in labels:
        ax, ay = int(lab.anchor_x), int(lab.anchor_y)
        if draw_markers and 0 <= ax < w and 0 <= ay < h:
            if lab.kind == "planet":
                cv2.circle(out, (ax, ay), 8, (80, 180, 255), 2)
            elif lab.kind == "constellation":
                cv2.circle(out, (ax, ay), 4, (180, 140, 255), 1)
            else:
                cv2.circle(out, (ax, ay), 5, (0, 200, 0), 1)

        cv2.putText(
            out,
            lab.text,
            (lab.x, lab.y),
            cv2.FONT_HERSHEY_SIMPLEX,
            lab.font_scale,
            lab.color_bgr,
            line_th,
            cv2.LINE_AA,
        )

    return out
