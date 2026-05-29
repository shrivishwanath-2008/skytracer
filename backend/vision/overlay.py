"""
Draw detection overlays and save annotated images.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np


def draw_stars(
    bgr: np.ndarray,
    stars: list[dict[str, Any]],
    *,
    circle_color: tuple[int, int, int] = (0, 255, 0),
    circle_thickness: int = 1,
    show_indices: bool = True,
    index_color: tuple[int, int, int] = (255, 255, 0),
    radius_scale: float = 1.5,
    min_radius: int = 4,
    max_radius: int = 20,
) -> np.ndarray:
    """
    Draw circles around each star; optionally label with 1-based index.
    """
    annotated = bgr.copy()
    for i, star in enumerate(stars):
        x, y = int(star["x"]), int(star["y"])
        r = star.get("radius", min_radius)
        draw_r = int(np.clip(float(r) * radius_scale, min_radius, max_radius))

        cv2.circle(annotated, (x, y), draw_r, circle_color, circle_thickness)
        if show_indices:
            label = str(i + 1)
            cv2.putText(
                annotated,
                label,
                (x + draw_r + 2, y - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                index_color,
                1,
                cv2.LINE_AA,
            )
    return annotated


def save_image(image: np.ndarray, path: str | Path) -> Path:
    """Save BGR or grayscale image; creates parent directories."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(path), image)
    if not ok:
        raise IOError(f"Failed to write image: {path}")
    return path


def overlay_and_save(
    bgr: np.ndarray,
    stars: list[dict[str, Any]],
    output_path: str | Path,
    **draw_kwargs: Any,
) -> Path:
    """Draw overlay and save to disk."""
    annotated = draw_stars(bgr, stars, **draw_kwargs)
    return save_image(annotated, output_path)
