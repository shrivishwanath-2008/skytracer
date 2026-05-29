"""
Star candidate detection using OpenCV only.

Combines adaptive thresholding + contour filtering with SimpleBlobDetector,
then merges nearby duplicates. Tuned for small, bright point sources and
to reject large blobs (trees, horizon glow, foreground).
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

StarCandidate = dict[str, float | int]


def _build_blob_detector(
    min_area: float = 4.0,
    max_area: float = 500.0,
    min_brightness: int = 30,
    max_brightness: int = 255,
    min_circularity: float = 0.5,
    min_inertia: float = 0.3,
) -> cv2.SimpleBlobDetector:
    params = cv2.SimpleBlobDetector_Params()
    params.filterByArea = True
    params.minArea = min_area
    params.maxArea = max_area
    params.filterByCircularity = True
    params.minCircularity = min_circularity
    params.filterByInertia = True
    params.minInertiaRatio = min_inertia
    params.filterByConvexity = False
    params.filterByColor = False
    # Threshold sweep params (used when the OpenCV build supports them)
    if hasattr(params, "filterByThreshold"):
        params.filterByThreshold = True
    if hasattr(params, "minThreshold"):
        params.minThreshold = min_brightness
    if hasattr(params, "maxThreshold"):
        params.maxThreshold = max_brightness
    if hasattr(params, "thresholdStep"):
        params.thresholdStep = 10
    return cv2.SimpleBlobDetector_create(params)


def threshold_mask(
    processed_gray: np.ndarray,
    block_size: int = 51,
    c: int = -10,
) -> np.ndarray:
    """
    Adaptive threshold → binary mask of bright regions.
    Negative C favors brighter-than-local pixels (stars).
    """
    bs = block_size if block_size % 2 == 1 else block_size + 1
    return cv2.adaptiveThreshold(
        processed_gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        bs,
        c,
    )


def detect_from_contours(
    processed_gray: np.ndarray,
    mask: np.ndarray,
    *,
    min_area: float = 4.0,
    max_area: float = 400.0,
    min_circularity: float = 0.55,
    max_aspect_ratio: float = 2.5,
) -> list[StarCandidate]:
    """Find star-like blobs via contours on the threshold mask."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    stars: list[StarCandidate] = []

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area or area > max_area:
            continue

        perimeter = cv2.arcLength(contour, True)
        if perimeter <= 0:
            continue
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        if circularity < min_circularity:
            continue

        x, y, w, h = cv2.boundingRect(contour)
        aspect = max(w, h) / max(min(w, h), 1)
        if aspect > max_aspect_ratio:
            continue

        cx = int(x + w / 2)
        cy = int(y + h / 2)
        radius = float(np.sqrt(area / np.pi))
        brightness = float(processed_gray[cy, cx]) if 0 <= cy < processed_gray.shape[0] and 0 <= cx < processed_gray.shape[1] else 0.0

        stars.append({
            "x": cx,
            "y": cy,
            "brightness": brightness,
            "radius": radius,
        })

    return stars


def detect_from_blobs(
    processed_gray: np.ndarray,
    detector: cv2.SimpleBlobDetector | None = None,
) -> list[StarCandidate]:
    """Detect stars with SimpleBlobDetector."""
    if detector is None:
        detector = _build_blob_detector()
    keypoints = detector.detect(processed_gray)
    stars: list[StarCandidate] = []
    for kp in keypoints:
        x, y = int(kp.pt[0]), int(kp.pt[1])
        brightness = float(processed_gray[y, x]) if 0 <= y < processed_gray.shape[0] and 0 <= x < processed_gray.shape[1] else 0.0
        stars.append({
            "x": x,
            "y": y,
            "brightness": brightness,
            "radius": float(kp.size / 2),
        })
    return stars


def _merge_nearby(
    stars: list[StarCandidate],
    min_distance: float = 8.0,
) -> list[StarCandidate]:
    """Merge detections within min_distance pixels; keep brighter one."""
    if not stars:
        return []
    sorted_stars = sorted(stars, key=lambda s: -float(s["brightness"]))
    kept: list[StarCandidate] = []

    for star in sorted_stars:
        sx, sy = int(star["x"]), int(star["y"])
        duplicate = False
        for other in kept:
            ox, oy = int(other["x"]), int(other["y"])
            dist = np.hypot(sx - ox, sy - oy)
            if dist < min_distance:
                duplicate = True
                break
        if not duplicate:
            kept.append(star)

    return kept


def detect_stars(
    processed_gray: np.ndarray,
    *,
    use_contours: bool = True,
    use_blobs: bool = True,
    merge_distance: float = 8.0,
    block_size: int = 51,
    threshold_c: int = -10,
    min_area: float = 4.0,
    max_area: float = 500.0,
) -> dict[str, Any]:
    """
    Detect star candidates in a preprocessed grayscale image.

    Returns:
      - stars: merged list of candidates
      - mask: threshold debug mask
      - contour_stars / blob_stars: raw detections before merge
    """
    mask = threshold_mask(processed_gray, block_size=block_size, c=threshold_c)

    contour_stars: list[StarCandidate] = []
    blob_stars: list[StarCandidate] = []

    if use_contours:
        contour_stars = detect_from_contours(
            processed_gray,
            mask,
            min_area=min_area,
            max_area=max_area * 0.8,
        )

    if use_blobs:
        detector = _build_blob_detector(min_area=min_area, max_area=max_area)
        blob_stars = detect_from_blobs(processed_gray, detector)

    combined = contour_stars + blob_stars
    stars = _merge_nearby(combined, min_distance=merge_distance)

    return {
        "stars": stars,
        "mask": mask,
        "contour_stars": contour_stars,
        "blob_stars": blob_stars,
    }


def draw_blob_preview(
    bgr: np.ndarray,
    stars: list[StarCandidate],
    color: tuple[int, int, int] = (0, 255, 255),
) -> np.ndarray:
    """Debug image: small dots for raw blob/contour detections."""
    preview = bgr.copy()
    for star in stars:
        x, y = int(star["x"]), int(star["y"])
        cv2.circle(preview, (x, y), 2, color, -1)
    return preview
