"""
Image preprocessing for night-sky star detection.

Pipeline: load → grayscale → blur → denoise → optional contrast boost.
Each step is a standalone function for independent testing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np


def load_image(path: str | Path) -> np.ndarray:
    """Load a BGR image from disk. Raises FileNotFoundError if missing."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Image not found: {path}")
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not decode image: {path}")
    return image


def to_grayscale(bgr: np.ndarray) -> np.ndarray:
    """Convert BGR image to single-channel grayscale."""
    if len(bgr.shape) == 2:
        return bgr.copy()
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)


def apply_gaussian_blur(
    gray: np.ndarray,
    kernel_size: int = 5,
    sigma: float = 1.0,
) -> np.ndarray:
    """Gaussian blur to suppress high-frequency sensor noise."""
    k = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
    return cv2.GaussianBlur(gray, (k, k), sigmaX=sigma, sigmaY=sigma)


def reduce_noise(
    gray: np.ndarray,
    method: str = "median",
    kernel_size: int = 3,
) -> np.ndarray:
    """
    Reduce salt-and-pepper / hot-pixel noise.

    method: 'median' | 'bilateral'
    """
    k = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
    if method == "bilateral":
        return cv2.bilateralFilter(gray, d=5, sigmaColor=50, sigmaSpace=50)
    return cv2.medianBlur(gray, k)


def enhance_contrast(
    gray: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: tuple[int, int] = (8, 8),
) -> np.ndarray:
    """CLAHE contrast enhancement — helps faint stars in light-polluted skies."""
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(gray)


def preprocess_pipeline(
    path: str | Path,
    *,
    blur_kernel: int = 5,
    blur_sigma: float = 1.0,
    denoise_method: str = "median",
    denoise_kernel: int = 3,
    apply_contrast: bool = True,
    clahe_clip: float = 2.0,
) -> dict[str, Any]:
    """
    Run the full preprocessing pipeline.

    Returns a dict with intermediate stages for debugging:
      - original (BGR)
      - gray
      - blurred
      - denoised
      - processed (final grayscale used for detection)
    """
    original = load_image(path)
    gray = to_grayscale(original)
    blurred = apply_gaussian_blur(gray, kernel_size=blur_kernel, sigma=blur_sigma)
    denoised = reduce_noise(blurred, method=denoise_method, kernel_size=denoise_kernel)
    processed = enhance_contrast(denoised, clip_limit=clahe_clip) if apply_contrast else denoised

    return {
        "original": original,
        "gray": gray,
        "blurred": blurred,
        "denoised": denoised,
        "processed": processed,
    }


def preprocess_from_array(
    bgr: np.ndarray,
    **kwargs: Any,
) -> dict[str, Any]:
    """Same pipeline as preprocess_pipeline but starting from an in-memory BGR array."""
    gray = to_grayscale(bgr)
    blur_kernel = kwargs.get("blur_kernel", 5)
    blur_sigma = kwargs.get("blur_sigma", 1.0)
    denoise_method = kwargs.get("denoise_method", "median")
    denoise_kernel = kwargs.get("denoise_kernel", 3)
    apply_contrast = kwargs.get("apply_contrast", True)
    clahe_clip = kwargs.get("clahe_clip", 2.0)

    blurred = apply_gaussian_blur(gray, kernel_size=blur_kernel, sigma=blur_sigma)
    denoised = reduce_noise(blurred, method=denoise_method, kernel_size=denoise_kernel)
    processed = enhance_contrast(denoised, clip_limit=clahe_clip) if apply_contrast else denoised

    return {
        "original": bgr,
        "gray": gray,
        "blurred": blurred,
        "denoised": denoised,
        "processed": processed,
    }
