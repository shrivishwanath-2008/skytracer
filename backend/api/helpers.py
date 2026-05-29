"""Shared HTTP helpers: image decode, location, static URLs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
from fastapi.responses import JSONResponse

from backend.api.config import STATIC_DIR
from backend.astronomy.locations import resolve_city


def decode_uploaded_image(data: bytes) -> np.ndarray:
    """Decode JPEG/PNG bytes to a BGR image array."""
    buffer = np.frombuffer(data, dtype=np.uint8)
    image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Could not decode uploaded image")
    return image


def static_url(path: Path) -> str:
    """URL path under /static for a file in STATIC_DIR."""
    relative = path.relative_to(STATIC_DIR).as_posix()
    return f"/static/{relative}"


def resolve_observer_location(
    latitude: float | None,
    longitude: float | None,
    city: str | None,
) -> tuple[float, float] | JSONResponse:
    """
    Return (lat, lon) or a 400 JSONResponse if location is missing/invalid.
    """
    if latitude is not None and longitude is not None:
        return latitude, longitude

    if not city:
        return JSONResponse(
            status_code=400,
            content={"error": "Provide latitude & longitude, or city"},
        )

    try:
        return resolve_city(city)
    except KeyError as exc:
        return JSONResponse(status_code=400, content={"error": str(exc)})


def is_error_response(value: Any) -> bool:
    return isinstance(value, JSONResponse)
