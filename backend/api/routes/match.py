"""Phase 4 — geometric pattern matching."""

from __future__ import annotations

import uuid

from fastapi import APIRouter, File, Form, UploadFile
from fastapi.responses import JSONResponse

from backend.api.config import OUTPUT_DIR
from backend.api.helpers import decode_uploaded_image, is_error_response, resolve_observer_location, static_url
from backend.matching.geometric_matcher import match_field
from backend.matching.overlay import draw_alignment_overlay
from backend.vision.detect_stars import detect_stars
from backend.vision.overlay import save_image
from backend.vision.preprocess import preprocess_from_array

router = APIRouter()


@router.post("/match-stars")
async def match_stars_endpoint(
    file: UploadFile = File(...),
    latitude: float | None = Form(None),
    longitude: float | None = Form(None),
    city: str | None = Form(None),
    timestamp: str = Form(...),
    center_azimuth_deg: float | None = Form(None),
    azimuth_tolerance_deg: float = Form(60.0),
    max_magnitude: float = Form(4.0),
    max_points: int = Form(10, ge=3, le=15),
    min_votes: int = Form(3, ge=1),
) -> JSONResponse:
    location = resolve_observer_location(latitude, longitude, city)
    if is_error_response(location):
        return location

    lat, lon = location
    data = await file.read()
    if not data:
        return JSONResponse(status_code=400, content={"error": "Empty file"})

    try:
        bgr = decode_uploaded_image(data)
    except ValueError as exc:
        return JSONResponse(status_code=400, content={"error": str(exc)})

    detections = detect_stars(preprocess_from_array(bgr)["processed"])["stars"]
    if len(detections) < 3:
        return JSONResponse(status_code=400, content={"error": "Need at least 3 detected stars"})

    result = match_field(
        latitude=lat,
        longitude=lon,
        timestamp=timestamp,
        detections=detections,
        max_magnitude=max_magnitude,
        min_altitude_deg=5.0,
        azimuth_deg=center_azimuth_deg,
        azimuth_tolerance_deg=azimuth_tolerance_deg if center_azimuth_deg is not None else 180.0,
        max_points=max_points,
        min_votes=min_votes,
    )

    if "error" in result and not result.get("matches"):
        return JSONResponse(status_code=400, content=result)

    stem = uuid.uuid4().hex[:12]
    label_rows = [
        {"detection_index": m["detection_index"], "catalog_name": m["catalog_name"]}
        for m in result["matches"]
    ]
    overlay = draw_alignment_overlay(
        bgr,
        predicted_catalog=[],
        suggested_labels=label_rows,
        detections=[{**d, "index": i} for i, d in enumerate(detections)],
    )
    overlay_path = OUTPUT_DIR / f"{stem}_matched.jpg"
    save_image(overlay, overlay_path)
    result["matched_image"] = static_url(overlay_path)
    result["location"] = {"latitude": lat, "longitude": lon}
    if city:
        result["city"] = city
    return JSONResponse(content=result)
