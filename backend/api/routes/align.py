"""Phase 3 — semi-manual sky alignment."""

from __future__ import annotations

import uuid

from fastapi import APIRouter, File, Form, UploadFile
from fastapi.responses import JSONResponse

from backend.api.config import OUTPUT_DIR
from backend.api.helpers import decode_uploaded_image, is_error_response, resolve_observer_location, static_url
from backend.matching.overlay import draw_alignment_overlay
from backend.matching.solver import align_field
from backend.vision.detect_stars import detect_stars
from backend.vision.overlay import save_image
from backend.vision.preprocess import preprocess_from_array

router = APIRouter()


@router.post("/align-sky")
async def align_sky_endpoint(
    file: UploadFile = File(...),
    latitude: float | None = Form(None),
    longitude: float | None = Form(None),
    city: str | None = Form(None),
    timestamp: str = Form(...),
    center_azimuth_deg: float = Form(..., ge=0, lt=360),
    center_altitude_deg: float = Form(..., ge=0, le=90),
    reference_star_name: str = Form(...),
    detection_index: int | None = Form(None),
    reference_x: float | None = Form(None),
    reference_y: float | None = Form(None),
    fov_deg: float | None = Form(None),
    max_magnitude: float = Form(4.0),
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

    height, width = bgr.shape[:2]
    detections = detect_stars(preprocess_from_array(bgr)["processed"])["stars"]
    if not detections:
        return JSONResponse(status_code=400, content={"error": "No stars detected in image"})

    try:
        result = align_field(
            latitude=lat,
            longitude=lon,
            timestamp=timestamp,
            detections=detections,
            image_width=width,
            image_height=height,
            center_azimuth_deg=center_azimuth_deg,
            center_altitude_deg=center_altitude_deg,
            reference_star_name=reference_star_name,
            detection_index=detection_index,
            reference_x=reference_x,
            reference_y=reference_y,
            fov_deg=fov_deg,
            max_magnitude=max_magnitude,
        )
    except (KeyError, ValueError, IndexError) as exc:
        return JSONResponse(status_code=400, content={"error": str(exc)})

    stem = uuid.uuid4().hex[:12]
    overlay = draw_alignment_overlay(
        bgr,
        predicted_catalog=result["predicted_catalog_stars"],
        suggested_labels=result["suggested_labels"],
        detections=result["detections"],
        reference_name=reference_star_name,
        reference_pixel=(result["reference"]["pixel_x"], result["reference"]["pixel_y"]),
    )
    overlay_path = OUTPUT_DIR / f"{stem}_aligned.jpg"
    save_image(overlay, overlay_path)
    result["aligned_image"] = static_url(overlay_path)
    result["location"] = {"latitude": lat, "longitude": lon}
    if city:
        result["city"] = city
    return JSONResponse(content=result)
