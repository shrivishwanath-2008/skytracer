"""Phase 5–6 — full annotate pipeline (primary user-facing endpoint)."""

from __future__ import annotations

import uuid

from fastapi import APIRouter, File, Form, UploadFile
from fastapi.responses import JSONResponse

from backend.api.config import OUTPUT_DIR
from backend.api.helpers import decode_uploaded_image, is_error_response, resolve_observer_location, static_url
from backend.labeling.annotate import annotate_image
from backend.vision.overlay import save_image

router = APIRouter()


@router.post("/annotate")
async def annotate_endpoint(
    file: UploadFile = File(...),
    latitude: float | None = Form(None),
    longitude: float | None = Form(None),
    city: str | None = Form(None),
    timestamp: str = Form(...),
    center_azimuth_deg: float = Form(..., ge=0, lt=360),
    center_altitude_deg: float = Form(..., ge=0, le=90),
    reference_star_name: str | None = Form(None),
    detection_index: int | None = Form(None),
    show_planets: bool = Form(True),
    show_constellations: bool = Form(True),
    max_magnitude: float = Form(4.0),
    azimuth_tolerance_deg: float = Form(60.0),
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

    result = annotate_image(
        bgr,
        latitude=lat,
        longitude=lon,
        timestamp=timestamp,
        center_azimuth_deg=center_azimuth_deg,
        center_altitude_deg=center_altitude_deg,
        reference_star_name=reference_star_name,
        detection_index=detection_index,
        show_planets=show_planets,
        show_constellations=show_constellations,
        max_magnitude=max_magnitude,
        azimuth_tolerance_deg=azimuth_tolerance_deg,
    )

    if "error" in result and not result.get("labels"):
        return JSONResponse(status_code=400, content={"error": result["error"]})

    stem = uuid.uuid4().hex[:12]
    overlay_path = OUTPUT_DIR / f"{stem}_labeled.jpg"
    save_image(result.pop("annotated_bgr"), overlay_path)
    result["annotated_image"] = static_url(overlay_path)
    result["location"] = {"latitude": lat, "longitude": lon}
    if city:
        result["city"] = city
    return JSONResponse(content=result)
