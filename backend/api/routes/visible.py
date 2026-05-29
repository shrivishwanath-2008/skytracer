"""Phase 2 — visible bright stars at observer time/place."""

from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from backend.api.helpers import is_error_response, resolve_observer_location
from backend.api.schemas import VisibleStarsRequest
from backend.astronomy.visible_stars import visible_stars_summary

router = APIRouter()


@router.post("/visible-stars")
def visible_stars_endpoint(body: VisibleStarsRequest) -> JSONResponse:
    location = resolve_observer_location(body.latitude, body.longitude, body.city)
    if is_error_response(location):
        return location

    lat, lon = location
    result = visible_stars_summary(
        lat,
        lon,
        body.timestamp,
        max_magnitude=body.max_magnitude,
        min_altitude_deg=body.min_altitude_deg,
        azimuth_deg=body.azimuth_deg,
        azimuth_tolerance_deg=body.azimuth_tolerance_deg,
    )
    if body.city:
        result["city"] = body.city
    return JSONResponse(content=result)
