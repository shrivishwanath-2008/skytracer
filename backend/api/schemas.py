"""Request/response models for JSON endpoints."""

from pydantic import BaseModel, Field


class VisibleStarsRequest(BaseModel):
    latitude: float | None = Field(None, ge=-90, le=90)
    longitude: float | None = Field(None, ge=-180, le=180)
    city: str | None = None
    timestamp: str = Field(..., description="UTC ISO-8601")
    max_magnitude: float = Field(3.5, ge=-2, le=6.5)
    min_altitude_deg: float = Field(0.0, ge=0, le=90)
    azimuth_deg: float | None = Field(None, ge=0, lt=360)
    azimuth_tolerance_deg: float = Field(45.0, ge=1, le=180)
