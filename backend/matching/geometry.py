"""
2D affine map between image pixels and tangent-plane sky coordinates.

One reference star fixes scale and rotation relative to the user-supplied field center.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from backend.astronomy.projection import altaz_to_tangent, tangent_to_altaz


@dataclass
class PixelSkyTransform:
    """
    Maps offsets from the image center to tangent-plane coordinates (radians).

    pixel_dx = scale * (cos_r * xi - sin_r * eta)
    pixel_dy = -scale * (sin_r * xi + cos_r * eta)   # image y points down
    """

    center_x: float
    center_y: float
    center_alt_deg: float
    center_az_deg: float
    scale_pixels_per_rad: float
    rotation_rad: float

    @property
    def scale_pixels_per_deg(self) -> float:
        return self.scale_pixels_per_rad * math.degrees(1.0)

    @property
    def field_of_view_deg(self) -> tuple[float, float]:
        """Rough FOV if applied to full image size (width, height)."""
        # Caller supplies image size when needed
        return (0.0, 0.0)

    def pixel_to_tangent(self, x: float, y: float) -> tuple[float, float]:
        """Image pixel → tangent plane (xi, eta) in radians."""
        dx = x - self.center_x
        dy = y - self.center_y
        # Inverse of forward with y-flip
        u = dx / self.scale_pixels_per_rad
        v = -dy / self.scale_pixels_per_rad
        cos_r = math.cos(self.rotation_rad)
        sin_r = math.sin(self.rotation_rad)
        xi = cos_r * u + sin_r * v
        eta = -sin_r * u + cos_r * v
        return xi, eta

    def pixel_to_altaz(self, x: float, y: float) -> tuple[float, float]:
        xi, eta = self.pixel_to_tangent(x, y)
        return tangent_to_altaz(xi, eta, self.center_alt_deg, self.center_az_deg)

    def tangent_to_pixel(self, xi: float, eta: float) -> tuple[float, float]:
        """Tangent plane → image pixel."""
        cos_r = math.cos(self.rotation_rad)
        sin_r = math.sin(self.rotation_rad)
        dx = self.scale_pixels_per_rad * (cos_r * xi - sin_r * eta)
        dy = -self.scale_pixels_per_rad * (sin_r * xi + cos_r * eta)
        return self.center_x + dx, self.center_y + dy

    def altaz_to_pixel(self, altitude_deg: float, azimuth_deg: float) -> tuple[float, float]:
        xi, eta = altaz_to_tangent(
            altitude_deg, azimuth_deg, self.center_alt_deg, self.center_az_deg
        )
        return self.tangent_to_pixel(xi, eta)


def fit_transform_from_reference(
    *,
    center_x: float,
    center_y: float,
    center_alt_deg: float,
    center_az_deg: float,
    ref_pixel_x: float,
    ref_pixel_y: float,
    ref_alt_deg: float,
    ref_az_deg: float,
    fov_deg: float | None = None,
    image_width: float | None = None,
) -> PixelSkyTransform:
    """
    Estimate scale and rotation so the reference pixel matches the reference star.

    If fov_deg and image_width are given, scale is fixed from horizontal FOV and only
    rotation is fitted (useful when the user trusts lens FOV more than star distance).
    """
    xi, eta = altaz_to_tangent(ref_alt_deg, ref_az_deg, center_alt_deg, center_az_deg)
    dx = ref_pixel_x - center_x
    dy = ref_pixel_y - center_y

    # Complex numbers encode 2D vectors: one sky offset, one pixel offset.
    # scale = |pixel| / |sky|; rotation = arg(pixel) - arg(sky).
    # Image y is flipped (y-down) so we use (dx, -dy) as the pixel vector.
    z_sky = complex(xi, eta)
    z_pix = complex(dx, -dy)

    if abs(z_sky) < 1e-15:
        raise ValueError("Reference star is at the field center; pick an off-center star.")

    if fov_deg is not None and image_width is not None and fov_deg > 0:
        # User-supplied horizontal FOV fixes plate scale; only rotation is free.
        scale = image_width / math.radians(fov_deg)
        z_fit = z_pix / scale
        rotation = math.atan2(z_fit.imag, z_fit.real) - math.atan2(z_sky.imag, z_sky.real)
    else:
        # One reference star constrains both scale and rotation.
        scale = abs(z_pix) / abs(z_sky)
        rotation = math.atan2(z_pix.imag, z_pix.real) - math.atan2(z_sky.imag, z_sky.real)

    if scale <= 0:
        raise ValueError("Could not estimate a positive image scale.")

    return PixelSkyTransform(
        center_x=center_x,
        center_y=center_y,
        center_alt_deg=center_alt_deg,
        center_az_deg=center_az_deg,
        scale_pixels_per_rad=scale,
        rotation_rad=rotation,
    )
