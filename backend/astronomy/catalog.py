"""
Lightweight bright-star catalog (JSON on disk).

~100–300 stars with name, RA (hours), Dec (degrees), and V magnitude.
No Gaia or full Hipparcos at runtime — only this bundled file is loaded.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

DATA_DIR = Path(__file__).resolve().parent / "data"
DEFAULT_CATALOG = DATA_DIR / "bright_stars.json"


@dataclass(frozen=True)
class CatalogStar:
    """One row from the bright-star catalog."""

    name: str
    ra_hours: float
    dec_deg: float
    magnitude: float
    hip: int | None = None

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "name": self.name,
            "ra_hours": self.ra_hours,
            "dec_deg": self.dec_deg,
            "magnitude": self.magnitude,
        }
        if self.hip is not None:
            out["hip"] = self.hip
        return out


@lru_cache(maxsize=1)
def load_catalog(path: Path | None = None) -> list[CatalogStar]:
    """
    Load the bundled bright-star catalog.

    Cached after first read. Raises FileNotFoundError if JSON is missing.
    """
    catalog_path = path or DEFAULT_CATALOG
    if not catalog_path.is_file():
        raise FileNotFoundError(
            f"Catalog not found: {catalog_path}. "
            "Run: python scripts/build_bright_catalog.py"
        )

    raw = json.loads(catalog_path.read_text(encoding="utf-8"))
    stars: list[CatalogStar] = []
    for row in raw:
        stars.append(
            CatalogStar(
                name=str(row["name"]),
                ra_hours=float(row["ra_hours"]),
                dec_deg=float(row["dec_deg"]),
                magnitude=float(row["magnitude"]),
                hip=int(row["hip"]) if row.get("hip") is not None else None,
            )
        )
    return stars


def filter_by_magnitude(
    stars: list[CatalogStar],
    max_magnitude: float,
) -> list[CatalogStar]:
    """Keep stars at or brighter than max_magnitude (lower mag number = brighter)."""
    return [s for s in stars if s.magnitude <= max_magnitude]


def catalog_stats(path: Path | None = None) -> dict[str, Any]:
    """Summary for debugging / API metadata."""
    stars = load_catalog(path)
    mags = [s.magnitude for s in stars]
    return {
        "count": len(stars),
        "path": str(path or DEFAULT_CATALOG),
        "brightest": min(mags) if mags else None,
        "faintest": max(mags) if mags else None,
    }
