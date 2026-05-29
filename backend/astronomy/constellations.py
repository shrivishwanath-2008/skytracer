"""
Lightweight constellation labels from bundled member-star names.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

DATA_FILE = Path(__file__).resolve().parent / "data" / "constellations.json"


@lru_cache(maxsize=1)
def load_constellation_map() -> dict[str, list[str]]:
    return json.loads(DATA_FILE.read_text(encoding="utf-8"))


def constellation_label_positions(
    visible_stars: list[dict[str, Any]],
    *,
    min_members: int = 2,
) -> list[dict[str, Any]]:
    """
    For each constellation with enough visible named members, return a label at
    the average alt/az of those members.
    """
    name_to_star = {s["name"].lower(): s for s in visible_stars}
    labels: list[dict[str, Any]] = []

    for constellation, members in load_constellation_map().items():
        found = [name_to_star[m.lower()] for m in members if m.lower() in name_to_star]
        if len(found) < min_members:
            continue
        alt = sum(float(s["altitude_deg"]) for s in found) / len(found)
        az = sum(float(s["azimuth_deg"]) for s in found) / len(found)
        labels.append({
            "name": constellation,
            "type": "constellation",
            "altitude_deg": round(alt, 3),
            "azimuth_deg": round(az % 360, 3),
            "member_count": len(found),
        })

    return labels
