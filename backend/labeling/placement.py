"""
Place text labels on an image while reducing overlap.

Simple greedy placement: try anchor offsets in priority order, skip if the
bounding box intersects an already-placed label.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class LabelSpec:
    """A label to place near (x, y)."""

    text: str
    x: float
    y: float
    kind: str = "star"  # star | planet | constellation
    priority: int = 50  # higher = placed first
    color_bgr: tuple[int, int, int] = (0, 255, 255)
    font_scale: float = 0.5


@dataclass
class PlacedLabel:
    text: str
    x: int
    y: int
    color_bgr: tuple[int, int, int]
    font_scale: float
    kind: str
    anchor_x: float
    anchor_y: float


# (dx, dy) offsets from anchor in pixels — tried in order
_OFFSETS: list[tuple[int, int]] = [
    (10, -6),
    (10, 12),
    (-80, -6),
    (-80, 12),
    (10, -28),
    (10, 28),
    (-120, -6),
    (0, -32),
    (0, 28),
]


def _estimate_size(text: str, font_scale: float) -> tuple[int, int]:
    """Rough text box (width, height) in pixels for OpenCV Hershey font."""
    w = int(len(text) * 9 * font_scale + 4)
    h = int(18 * font_scale + 4)
    return w, h


def _boxes_overlap(
    ax: int,
    ay: int,
    aw: int,
    ah: int,
    bx: int,
    by: int,
    bw: int,
    bh: int,
    margin: int = 4,
) -> bool:
    return not (
        ax + aw + margin < bx
        or bx + bw + margin < ax
        or ay + ah + margin < by
        or by + bh + margin < ay
    )


def place_labels(
    specs: list[LabelSpec],
    image_width: int,
    image_height: int,
) -> list[PlacedLabel]:
    """
    Greedy non-overlapping placement sorted by priority (descending).
    """
    placed_boxes: list[tuple[int, int, int, int]] = []
    result: list[PlacedLabel] = []

    for spec in sorted(specs, key=lambda s: -s.priority):
        tw, th = _estimate_size(spec.text, spec.font_scale)
        best: tuple[int, int] | None = None

        for dx, dy in _OFFSETS:
            tx = int(spec.x + dx)
            ty = int(spec.y + dy)
            if tx < 2 or ty < 2 or tx + tw > image_width - 2 or ty + th > image_height - 2:
                continue
            if any(_boxes_overlap(tx, ty, tw, th, *box) for box in placed_boxes):
                continue
            best = (tx, ty)
            break

        if best is None:
            # Fallback: place anyway at default offset (may overlap)
            best = (int(spec.x + 10), int(spec.y - 6))
            best = (
                max(2, min(best[0], image_width - tw - 2)),
                max(2, min(best[1], image_height - th - 2)),
            )

        tx, ty = best
        tw, th = _estimate_size(spec.text, spec.font_scale)
        placed_boxes.append((tx, ty, tw, th))
        result.append(
            PlacedLabel(
                text=spec.text,
                x=tx,
                y=ty,
                color_bgr=spec.color_bgr,
                font_scale=spec.font_scale,
                kind=spec.kind,
                anchor_x=spec.x,
                anchor_y=spec.y,
            )
        )

    return result


def specs_from_annotations(
    items: list[dict[str, Any]],
    *,
    image_scale: float = 1.0,
) -> list[LabelSpec]:
    """Build LabelSpec list from annotate pipeline dicts."""
    specs: list[LabelSpec] = []
    for item in items:
        kind = item.get("type", "star")
        if kind == "planet":
            priority, color, scale = 90, (80, 180, 255), 0.55 * image_scale
        elif kind == "constellation":
            priority, color, scale = 20, (180, 140, 255), 0.7 * image_scale
        else:
            priority, color, scale = 60, (0, 255, 255), 0.5 * image_scale
        specs.append(
            LabelSpec(
                text=str(item["name"]),
                x=float(item["pixel_x"]),
                y=float(item["pixel_y"]),
                kind=kind,
                priority=priority,
                color_bgr=color,
                font_scale=max(0.35, min(0.9, scale)),
            )
        )
    return specs
