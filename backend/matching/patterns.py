"""
Scale-invariant geometric patterns for star-field matching.

Why this works for DSLR photos:
  - Pixel distances between stars are unknown in arcseconds until we solve scale.
  - Ratios of sides (short/medium/long) and interior angles are the same for
    similar triangles whether measured in pixels or in degrees on the sky.
  - We never need a full plate solve — only pattern agreement + voting.

Triangle interior angles use the law of cosines on the three side lengths.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Callable, Sequence


@dataclass(frozen=True)
class TriangleShape:
    """Sorted side ratios and angles — identifies triangle shape only."""

    ratio_small: float  # shortest / longest
    ratio_mid: float  # middle / longest
    angles_deg: tuple[float, float, float]  # sorted ascending

    def close_to(self, other: TriangleShape, *, ratio_tol: float, angle_tol_deg: float) -> bool:
        if abs(self.ratio_small - other.ratio_small) > ratio_tol:
            return False
        if abs(self.ratio_mid - other.ratio_mid) > ratio_tol:
            return False
        for a, b in zip(self.angles_deg, other.angles_deg):
            if abs(a - b) > angle_tol_deg:
                return False
        return True


@dataclass
class LabeledTriangle:
    """Triangle with vertex labels (indices into a point list)."""

    vertex_ids: tuple[int, int, int]
    sides: tuple[float, float, float]  # opposite vertex order (d01, d12, d20) unsorted
    shape: TriangleShape


def pairwise_distances(
    n: int,
    dist: Callable[[int, int], float],
) -> list[list[float]]:
    """Build n×n symmetric distance matrix; dist(i,j) for i < j."""
    m = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            d = dist(i, j)
            m[i][j] = m[j][i] = d
    return m


def _angles_from_sides(a: float, b: float, c: float) -> tuple[float, float, float]:
    """Interior angles (degrees) opposite sides a, b, c respectively."""
    def angle(opposite: float, s1: float, s2: float) -> float:
        if s1 <= 0 or s2 <= 0:
            return 0.0
        cos_v = (s1 * s1 + s2 * s2 - opposite * opposite) / (2 * s1 * s2)
        cos_v = max(-1.0, min(1.0, cos_v))
        return math.degrees(math.acos(cos_v))

    return (
        angle(a, b, c),
        angle(b, a, c),
        angle(c, a, b),
    )


def triangle_shape(sides: tuple[float, float, float]) -> TriangleShape | None:
    """
    Normalized shape from three side lengths.

    Returns None for degenerate (collinear) triangles.
    """
    a, b, c = sides
    if min(a, b, c) <= 1e-9:
        return None
    s = sorted((a, b, c))
    angles = _angles_from_sides(s[0], s[1], s[2])
    return TriangleShape(
        ratio_small=s[0] / s[2],
        ratio_mid=s[1] / s[2],
        angles_deg=tuple(sorted(angles)),
    )


def build_triangles(
    n_points: int,
    dist_matrix: list[list[float]],
) -> list[LabeledTriangle]:
    """All triangles from point indices 0..n-1."""
    triangles: list[LabeledTriangle] = []
    for i, j, k in itertools.combinations(range(n_points), 3):
        d_ij = dist_matrix[i][j]
        d_jk = dist_matrix[j][k]
        d_ki = dist_matrix[k][i]
        shape = triangle_shape((d_ij, d_jk, d_ki))
        if shape is None:
            continue
        triangles.append(
            LabeledTriangle(
                vertex_ids=(i, j, k),
                sides=(d_ij, d_jk, d_ki),
                shape=shape,
            )
        )
    return triangles


def vertex_permutations(
    cat_ids: tuple[int, int, int],
    det_ids: tuple[int, int, int],
) -> list[tuple[tuple[int, int], tuple[int, int], tuple[int, int]]]:
    """All 6 bijections between catalog and detection triangle vertices."""
    c = cat_ids
    d = det_ids
    return [
        ((c[0], d[0]), (c[1], d[1]), (c[2], d[2])),
        ((c[0], d[0]), (c[1], d[2]), (c[2], d[1])),
        ((c[0], d[1]), (c[1], d[0]), (c[2], d[2])),
        ((c[0], d[1]), (c[1], d[2]), (c[2], d[0])),
        ((c[0], d[2]), (c[1], d[0]), (c[2], d[1])),
        ((c[0], d[2]), (c[1], d[1]), (c[2], d[0])),
    ]


def distance_ratio_signature(
    center: int,
    others: list[int],
    dist_matrix: list[list[float]],
) -> tuple[float, ...]:
    """
    Sorted distance ratios from center to other points.

    Invariant to uniform scaling — used for pairwise pattern comparison.
    """
    if len(others) < 2:
        return ()
    dists = [dist_matrix[center][o] for o in others if dist_matrix[center][o] > 1e-9]
    if len(dists) < 2:
        return ()
    base = min(dists)
    ratios = sorted(d / base for d in dists)
    return tuple(round(r, 4) for r in ratios)


def angular_signature_at_vertex(
    vertex: int,
    others: list[int],
    dist_matrix: list[list[float]],
) -> tuple[float, ...]:
    """
    Sorted angles (degrees) at vertex between rays to other stars.

    Computed via law of cosines; scale-invariant.
    """
    if len(others) < 2:
        return ()
    angles: list[float] = []
    for i in range(len(others)):
        for j in range(i + 1, len(others)):
            a, b = others[i], others[j]
            dab = dist_matrix[vertex][a]
            dac = dist_matrix[vertex][b]
            dbc = dist_matrix[a][b]
            if min(dab, dac, dbc) <= 1e-9:
                continue
            cos_v = (dab * dab + dac * dac - dbc * dbc) / (2 * dab * dac)
            cos_v = max(-1.0, min(1.0, cos_v))
            angles.append(math.degrees(math.acos(cos_v)))
    return tuple(round(a, 2) for a in sorted(angles))
