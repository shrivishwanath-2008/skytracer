"""
Lightweight geometric matching: detected stars ↔ visible catalog stars.

Methods (all explainable, no optimization solvers):
  1. Triangle matching — equal shape (side ratios + angles) → vote on vertex pairs
  2. Distance-ratio signatures — compare scale-invariant distance patterns from each star
  3. Angular signatures — compare angles between rays at each star

Works without plate solving; optional alignment transform can tighten distance checks.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from backend.astronomy.projection import angular_separation_deg
from backend.astronomy.visible_stars import visible_stars
from backend.matching.geometry import PixelSkyTransform
from backend.matching.patterns import (
    LabeledTriangle,
    angular_signature_at_vertex,
    build_triangles,
    distance_ratio_signature,
    pairwise_distances,
    triangle_shape,
    vertex_permutations,
)


@dataclass
class GeometricMatch:
    detection_index: int
    catalog_name: str
    votes: int
    confidence: float
    methods: list[str] = field(default_factory=list)


def _select_brightest_detections(
    detections: list[dict[str, Any]],
    max_points: int,
) -> list[tuple[int, dict[str, Any]]]:
    indexed = list(enumerate(detections))
    indexed.sort(key=lambda t: -float(t[1].get("brightness", 0)))
    return indexed[:max_points]


def _select_brightest_catalog(
    catalog: list[dict[str, Any]],
    max_points: int,
) -> list[tuple[int, dict[str, Any]]]:
    indexed = list(enumerate(catalog))
    indexed.sort(key=lambda t: float(t[1].get("magnitude", 99)))
    return indexed[:max_points]


def _catalog_dist(i: int, j: int, cats: list[dict[str, Any]]) -> float:
    a, b = cats[i], cats[j]
    return angular_separation_deg(
        float(a["altitude_deg"]),
        float(a["azimuth_deg"]),
        float(b["altitude_deg"]),
        float(b["azimuth_deg"]),
    )


def _pixel_dist(
    i: int,
    j: int,
    dets: list[dict[str, Any]],
) -> float:
    a, b = dets[i], dets[j]
    return math.hypot(float(a["x"]) - float(b["x"]), float(a["y"]) - float(b["y"]))


def _vote_from_triangles(
    cat_tris: list[LabeledTriangle],
    det_tris: list[LabeledTriangle],
    cat_index_map: list[int],
    det_index_map: list[int],
    *,
    ratio_tol: float,
    angle_tol_deg: float,
) -> dict[tuple[int, int], int]:
    """Vote (catalog_local, detection_local) pairs from matching triangle shapes."""
    votes: dict[tuple[int, int], int] = defaultdict(int)

    for ct in cat_tris:
        for dt in det_tris:
            if not ct.shape.close_to(dt.shape, ratio_tol=ratio_tol, angle_tol_deg=angle_tol_deg):
                continue
            for perm in vertex_permutations(ct.vertex_ids, dt.vertex_ids):
                # perm is ((c0,d0),(c1,d1),(c2,d2)) in local triangle indices
                for (cl, dl) in perm:
                    votes[(cl, dl)] += 1

    return votes


def _vote_from_distance_ratios(
    n_cat: int,
    n_det: int,
    cat_dist: list[list[float]],
    det_dist: list[list[float]],
    *,
    ratio_tol: float,
) -> dict[tuple[int, int], int]:
    votes: dict[tuple[int, int], int] = defaultdict(int)

    for ci in range(n_cat):
        cat_others = [j for j in range(n_cat) if j != ci]
        cat_sig = distance_ratio_signature(ci, cat_others, cat_dist)
        if len(cat_sig) < 2:
            continue
        for di in range(n_det):
            det_others = [j for j in range(n_det) if j != di]
            det_sig = distance_ratio_signature(di, det_others, det_dist)
            if len(det_sig) != len(cat_sig):
                continue
            if _signatures_close(cat_sig, det_sig, ratio_tol):
                votes[(ci, di)] += 2

    return votes


def _vote_from_angles(
    n_cat: int,
    n_det: int,
    cat_dist: list[list[float]],
    det_dist: list[list[float]],
    *,
    angle_tol_deg: float,
) -> dict[tuple[int, int], int]:
    votes: dict[tuple[int, int], int] = defaultdict(int)

    for ci in range(n_cat):
        cat_others = [j for j in range(n_cat) if j != ci]
        cat_sig = angular_signature_at_vertex(ci, cat_others, cat_dist)
        if len(cat_sig) < 1:
            continue
        for di in range(n_det):
            det_others = [j for j in range(n_det) if j != di]
            det_sig = angular_signature_at_vertex(di, det_others, det_dist)
            if len(cat_sig) != len(det_sig):
                continue
            if _signatures_close(cat_sig, det_sig, angle_tol_deg / 10):
                votes[(ci, di)] += 2

    return votes


def _signatures_close(
    a: tuple[float, ...],
    b: tuple[float, ...],
    tol: float,
) -> bool:
    if len(a) != len(b):
        return False
    return all(abs(x - y) <= tol for x, y in zip(a, b))


def _merge_votes(*vote_dicts: dict[tuple[int, int], int]) -> dict[tuple[int, int], int]:
    merged: dict[tuple[int, int], int] = defaultdict(int)
    for vd in vote_dicts:
        for k, v in vd.items():
            merged[k] += v
    return merged


def _assign_from_votes(
    votes: dict[tuple[int, int], int],
    cat_labels: list[str],
    det_original_indices: list[int],
    *,
    min_votes: int,
) -> list[GeometricMatch]:
    """Greedy one-to-one assignment by vote count."""
    pairs = sorted(votes.items(), key=lambda kv: -kv[1])
    used_cat: set[int] = set()
    used_det: set[int] = set()
    matches: list[GeometricMatch] = []

    max_vote = max(votes.values()) if votes else 1

    for (ci, di), count in pairs:
        if count < min_votes:
            break
        if ci in used_cat or di in used_det:
            continue
        used_cat.add(ci)
        used_det.add(di)
        methods = []
        if count >= 4:
            methods.append("triangle")
        if count >= 2:
            methods.append("distance_ratio")
        methods.append("angular")
        matches.append(
            GeometricMatch(
                detection_index=det_original_indices[di],
                catalog_name=cat_labels[ci],
                votes=count,
                confidence=round(min(1.0, count / max(max_vote, 1)), 3),
                methods=methods,
            )
        )

    return matches


def geometric_match(
    detections: list[dict[str, Any]],
    catalog_stars: list[dict[str, Any]],
    *,
    max_points: int = 10,
    ratio_tol: float = 0.1,
    angle_tol_deg: float = 10.0,
    min_votes: int = 3,
) -> dict[str, Any]:
    """
    Match detections to catalog stars using relative geometry only.

    Parameters
    ----------
    detections
        Phase-1 detections with x, y, brightness.
    catalog_stars
        Phase-2 visible stars with name, altitude_deg, azimuth_deg, magnitude.
    max_points
        Brightest stars used (keeps triangle count manageable).
    """
    if len(detections) < 3 or len(catalog_stars) < 3:
        return {
            "matches": [],
            "error": "Need at least 3 detections and 3 catalog stars",
            "debug": {},
        }

    det_sel = _select_brightest_detections(detections, max_points)
    cat_sel = _select_brightest_catalog(catalog_stars, max_points)

    det_list = [d for _, d in det_sel]
    cat_list = [c for _, c in cat_sel]
    det_orig_idx = [i for i, _ in det_sel]
    cat_names = [str(c["name"]) for c in cat_list]

    n_det, n_cat = len(det_list), len(cat_list)
    det_dist = pairwise_distances(n_det, lambda i, j: _pixel_dist(i, j, det_list))
    cat_dist = pairwise_distances(n_cat, lambda i, j: _catalog_dist(i, j, cat_list))

    det_tris = build_triangles(n_det, det_dist)
    cat_tris = build_triangles(n_cat, cat_dist)

    tri_votes = _vote_from_triangles(
        cat_tris,
        det_tris,
        list(range(n_cat)),
        list(range(n_det)),
        ratio_tol=ratio_tol,
        angle_tol_deg=angle_tol_deg,
    )
    dist_votes = _vote_from_distance_ratios(
        n_cat, n_det, cat_dist, det_dist, ratio_tol=ratio_tol
    )
    ang_votes = _vote_from_angles(
        n_cat, n_det, cat_dist, det_dist, angle_tol_deg=angle_tol_deg
    )

    all_votes = _merge_votes(tri_votes, dist_votes, ang_votes)
    matches = _assign_from_votes(
        all_votes,
        cat_names,
        det_orig_idx,
        min_votes=min_votes,
    )

    matched_det = {m.detection_index for m in matches}
    matched_cat = {m.catalog_name for m in matches}

    return {
        "matches": [
            {
                "detection_index": m.detection_index,
                "catalog_name": m.catalog_name,
                "votes": m.votes,
                "confidence": m.confidence,
                "methods": m.methods,
            }
            for m in matches
        ],
        "unmatched_detections": [
            i for i in range(len(detections)) if i not in matched_det
        ],
        "unmatched_catalog": [
            c["name"] for c in catalog_stars if c["name"] not in matched_cat
        ],
        "debug": {
            "points_used": {"detections": n_det, "catalog": n_cat},
            "triangles": {"detection": len(det_tris), "catalog": len(cat_tris)},
            "ratio_tolerance": ratio_tol,
            "angle_tolerance_deg": angle_tol_deg,
            "min_votes": min_votes,
            "explanation": (
                "Triangles with matching side ratios and angles vote for star pairs; "
                "distance-ratio and angular signatures add votes. "
                "Highest-vote one-to-one pairs are returned."
            ),
        },
    }


def match_field(
    *,
    latitude: float,
    longitude: float,
    timestamp: str,
    detections: list[dict[str, Any]],
    max_magnitude: float = 4.0,
    min_altitude_deg: float = 5.0,
    azimuth_deg: float | None = None,
    azimuth_tolerance_deg: float = 60.0,
    transform: PixelSkyTransform | None = None,
    **match_kwargs: Any,
) -> dict[str, Any]:
    """
    Load visible catalog for time/place, run geometric_match, optional alignment check.
    """
    catalog = visible_stars(
        latitude,
        longitude,
        timestamp,
        max_magnitude=max_magnitude,
        min_altitude_deg=min_altitude_deg,
        azimuth_deg=azimuth_deg,
        azimuth_tolerance_deg=azimuth_tolerance_deg,
    )

    result = geometric_match(detections, catalog, **match_kwargs)
    result["catalog_count"] = len(catalog)

    if transform is not None:
        result["alignment_check"] = _alignment_check(matches=result["matches"], transform=transform, catalog=catalog, detections=detections)

    return result


def _alignment_check(
    *,
    matches: list[dict[str, Any]],
    transform: PixelSkyTransform,
    catalog: list[dict[str, Any]],
    detections: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """After geometric match, report pixel error if a transform from Phase 3 exists."""
    name_to_cat = {c["name"]: c for c in catalog}
    checks = []
    for m in matches:
        cat = name_to_cat.get(m["catalog_name"])
        det = detections[m["detection_index"]]
        if not cat:
            continue
        px, py = transform.altaz_to_pixel(
            float(cat["altitude_deg"]),
            float(cat["azimuth_deg"]),
        )
        err = math.hypot(px - float(det["x"]), py - float(det["y"]))
        checks.append({
            "catalog_name": m["catalog_name"],
            "pixel_error": round(err, 1),
        })
    return checks
