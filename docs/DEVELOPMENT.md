# Skytrace development guide

Skytrace is a **practical astrophotography assistant for hobbyists** — not a research-grade astrometry engine.

## Final goal

The system should:

- Work on real DSLR night-sky images (kit lenses, wide field)
- Function under light-polluted urban skies (sparse bright stars)
- Identify major stars and planets from a **small** bright-star catalog
- Annotate images well enough for hobby labeling and learning
- Stay lightweight, modular, and understandable

## Phase discipline

Implement **one phase at a time**. After each phase:

1. Stop — do not start the next phase in the same change set
2. Verify — run the phase smoke test (see below)
3. Commit a **working** intermediate result

| Phase | Deliverable | Verify with |
|-------|-------------|---------------|
| 1 | Star detection + overlay | `python scripts/smoke_test.py` |
| 2 | Visible catalog + alt/az | `python scripts/test_visible_stars.py` |
| 3 | Semi-manual alignment | `python scripts/test_align.py` |
| 4 | Geometric matching | `python scripts/test_geometric_match.py` |
| 5 | Final labeled image | `python scripts/test_annotate.py` |
| 6 | Web UI | Open http://127.0.0.1:8000/ after `uvicorn` |

Run everything: `python scripts/verify_all.py`

## Coding style

- **Modular functions** — one clear job per function; split files before ~250 lines
- **Clear names** — `altitude_deg`, `center_azimuth_deg`, not `a`, `ca`
- **Type hints** on public functions
- **Docstrings** on modules and public APIs (what / inputs / returns)
- **Astronomy comments** where math is non-obvious (gnomonic projection, spherical separation, scale-invariant triangles)
- **Readability over cleverness** — explicit steps beat one-liners
- **No jumping ahead** — e.g. do not add Gaia, ML plate solving, or auto-solve in a “cleanup” PR

## What we deliberately avoid

- Full Astrometry.net-style plate solving
- Machine learning for detection or matching (initially)
- Huge catalogs (Gaia DR3, full Hipparcos at runtime)
- Heavy numerical optimizers for field fitting

## Module map

```
vision/       Phase 1 — OpenCV detection
astronomy/    Phase 2 — catalog, Skyfield alt/az, planets, constellations
matching/     Phase 3–4 — alignment + geometric patterns
labeling/     Phase 5 — placement + final annotate pipeline
api/          Phase 6 — FastAPI routes (thin handlers)
```

Handlers in `api/routes/` should only parse HTTP, call domain code, and return JSON — no astronomy math in route files.
