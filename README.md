# Skytrace

A **practical astrophotography assistant for hobbyists** — upload a DSLR night-sky photo, get labeled stars (and optional planets / constellation names). Lightweight and explainable; not a research-grade astrometry engine.

## Goals

- Urban / light-polluted skies, sparse stars, kit lenses (~55mm)
- Modular, explainable pipeline (OpenCV detection + small bright-star catalog)
- One working deliverable per development phase

See **[docs/DEVELOPMENT.md](docs/DEVELOPMENT.md)** for coding style, phase discipline, and verification commands.

## Project layout

```
skytrace1/
├── backend/
│   ├── main.py              # uvicorn entry (imports api.app)
│   ├── api/                 # HTTP routes only
│   ├── vision/              # Phase 1
│   ├── astronomy/           # Phase 2
│   ├── matching/            # Phase 3–4
│   ├── labeling/            # Phase 5
│   ├── static/              # Output + debug images
│   └── uploads/             # Uploaded originals
├── frontend/                # Minimal upload UI
├── requirements.txt
└── README.md
```

## Setup

```bash
cd skytrace1
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

## Run API

From the **project root** (`skytrace1/`):

```bash
python -m uvicorn backend.main:app --reload
```

- API docs: http://127.0.0.1:8000/docs  
- Health: http://127.0.0.1:8000/health  

## Phase 1 endpoint

**`POST /detect-stars`** — multipart form field `file` (JPEG/PNG)

Response:

```json
{
  "annotated_image": "/static/output/<id>_annotated.jpg",
  "star_count": 12,
  "stars": [
    { "x": 100, "y": 200, "brightness": 210.5, "radius": 3.2 }
  ],
  "debug": {
    "threshold_mask": "/static/debug/<id>_threshold_mask.png",
    "detected_blobs": "/static/debug/<id>_detected_blobs.png",
    "final_overlay": "/static/debug/<id>_final_overlay.png"
  }
}
```

### curl example

```bash
curl -X POST "http://127.0.0.1:8000/detect-stars" \
  -F "file=@your_night_sky.jpg"
```

## Frontend (Phase 6)

With the API running, open **http://127.0.0.1:8000/** in your browser.

Workflow:

1. Upload a night-sky image  
2. Enter **city**, **direction** (compass), **height above horizon**, and **UTC time**  
3. Click **Process image**  
4. View the labeled result  

Static assets live in `frontend/` (`index.html`, `style.css`, `app.js`).

## Verify all phases

```bash
python scripts/verify_all.py
```

## CLI smoke test (synthetic stars)

```bash
python scripts/smoke_test.py
```

Generates a synthetic star field, runs the vision pipeline, and writes debug images under `backend/static/debug/`.

## Pipeline (Phase 1)

1. **Preprocess** — grayscale → Gaussian blur → median denoise → CLAHE contrast  
2. **Detect** — adaptive threshold + contours, SimpleBlobDetector, merge nearby  
3. **Overlay** — green circles + optional indices → save JPEG/PNG  

## Phase 2 — visible bright stars

Bundled catalog: `backend/astronomy/data/bright_stars.json` (~150+ bright stars, no Gaia).

**`POST /visible-stars`** — JSON body:

```json
{
  "city": "new york",
  "timestamp": "2024-07-15T02:00:00Z",
  "max_magnitude": 3.0,
  "min_altitude_deg": 5.0,
  "azimuth_deg": 180,
  "azimuth_tolerance_deg": 45
}
```

Or use `"latitude": 40.71, "longitude": -74.01` instead of `city`.

Response stars include `altitude_deg`, `azimuth_deg`, `magnitude`, `ra_hours`, `dec_deg`, `name`.

```bash
curl -X POST http://127.0.0.1:8000/visible-stars \
  -H "Content-Type: application/json" \
  -d "{\"city\":\"new york\",\"timestamp\":\"2024-07-15T02:00:00Z\",\"max_magnitude\":3.0}"
```

Smoke test:

```bash
python scripts/test_visible_stars.py
```

Rebuild catalog from Hipparcos (dev only, downloads ~1 MB):

```bash
python scripts/build_bright_catalog.py --limit 250 --max-mag 4.5
```

## Phase 3 — semi-manual image ↔ sky alignment

**Not** automatic plate solving. You provide:

1. Approximate **look direction** (where the image center points: azimuth + altitude).
2. One **reference star**: name (e.g. `Vega`) + which detection is that star (`detection_index` from `/detect-stars`, 0-based).

The solver fits **scale** and **rotation** (affine map on the tangent plane) from that single match, then maps every detection to estimated alt/az and draws predicted catalog positions for validation.

**`POST /align-sky`** — multipart form:

| Field | Description |
|-------|-------------|
| `file` | Image |
| `timestamp` | UTC ISO-8601 |
| `latitude` / `longitude` or `city` | Observer location |
| `center_azimuth_deg`, `center_altitude_deg` | Approximate center of view |
| `reference_star_name` | e.g. `Vega` |
| `detection_index` | 0-based index on annotated detections |
| `fov_deg` | Optional: fix scale from lens FOV, fit rotation only |

```bash
python scripts/test_align.py
```

## Phase 4 — geometric pattern matching

Compares **relative geometry** of detections vs. visible catalog stars:

1. **Triangle matching** — equal side ratios + angles (scale-invariant) → votes for star pairs  
2. **Distance-ratio signatures** — from each star, sorted ratios to neighbors  
3. **Angular signatures** — angles between rays (law of cosines)

**`POST /match-stars`** — multipart: `file`, `timestamp`, location, optional `center_azimuth_deg` to narrow catalog.

```bash
python scripts/test_geometric_match.py
```

## Phase 5 — labeling & final image

**`POST /annotate`** — full pipeline: detect → geometric match → sky alignment (best match or your reference star) → labels.

| Label type | Color (approx.) | Notes |
|------------|-----------------|-------|
| Stars | Yellow | From geometric match at detection pixels |
| Planets | Orange | Projected via alignment (Skyfield) |
| Constellations | Purple | Centroid of visible member stars |

Text uses greedy placement with alternate offsets to reduce overlap; font size scales with image width.

```bash
python scripts/test_annotate.py
```

Use the web UI at http://127.0.0.1:8000/ (see Frontend above).

## Roadmap

| Phase | Scope |
|-------|--------|
| 1–4 | Detection → catalog → align → geometry |
| 5 | Final labeled image (implemented) |

## Tuning

Edit parameters in `backend/vision/detect_stars.py` (`block_size`, `threshold_c`, `min_area`, `max_area`) and `backend/vision/preprocess.py` (CLAHE, blur) for your camera, lens, and sky conditions.
