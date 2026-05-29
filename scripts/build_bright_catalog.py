"""
One-time dev script: build backend/astronomy/data/bright_stars.json from Hipparcos.

Run once during development; commit the JSON so runtime needs no catalog download.
  python scripts/build_bright_catalog.py --limit 250 --max-mag 4.5

Requires: pip install skyfield
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "backend" / "astronomy" / "data" / "bright_stars.json"

# Common proper names for well-known Hipparcos IDs (subset)
NAMES: dict[int, str] = {
    32349: "Sirius",
    30438: "Canopus",
    69673: "Arcturus",
    91262: "Vega",
    24608: "Capella",
    27989: "Rigel",
    24436: "Procyon",
    37279: "Betelgeuse",
    25336: "Achernar",
    21421: "Aldebaran",
    102098: "Spica",
    113368: "Antares",
    62434: "Pollux",
    72622: "Fomalhaut",
    65378: "Deneb",
    80763: "Altair",
    71683: "Alpha Centauri",
    61084: "Regulus",
    26311: "Adhara",
    82273: "Albireo",
    85927: "Alnair",
    63125: "Gacrux",
    60718: "Mimosa",
    45238: "Alioth",
    54061: "Dubhe",
    11767: "Polaris",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=250)
    parser.add_argument("--max-mag", type=float, default=4.5)
    args = parser.parse_args()

    try:
        from skyfield.api import Star, load
        from skyfield.data import hipparcos
    except ImportError:
        print("Install skyfield: pip install skyfield", file=sys.stderr)
        sys.exit(1)

    with load.open(hipparcos.URL) as f:
        df = hipparcos.load_dataframe(f)

    df = df.dropna(subset=["magnitude"])
    df = df[df["magnitude"] <= args.max_mag]
    df = df.sort_values("magnitude").head(args.limit)

    stars = []
    for hip, row in df.iterrows():
        hip_id = int(hip)
        name = NAMES.get(hip_id)
        if not name:
            # Bayer from proxied columns when present
            name = str(row.get("bf", "") or row.get("gl", "") or f"HIP {hip_id}").strip()
        stars.append({
            "hip": hip_id,
            "name": name,
            "ra_hours": round(float(row["ra_hours"]), 6),
            "dec_deg": round(float(row["dec_degrees"]), 6),
            "magnitude": round(float(row["magnitude"]), 2),
        })

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(stars, indent=2), encoding="utf-8")
    print(f"Wrote {len(stars)} stars to {OUT}")


if __name__ == "__main__":
    main()
