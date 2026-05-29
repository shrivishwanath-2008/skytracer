"""Paths and directories used by the API layer."""

from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = BACKEND_DIR.parent
FRONTEND_DIR = PROJECT_ROOT / "frontend"
UPLOADS_DIR = BACKEND_DIR / "uploads"
STATIC_DIR = BACKEND_DIR / "static"
DEBUG_DIR = STATIC_DIR / "debug"
OUTPUT_DIR = STATIC_DIR / "output"

for directory in (UPLOADS_DIR, STATIC_DIR, DEBUG_DIR, OUTPUT_DIR):
    directory.mkdir(parents=True, exist_ok=True)
