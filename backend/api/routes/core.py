"""Health, API info, frontend, catalog stats."""

from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import FileResponse

from backend.api.config import FRONTEND_DIR
from backend.astronomy.catalog import catalog_stats

router = APIRouter()


@router.get("/")
def serve_frontend() -> FileResponse:
    return FileResponse(FRONTEND_DIR / "index.html")


@router.get("/api")
def api_info() -> dict[str, str]:
    return {
        "service": "skytrace",
        "ui": "/",
        "docs": "/docs",
        "annotate": "POST /annotate",
    }


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/catalog/stats")
def catalog_stats_endpoint() -> dict:
    return catalog_stats()
