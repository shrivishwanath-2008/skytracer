"""Phase 1 — star detection only."""

from __future__ import annotations

import uuid
from pathlib import Path

from fastapi import APIRouter, File, UploadFile
from fastapi.responses import JSONResponse

from backend.api.config import UPLOADS_DIR
from backend.api.helpers import decode_uploaded_image
from backend.api.vision_pipeline import run_detection_with_debug

router = APIRouter()


@router.post("/detect-stars")
async def detect_stars_endpoint(
    file: UploadFile = File(..., description="DSLR night-sky image"),
) -> JSONResponse:
    if file.content_type and not file.content_type.startswith("image/"):
        return JSONResponse(status_code=400, content={"error": "Upload must be an image file"})

    data = await file.read()
    if not data:
        return JSONResponse(status_code=400, content={"error": "Empty file"})

    try:
        bgr = decode_uploaded_image(data)
    except ValueError as exc:
        return JSONResponse(status_code=400, content={"error": str(exc)})

    stem = uuid.uuid4().hex[:12]
    suffix = Path(file.filename or "image.jpg").suffix or ".jpg"
    (UPLOADS_DIR / f"{stem}{suffix}").write_bytes(data)

    result = run_detection_with_debug(bgr, stem)
    result["upload"] = f"/uploads/{stem}{suffix}"
    return JSONResponse(content=result)
