"""
FastAPI application factory.

HTTP routes live in api/routes/; astronomy and vision logic stay in their packages.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.api.config import FRONTEND_DIR, STATIC_DIR, UPLOADS_DIR
from backend.api.routes import align, annotate, core, detect, match, visible

app = FastAPI(
    title="Skytrace",
    description="Hobby DSLR night-sky annotation — detect, match, and label bright stars",
    version="0.6.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
app.mount("/uploads", StaticFiles(directory=str(UPLOADS_DIR)), name="uploads")
app.mount("/static-ui", StaticFiles(directory=str(FRONTEND_DIR)), name="frontend-ui")

app.include_router(core.router)
app.include_router(visible.router)
app.include_router(detect.router)
app.include_router(align.router)
app.include_router(match.router)
app.include_router(annotate.router)
