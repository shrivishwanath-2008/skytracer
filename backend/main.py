"""
Entry point for uvicorn.

  python -m uvicorn backend.main:app --reload
"""

from backend.api.app import app

__all__ = ["app"]
