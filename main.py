"""ASGI entry point for Vercel's FastAPI preset (same app as api/index.py and run.py)."""
from api.index import app  # noqa: F401

__all__ = ["app"]
