"""Vercel entry point: exposes the draftadvisor FastAPI app (stateless mode).

Vercel's Python runtime imports ``app`` from this file. The package lives one
directory up; the lean dependency set is in requirements.txt (no pandas/sklearn).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("DRAFTADVISOR_HOME", "/tmp/draftadvisor")
os.environ.setdefault("DRAFTADVISOR_STATELESS", "1")

from draftadvisor.web.server import app  # noqa: E402

__all__ = ["app"]
