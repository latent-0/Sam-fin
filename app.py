"""Vercel entry point (FastAPI is detected from this file): the whole platform as one ASGI app.

Each service is still its own FastAPI app; the gateway calls them in-process (INPROCESS=1).
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("INPROCESS", "1")
os.environ.setdefault("FEEDBACK_DB", "/tmp/feedback.db")      # deployment filesystem is read-only except /tmp
sys.path.insert(0, str(Path(__file__).resolve().parent))

from services.gateway import app  # noqa: E402,F401
