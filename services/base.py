"""Shared bits for the FastAPI microservices."""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from fastapi import FastAPI  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402

SERVICES = ["income", "credit", "agents", "rl", "xai", "feedback"]
INPROCESS = bool(os.getenv("INPROCESS"))      # serverless mode: services mounted in one process, no ports
PORTS = {"gateway": 8100, "income": 8101, "credit": 8102, "agents": 8103, "rl": 8104, "xai": 8105, "feedback": 8106}
URLS = {k: f"http://{k}.local" for k in SERVICES} if INPROCESS else {
    k: os.getenv(f"{k.upper()}_URL", f"http://127.0.0.1:{PORTS[k]}") for k in SERVICES
}


def make_app(name: str) -> FastAPI:
    app = FastAPI(title=f"{name}-service")
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

    @app.get("/health")
    def health():
        return {"service": name, "status": "ok"}

    return app


def to_params(d: dict) -> dict:
    """JSON simulator parameters (scalars or lists) -> arrays of length 1."""
    import numpy as np
    return {k: np.atleast_1d(np.asarray(v, float)) for k, v in d.items()}
