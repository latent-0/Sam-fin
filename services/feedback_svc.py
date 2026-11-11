"""Feedback service: stores ratings (SQLite) and serves the per-bucket preference bias."""
import os
from functools import lru_cache

from fastapi import HTTPException
from pydantic import BaseModel, Field

from services.base import ROOT, make_app
from fa import feedback as F

app = make_app("feedback")


@lru_cache(maxsize=1)
def store() -> F.Store:
    return F.Store(os.getenv("FEEDBACK_DB", str(ROOT / "artifacts" / "feedback.db")))


class Rating(BaseModel):
    bucket: str
    mask: int = Field(..., ge=1, le=63)
    rating: float = Field(..., ge=1, le=5)


@app.post("/feedback")
def feedback(r: Rating):
    try:
        return store().add(r.bucket, r.mask, r.rating)
    except ValueError as e:
        raise HTTPException(422, str(e))


@app.get("/bias/{bucket}")
def bias(bucket: str):
    if bucket not in F.BUCKETS:
        raise HTTPException(404, "unknown bucket")
    return store().get(bucket)
