"""User ratings and the online preference update (paper Sec. IV-G).

Users rate a plan from 1 to 5. For the household's income bucket (variable if
CV > 0.15, else steady), the chosen subset's bias is updated by

    delta = lambda * (rating - r_bar),   lambda = 0.15,

where r_bar is the bucket's running mean rating (3 until five ratings exist).
One-agent neighbours receive delta / 4, and every bias is clipped to
[-0.25, 0.25]. At decision time the orchestrator takes argmax_a Q(s,a) + b_bucket(a).
"""
from __future__ import annotations

import sqlite3
import threading
from pathlib import Path

import numpy as np

LAMBDA = 0.15
CLIP = 0.25
PRIOR_MEAN = 3.0
MIN_RATINGS = 5
BUCKETS = ("steady", "variable")


def bucket_of(cv: float) -> str:
    return "variable" if cv > 0.15 else "steady"


def neighbours(mask: int) -> list:
    return [mask ^ (1 << i) for i in range(6) if mask ^ (1 << i)]


class Preferences:
    """In-memory bias table; used directly by the simulated feedback study."""

    def __init__(self):
        self.bias = {b: np.zeros(63) for b in BUCKETS}
        self.ratings = {b: [] for b in BUCKETS}

    def baseline(self, bucket: str) -> float:
        r = self.ratings[bucket]
        return float(np.mean(r)) if len(r) >= MIN_RATINGS else PRIOR_MEAN

    def update(self, bucket: str, mask: int, rating: float) -> float:
        d = LAMBDA * (rating - self.baseline(bucket))
        b = self.bias[bucket]
        b[mask - 1] += d
        for m in neighbours(mask):
            b[m - 1] += d / 4
        np.clip(b, -CLIP, CLIP, out=b)
        self.ratings[bucket].append(float(rating))
        return d


class Store:
    """SQLite-backed ratings + biases for the feedback service."""

    def __init__(self, path):
        self.path = str(path)
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self.prefs = Preferences()
        with self._conn() as c:
            c.execute("CREATE TABLE IF NOT EXISTS ratings (id INTEGER PRIMARY KEY, ts REAL DEFAULT "
                      "(strftime('%s','now')), bucket TEXT, mask INTEGER, rating REAL)")
            rows = c.execute("SELECT bucket, mask, rating FROM ratings ORDER BY id").fetchall()
        for b, m, r in rows:                 # replay history to rebuild the bias table
            if b in BUCKETS:
                self.prefs.update(b, int(m), float(r))

    def _conn(self):
        return sqlite3.connect(self.path)

    def add(self, bucket: str, mask: int, rating: float) -> dict:
        if bucket not in BUCKETS or not 1 <= mask <= 63 or not 1 <= rating <= 5:
            raise ValueError("invalid feedback")
        with self.lock:
            with self._conn() as c:
                c.execute("INSERT INTO ratings (bucket, mask, rating) VALUES (?, ?, ?)", (bucket, mask, rating))
            d = self.prefs.update(bucket, mask, rating)
        return {"delta": d, "n": len(self.prefs.ratings[bucket])}

    def get(self, bucket: str) -> dict:
        b = self.prefs.bias[bucket]
        return {"bucket": bucket, "bias": b.round(4).tolist(), "n": len(self.prefs.ratings[bucket]),
                "learned": int((np.abs(b) > 1e-9).sum())}
