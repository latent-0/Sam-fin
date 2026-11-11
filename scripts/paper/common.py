"""Shared helpers for the paper experiments: paths, statistics, JSON output."""
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
# RESULTS_DIR lets a trial run write somewhere other than the published results
RESULTS = Path(os.getenv("RESULTS_DIR", ROOT / "paper" / "results"))
FIGURES = Path(os.getenv("FIGURES_DIR", ROOT / "paper" / "figures"))
RESULTS.mkdir(parents=True, exist_ok=True)


def save(name: str, obj) -> Path:
    path = RESULTS / f"{name}.json"
    path.write_text(json.dumps(obj, indent=1, default=_default))
    print(f"wrote {path}")
    return path


def load(name: str):
    return json.loads((RESULTS / f"{name}.json").read_text())


def _default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def cvar(x, q: float = 0.05) -> float:
    x = np.sort(np.asarray(x))
    return float(x[: max(1, int(np.ceil(q * len(x))))].mean())


def bootstrap_ci(x, stat=np.mean, reps: int = 4000, seed: int = 0, level: float = 0.95) -> list:
    """Percentile bootstrap over households."""
    x = np.asarray(x)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), (reps, len(x)))
    if stat is np.mean:
        s = x[idx].mean(axis=1)
    else:
        s = np.array([stat(x[i]) for i in idx])
    a = (1 - level) / 2
    return [float(np.quantile(s, a)), float(np.quantile(s, 1 - a))]


def wilcoxon_p(d) -> float:
    from scipy.stats import wilcoxon
    d = np.asarray(d)
    if np.allclose(d, 0):
        return 1.0
    return float(wilcoxon(d).pvalue)


def paired(a, b, seed: int = 0) -> dict:
    d = np.asarray(a) - np.asarray(b)
    return dict(mean_diff=float(d.mean()), ci=bootstrap_ci(d, seed=seed), wilcoxon_p=wilcoxon_p(d))


def mean_sd(xs) -> list:
    xs = np.asarray(xs, float)
    return [float(xs.mean()), float(xs.std(ddof=1)) if len(xs) > 1 else 0.0]


def auc(score, label) -> float:
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(np.asarray(label), np.asarray(score)))
