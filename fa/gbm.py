"""Monotone LightGBM loan-readiness model, conformal calibration and metrics.

Monotone constraints (paper Sec. IV-C): +1 for debt ratio, utilisation, the
three delinquency counts, CV, downside ratio, EMI share and expense share;
-1 for buffer and history length.
"""
from __future__ import annotations

import numpy as np

from .data import FEATS

PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=31, min_data_in_leaf=200,
              lambda_l2=5.0, bagging_fraction=0.8, bagging_freq=1, feature_fraction=0.9,
              verbose=-1, num_threads=0)
ROUNDS = 400
ALPHA = 0.10                 # conformal miscoverage
APPROVE_PD, REVIEW_PD = 0.10, 0.22

MONO_UP = {"debt_ratio", "utilization", "d30", "d60", "d90", "income_cv", "semi_dev_ratio", "emi_ratio",
           "expense_ratio"}
MONO_DOWN = {"buffer_months", "history_months"}


def monotone_vector(feats) -> list:
    return [1 if f in MONO_UP else -1 if f in MONO_DOWN else 0 for f in feats]


def train(X, y, feats=FEATS, monotone: bool = True, seed: int = 0, rounds: int = ROUNDS):
    import lightgbm as lgb
    p = dict(PARAMS, seed=seed, bagging_seed=seed, feature_fraction_seed=seed)
    if monotone:
        p["monotone_constraints"] = monotone_vector(feats)
        p["monotone_constraints_method"] = "advanced"
    ds = lgb.Dataset(np.asarray(X, float), label=np.asarray(y), feature_name=list(feats))
    return lgb.train(p, ds, num_boost_round=rounds)


# ---------------------------------------------------------------------------
# class-conditional (Mondrian) conformal prediction
# ---------------------------------------------------------------------------
def conformal_qhat(p_cal, y_cal, alpha: float = ALPHA) -> dict:
    """Per-class quantile of nonconformity s(x,1)=1-p, s(x,0)=p."""
    p_cal, y_cal = np.asarray(p_cal), np.asarray(y_cal)
    q = {}
    for c in (0, 1):
        s = np.sort(1 - p_cal[y_cal == 1] if c == 1 else p_cal[y_cal == 0])
        m = len(s)
        k = min(int(np.ceil((m + 1) * (1 - alpha))), m) - 1
        q[c] = float(s[k])
    return q


def conformal_sets(p, qhat: dict) -> np.ndarray:
    """Boolean array (n, 2): class c is in the set when s(x, c) <= qhat[c]."""
    p = np.asarray(p)
    return np.stack([p <= qhat[0], 1 - p <= qhat[1]], axis=1)


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------
def ece(p, y, bins: int = 10) -> float:
    p, y = np.asarray(p), np.asarray(y)
    edges = np.quantile(p, np.linspace(0, 1, bins + 1))
    idx = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, bins - 1)
    tot = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            tot += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(tot)


def metrics(p, y) -> dict:
    from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
    p, y = np.asarray(p), np.asarray(y)
    return dict(auc=float(roc_auc_score(y, p)), pr_auc=float(average_precision_score(y, p)),
                brier=float(brier_score_loss(y, p)), ece=ece(p, y))
