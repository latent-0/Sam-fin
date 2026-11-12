"""Loan-readiness inference: score, conformal set, explanations, counterfactuals.

Uses the monotone LightGBM model trained by ``scripts/train_credit.py`` and
evaluated with the dependency-free ``treemodel`` reader.

* score shown is 100 (1 - PD); thresholds 0.10 (likely) and 0.22 (borderline)
* class-conditional conformal set: singleton = "confident", two-class = "uncertain"
* counterfactuals: a grid over four actionable levers (1,152 combinations),
  the three cheapest plans reaching PD <= 0.10 with distinct lever sets
* history-builder: PD after m more months of saving and recording
"""
from __future__ import annotations

import itertools
import json
from functools import lru_cache
from pathlib import Path

import numpy as np

from .features import FEATS
from .treemodel import TreeModel

ROOT = Path(__file__).resolve().parent.parent
ART = ROOT / "artifacts"

LABELS = {
    "age": "Age",
    "debt_ratio": "Existing debt payments / income",
    "utilization": "Credit utilisation",
    "d30": "30-59 day late payments",
    "d60": "60-89 day late payments",
    "d90": "90+ day late payments",
    "open_lines": "Open credit lines",
    "dependents": "Dependents",
    "income_cv": "Income variation (CV)",
    "semi_dev_ratio": "Downside income deviation",
    "history_months": "Months of income history",
    "buffer_months": "Cash buffer (months of expenses)",
    "emi_ratio": "New EMI / average income",
    "gig": "Variable-income worker",
    "expense_ratio": "Current expenses / average income",
}

# counterfactual grid (paper Sec. IV-C)
EMI_FACTORS = (1.0, 0.85, 0.7, 0.55, 0.4, 0.25)
BUFFER_ADD = (0, 1, 2, 3, 4, 6, 9, 12)
HISTORY_ADD = (0, 3, 6, 9, 12, 18)
UTIL_FACTORS = (1.0, 0.75, 0.5, 0.25)
COST = dict(emi_per_0p1=1.0, buffer_per_month=0.8, history_per_month=0.25, util_per_0p1=0.6)
HISTORY_HORIZON = 24


@lru_cache(maxsize=1)
def load():
    """Model + metadata. Uses native LightGBM when installed (fast exact SHAP), else the
    dependency-free reader (serverless deployment); both give identical numbers."""
    meta = json.loads((ART / "credit_meta.json").read_text())
    path = ART / "credit_model.txt"
    try:
        model = _NativeModel(path)
    except ImportError:
        model = TreeModel.load(path)
    return model, meta


class _NativeModel:
    def __init__(self, path):
        import lightgbm as lgb
        self.b = lgb.Booster(model_file=str(path))
        self.base_value = float(self.b.predict(np.zeros((1, len(FEATS))), pred_contrib=True)[0, -1])

    def predict(self, X):
        return self.b.predict(np.atleast_2d(np.asarray(X, float)))

    def contributions(self, x):
        return self.b.predict(np.asarray(x, float).reshape(1, -1), pred_contrib=True)[0, :-1]


def features(profile: dict, stab: dict, new_emi: float) -> dict:
    """Map a user profile + income statistics to the 15 model features."""
    mean = max(stab["mean"], 1e-9)
    expenses = float(profile["monthly_expenses"])
    savings = float(profile.get("liquid_savings", 0)) + float(profile.get("emergency_fund", 0))
    return dict(
        age=profile.get("age", 35),
        debt_ratio=float(profile.get("existing_emi", 0)) / mean,
        utilization=profile.get("utilization", 0.3),
        d30=profile.get("late_30", 0),
        d60=profile.get("late_60", 0),
        d90=profile.get("late_90", 0),
        open_lines=profile.get("open_lines", 2),
        dependents=profile.get("dependents", 0),
        income_cv=stab["cv"],
        semi_dev_ratio=stab["semi_dev_ratio"],
        history_months=stab["n"],
        buffer_months=savings / expenses if expenses > 0 else 0.0,
        emi_ratio=new_emi / mean,
        gig=int(stab["cv"] > 0.15),
        expense_ratio=expenses / mean,
    )


def _vec(f: dict) -> np.ndarray:
    return np.array([float(f[k]) for k in FEATS])


def decision(pd_: float, meta: dict) -> str:
    return "likely" if pd_ <= meta["approve_pd"] else "borderline" if pd_ <= meta["review_pd"] else "unlikely"


def conformal(pd_: float, qhat: dict) -> list:
    out = []
    if pd_ <= qhat["0"]:
        out.append(0)
    if 1 - pd_ <= qhat["1"]:
        out.append(1)
    return out


def counterfactuals(model: TreeModel, f: dict, target: float, k: int = 3) -> list:
    combos = list(itertools.product(EMI_FACTORS, BUFFER_ADD, HISTORY_ADD, UTIL_FACTORS))
    base = _vec(f)
    X = np.repeat(base[None, :], len(combos), axis=0)
    ix = {n: FEATS.index(n) for n in ("emi_ratio", "buffer_months", "history_months", "utilization")}
    c = np.array(combos)
    X[:, ix["emi_ratio"]] = f["emi_ratio"] * c[:, 0]
    X[:, ix["buffer_months"]] = f["buffer_months"] + c[:, 1]
    X[:, ix["history_months"]] = f["history_months"] + c[:, 2]
    X[:, ix["utilization"]] = f["utilization"] * c[:, 3]
    p = model.predict(X)
    cost = (COST["emi_per_0p1"] * (f["emi_ratio"] * (1 - c[:, 0])) / 0.1
            + COST["buffer_per_month"] * c[:, 1]
            + COST["history_per_month"] * c[:, 2]
            + COST["util_per_0p1"] * (f["utilization"] * (1 - c[:, 3])) / 0.1)
    out, seen = [], set()
    for i in np.argsort(cost, kind="stable"):
        if p[i] > target:
            continue
        ch = {}
        if c[i, 0] != 1:
            ch["emi_ratio"] = {"frm": round(float(f["emi_ratio"]), 3), "to": round(float(f["emi_ratio"] * c[i, 0]), 3)}
        if c[i, 1]:
            ch["buffer_months"] = {"frm": round(float(f["buffer_months"]), 2), "to": round(float(f["buffer_months"] + c[i, 1]), 2)}
        if c[i, 2]:
            ch["history_months"] = {"frm": float(f["history_months"]), "to": float(f["history_months"] + c[i, 2])}
        if c[i, 3] != 1:
            ch["utilization"] = {"frm": round(float(f["utilization"]), 3), "to": round(float(f["utilization"] * c[i, 3]), 3)}
        key = frozenset(ch)
        if not ch or key in seen:
            continue
        seen.add(key)
        out.append({"cost": round(float(cost[i]), 2), "new_pd": round(float(p[i]), 4), "changes": ch})
        if len(out) == k:
            break
    return out


def history_path(model: TreeModel, f: dict, monthly_saving: float, living_cost: float, target: float) -> dict:
    """PD after m = 0..24 more months of recording and saving."""
    m = np.arange(HISTORY_HORIZON + 1)
    X = np.repeat(_vec(f)[None, :], len(m), axis=0)
    X[:, FEATS.index("history_months")] = f["history_months"] + m
    X[:, FEATS.index("buffer_months")] = f["buffer_months"] + m * monthly_saving / max(living_cost, 1e-9)
    p = model.predict(X)
    hit = np.nonzero(p <= target)[0]
    return {"pd_path": np.round(p, 4).tolist(), "months_to_target": int(hit[0]) if len(hit) else None,
            "target_pd": target}


def assess(profile: dict, stab: dict, new_emi: float) -> dict:
    model, meta = load()
    f = features(profile, stab, new_emi)
    x = _vec(f)
    pd_ = float(model.predict(x[None, :])[0])
    contrib = model.contributions(x)
    cs = conformal(pd_, meta["qhat"])
    contributions = {k: float(v) for k, v in zip(FEATS, contrib)}
    order = np.argsort(-np.abs(contrib))[:8]
    factors = [dict(feature=FEATS[i], label=LABELS[FEATS[i]], value=float(x[i]), effect=round(float(contrib[i]), 4),
                    direction="raises risk" if contrib[i] > 0 else "lowers risk") for i in order]
    target = meta["approve_pd"]
    saving = max(stab["mean"] - float(profile["monthly_expenses"]), 0.0)
    return dict(
        pd=pd_,
        approval_score=round(100 * (1 - pd_), 1),
        decision=decision(pd_, meta),
        conformal_set=cs,
        certainty="confident" if len(cs) == 1 else "uncertain",
        base_value=model.base_value,
        contributions=contributions,
        labels=LABELS,
        counterfactuals=counterfactuals(model, f, target) if pd_ > target else [],
        history_path=history_path(model, f, saving, float(profile["monthly_expenses"]), target),
        thresholds={"approve": meta["approve_pd"], "review": meta["review_pd"]},
        factors=factors,
        features=f,
    )
