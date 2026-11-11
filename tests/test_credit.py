import json
from pathlib import Path

import numpy as np
import pytest

from fa import credit as C, gbm as G, income as I
from fa.data import FEATS
from fa.treemodel import TreeModel

ROOT = Path(__file__).resolve().parent.parent
MODEL = ROOT / "artifacts" / "credit_model.txt"
PRESETS = json.loads((ROOT / "frontend" / "presets.json").read_text())

needs_model = pytest.mark.skipif(not MODEL.exists(), reason="run scripts/train_credit.py first")


@needs_model
def test_treemodel_matches_lightgbm():
    lgb = pytest.importorskip("lightgbm")
    b = lgb.Booster(model_file=str(MODEL))
    m = TreeModel.load(MODEL)
    rng = np.random.default_rng(0)
    X = np.abs(rng.normal(1, 1, (300, len(FEATS))))
    assert np.allclose(b.predict(X), m.predict(X), atol=1e-12)
    ref = b.predict(X[:3], pred_contrib=True)
    mine = np.array([m.contributions(x) for x in X[:3]])
    assert np.allclose(ref[:, :-1], mine, atol=1e-10)
    assert np.allclose(ref[:, -1], m.base_value, atol=1e-10)


@needs_model
def test_shap_additivity():
    m = TreeModel.load(MODEL)
    x = np.abs(np.random.default_rng(1).normal(1, 1, len(FEATS)))
    assert m.contributions(x).sum() + m.base_value == pytest.approx(m.predict_raw(x)[0], abs=1e-9)


@needs_model
def test_monotone_in_emi_and_buffer():
    p = dict(PRESETS["gig"])
    st = I.stats(p["income_history"])
    pds = [C.assess(p, st, e)["pd"] for e in (0, 3000, 6000, 9000)]
    assert all(a <= b + 1e-12 for a, b in zip(pds, pds[1:]))
    p2 = dict(p, emergency_fund=p["emergency_fund"] + 50000)
    assert C.assess(p2, st, 6000)["pd"] <= pds[2] + 1e-12


@needs_model
def test_counterfactuals_reach_target_and_are_distinct():
    p = PRESETS["gig"]
    st = I.stats(p["income_history"])
    r = C.assess(p, st, 9964.29)
    assert r["decision"] == "unlikely"
    cfs = r["counterfactuals"]
    assert 1 <= len(cfs) <= 3
    assert all(cf["new_pd"] <= 0.10 for cf in cfs)
    assert len({frozenset(cf["changes"]) for cf in cfs}) == len(cfs)
    assert [cf["cost"] for cf in cfs] == sorted(cf["cost"] for cf in cfs)
    json.dumps(r)                                             # fully serialisable


def test_conformal_qhat_and_coverage():
    rng = np.random.default_rng(0)
    y = rng.random(20000) < 0.15
    p = np.clip(np.where(y, rng.beta(4, 3, y.size), rng.beta(1.5, 8, y.size)), 1e-6, 1 - 1e-6)
    q = G.conformal_qhat(p[:10000], y[:10000])
    sets = G.conformal_sets(p[10000:], q)
    cov = sets[np.arange(10000), y[10000:].astype(int)].mean()
    assert 0.88 <= cov <= 0.92


def test_monotone_vector():
    v = dict(zip(FEATS, G.monotone_vector(FEATS)))
    assert v["emi_ratio"] == 1 and v["buffer_months"] == -1 and v["history_months"] == -1 and v["age"] == 0
