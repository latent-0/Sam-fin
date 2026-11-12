import json
from pathlib import Path

import numpy as np
import pytest

from fa import attention as A, dqn as Q, env as E, income as I, xai as X

ROOT = Path(__file__).resolve().parent.parent
PRESETS = json.loads((ROOT / "frontend" / "presets.json").read_text())
GIG = PRESETS["gig"]


def test_attention_gradients_match_finite_differences():
    rng = np.random.default_rng(0)
    B = 4
    Xb = rng.normal(size=(B, A.T_MAX, A.N_TOK))
    M = np.zeros((B, A.T_MAX))
    for b in range(B):
        M[b, :rng.integers(3, A.T_MAX + 1)] = 1
    Xb *= M[:, :, None]
    G = rng.normal(size=(B, A.N_GLOB))
    y = rng.random(B)
    p = A.init(3)
    prob, C = A.forward(p, Xb, M, G, cache=True)
    _, g = A.backward(p, prob, y, C)

    def loss():
        pr = np.clip(A.forward(p, Xb, M, G), 1e-7, 1 - 1e-7)
        return -np.mean(y * np.log(pr) + (1 - y) * np.log(1 - pr))

    for k in ("We", "Wq", "Wk", "Wv", "Wo", "cls", "Wg", "W1", "w2"):
        flat, gf = p[k].reshape(-1), g[k].reshape(-1)
        for i in rng.choice(flat.size, 3, replace=False):
            old = flat[i]
            flat[i] = old + 1e-6
            lp = loss()
            flat[i] = old - 1e-6
            lm = loss()
            flat[i] = old
            assert (lp - lm) / 2e-6 == pytest.approx(gf[i], rel=1e-4, abs=1e-9)


def test_attention_padding_is_ignored():
    p = A.init(0)
    tok, mask, g = A.encode([100, 120, 80], 70, 10, 50)
    a = A.forward(p, tok[None], mask[None], g[None])
    tok2 = tok.copy()
    tok2[3:] = 99.0                                   # garbage in padded slots
    assert A.forward(p, tok2[None], mask[None], g[None]) == pytest.approx(a)


@pytest.mark.skipif(not A.WEIGHTS.exists(), reason="run scripts/train_attention.py first")
def test_pattern_reader_output():
    r = A.read(GIG["income_history"], 21000, 9964, 25000)
    assert 0 <= r["ruin_probability"] <= 1
    assert len(r["focus"]) == 3 and sum(r["focus"]) == pytest.approx(1, abs=1e-3)
    assert len(r["head_maps"]) == A.HEADS and len(r["head_maps"][0]) == 3
    # more instalment, more risk
    lo = A.read(GIG["income_history"], 21000, 0, 25000)["ruin_probability"]
    assert r["ruin_probability"] >= lo


def test_shock_raises_risk_and_lowers_safe_emi():
    h = [52000, 48000, 61000, 70000, 66000, 40000, 35000, 38000, 55000, 72000, 68000, 45000]
    s = I.shock_test(h, 34000, 10000, 130000, drop=0.5, months=4, start=2)
    assert s["ruin"] >= s["ruin_base"]
    assert s["safe_emi"] <= s["safe_emi_base"]
    assert len(s["density"]) == I.CASH_BINS and len(s["density"][0]) == I.HORIZON
    assert len(s["paths"]) == I.SAMPLE_PATHS
    none = I.shock_test(h, 34000, 10000, 130000, drop=0.0, months=4, start=2)
    assert none["ruin"] == none["ruin_base"]


def test_q_input_gradient():
    net = Q.QNet(np.random.default_rng(0))
    x = np.random.default_rng(1).normal(size=E.OBS_DIM)
    g = net.input_grad(x, 5)
    for i in range(E.OBS_DIM):
        d = np.zeros(E.OBS_DIM)
        d[i] = 1e-6
        num = (net(x + d)[5] - net(x - d)[5]) / 2e-6
        assert num == pytest.approx(g[i], rel=1e-4, abs=1e-7)


def test_interaction_matrix_symmetric():
    st = I.stats(GIG["income_history"])
    P1, _ = E.profile_to_params(GIG, st["cv"], st["mean"])
    x = X.explain(P1, 13, draws=4)
    m = np.array(x["interaction_matrix"])
    assert m.shape == (6, 6) and np.allclose(m, m.T)
    assert len(x["coalition_values"]) == 64
