import json
from pathlib import Path

import numpy as np
import pytest

from fa import agents as A, dqn as Q, env as E, feedback as F, income as I, xai as X

ROOT = Path(__file__).resolve().parent.parent
PRESETS = json.loads((ROOT / "frontend" / "presets.json").read_text())


def gig_params():
    p = PRESETS["gig"]
    st = I.stats(p["income_history"])
    return E.profile_to_params(p, st["cv"], st["mean"])


def test_masks():
    assert E.mask_names(13) == ["budget", "emergency", "goal"]
    assert E.names_to_mask(["budget", "emergency", "goal"]) == 13
    assert E.mask_to_bits(np.array([63])).sum() == 6


def test_env_shapes_and_debt_mechanics():
    P1, scale = gig_params()
    e = E.Env(P1, seed=0)
    assert e.obs().shape == (1, E.OBS_DIM)
    e.z[:] = 0
    e.step(np.array([0]), noise=(np.zeros(1), np.zeros(1)))
    # 60,000 at 14% a year less a 3,000 instalment
    assert e.S["D"][0] * scale == pytest.approx(60000 * (1 + 0.14 / 12) - 3000)


def test_rule_policy_on_gig_preset():
    P1, _ = gig_params()
    assert E.mask_names(int(E.rule_policy(E.Env(P1))[0])) == ["debt", "emergency", "goal", "risk"]


def test_population_split():
    tr, te = E.population(n=1000)
    assert len(tr["c"]) == 800 and len(te["c"]) == 200


def test_rule_beats_random_and_all():
    _, te = E.population(n=1500)
    rng = np.random.default_rng(1)
    rule = Q.evaluate(te, E.rule_policy)["ret"].mean()
    rand = Q.evaluate(te, lambda env: rng.integers(1, 64, env.n))["ret"].mean()
    allsix = Q.evaluate(te, lambda env: np.full(env.n, 63))["ret"].mean()
    assert rule > rand and rule > allsix


def test_dqn_trains_and_saves(tmp_path):
    tr, _ = E.population(n=500)
    net, curve = Q.train(tr, ticks=300, n_env=16, batch=32)
    assert curve and np.isfinite(curve).all()
    net.save(tmp_path / "q.npz")
    q = Q.QNet.load(tmp_path / "q.npz")(np.zeros((2, E.OBS_DIM)))
    assert q.shape == (2, E.N_ACTIONS)


def test_shapley_efficiency_and_symmetry():
    # additive game: Shapley value equals each player's weight
    w = np.array([1.0, -2.0, 0.5, 0.0, 3.0, -1.0])
    v = np.array([sum(w[i] for i in range(6) if m >> i & 1) for m in range(64)])
    assert np.allclose(X.shapley(v), w)
    assert all(abs(x) < 1e-12 for x in X.interactions(v).values())


def test_explain_and_plan_on_preset():
    P1, scale = gig_params()
    q = np.linspace(0, 1, 63)
    x = X.explain(P1, 13, q, draws=8)
    assert abs(x["efficiency_gap"]) < 1e-9
    assert set(x["shapley"]) == set(E.AGENTS) and len(x["q_top"]) == 3
    plan = A.build_plan(P1, scale, 13)
    assert [a["agent"] for a in plan["actions"]] == ["budget", "emergency", "goal"]
    assert len(plan["projection"]["with_plan"]) == 12
    assert plan["reserve_target_months"] == pytest.approx(5.3)
    json.dumps(plan), json.dumps(x)


def test_feedback_update_rule(tmp_path):
    p = F.Preferences()
    d = p.update("variable", 13, 5)
    assert d == pytest.approx(0.15 * (5 - 3))
    assert p.bias["variable"][12] == pytest.approx(0.25)          # clipped
    assert p.bias["variable"][(13 ^ 1) - 1] == pytest.approx(0.075)  # neighbour gets delta / 4
    s = F.Store(tmp_path / "fb.db")
    s.add("steady", 3, 4)
    assert F.Store(tmp_path / "fb.db").get("steady")["n"] == 1      # replayed from SQLite
    with pytest.raises(ValueError):
        s.add("steady", 0, 4)
