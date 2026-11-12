"""Agent-Shapley explanations over the coalition lattice (paper Sec. IV-F).

For a household, the characteristic function of a coalition S of agents is the
expected total reward over six months when exactly the agents in S act every
month, with v(empty) the no-agent baseline. Because the orchestrator chooses
among all coalitions, all 64 values are simulated with common random numbers
and the Shapley values are exact up to the Monte Carlo error of v.

Pairwise Shapley interaction indices follow Grabisch & Roubens; positive means
synergy. The orchestrator view (Q-margins) explains the policy, while Shapley
values explain the simulated outcome; where they disagree the interface shows it.
"""
from __future__ import annotations

from itertools import combinations
from math import factorial

import numpy as np

from . import env as E

DRAWS = 32
HORIZON = 6
NAMES = {"budget": "Budget", "debt": "Debt", "emergency": "Emergency", "goal": "Goal", "risk": "Risk",
         "investment": "Investment"}


def coalition_values(P1: dict, draws: int = DRAWS, horizon: int = HORIZON, seed: int = 11) -> np.ndarray:
    """v(S) for all 64 masks, one vectorised run of 64 x draws copies."""
    masks = np.repeat(np.arange(64), draws)
    sub = {k: np.repeat(np.asarray(v, float)[:1], len(masks)) for k, v in P1.items()}
    env = E.Env(sub, seed=seed)
    rng = np.random.default_rng(seed)
    env.z = np.tile(rng.normal(0, float(P1["c"][0]), draws), 64)
    tot = np.zeros(len(masks))
    for _ in range(horizon):
        noise = (np.tile(rng.standard_normal(draws), 64), np.tile(rng.standard_normal(draws), 64))
        r, _, _ = env.step(masks, noise=noise)
        tot += r
    return tot.reshape(64, draws).mean(axis=1)


def shapley(v: np.ndarray, n: int = E.N_AGENTS) -> np.ndarray:
    phi = np.zeros(n)
    for i in range(n):
        for S in range(64):
            if S >> i & 1:
                continue
            k = bin(S).count("1")
            phi[i] += factorial(k) * factorial(n - k - 1) / factorial(n) * (v[S | 1 << i] - v[S])
    return phi


def interactions(v: np.ndarray, n: int = E.N_AGENTS) -> dict:
    out = {}
    for i, j in combinations(range(n), 2):
        tot = 0.0
        for S in range(64):
            if S >> i & 1 or S >> j & 1:
                continue
            k = bin(S).count("1")
            w = factorial(k) * factorial(n - k - 2) / factorial(n - 1)
            tot += w * (v[S | 1 << i | 1 << j] - v[S | 1 << i] - v[S | 1 << j] + v[S])
        out[(i, j)] = tot
    return out


def _matrix(inter: dict) -> list:
    m = np.zeros((E.N_AGENTS, E.N_AGENTS))
    for (i, j), x in inter.items():
        m[i, j] = m[j, i] = x
    return m.round(4).tolist()


def q_margins(q: np.ndarray) -> dict:
    """Mean Q over subsets containing each agent minus mean over subsets without it."""
    bits = E.mask_to_bits(np.arange(1, 64))
    return {a: round(float(q[bits[:, i] == 1].mean() - q[bits[:, i] == 0].mean()), 4)
            for i, a in enumerate(E.AGENTS)}


def q_top(q: np.ndarray, k: int = 3) -> list:
    return [dict(agents=E.mask_names(a + 1), q=round(float(q[a]), 3)) for a in np.argsort(-q)[:k]]


def narrative(phi: dict, chosen: list) -> list:
    out = []
    for a, val in sorted(phi.items(), key=lambda kv: -kv[1]):
        nm = NAMES[a]
        if val > 0 and a in chosen:
            out.append(f"{nm} adds about {val:.2f} units of value for this household and was selected.")
        elif val > 0:
            out.append(f"{nm} adds about {val:.2f} units of value for this household but was not selected "
                       f"(cost or conflicts outweigh it).")
        elif a in chosen:
            out.append(f"{nm} shows little or negative simulated value here ({val:.2f}) but the learned policy "
                       f"kept it.")
        else:
            out.append(f"{nm} would reduce value here ({val:.2f}), so it was skipped.")
    return out


def explain(P1: dict, mask: int, q: np.ndarray | None = None, draws: int = DRAWS, horizon: int = HORIZON) -> dict:
    v = coalition_values(P1, draws=draws, horizon=horizon)
    phi = shapley(v)
    inter = interactions(v)
    top = sorted(inter.items(), key=lambda kv: -abs(kv[1]))[:4]
    best = int(np.argmax(v[1:]) + 1)
    chosen = E.mask_names(mask)
    sh = {a: round(float(phi[i]), 4) for i, a in enumerate(E.AGENTS)}
    out = dict(
        baseline_value=float(v[0]),
        chosen_value=float(v[mask]),
        best_one_step_mask=best,
        best_one_step_agents=E.mask_names(best),
        best_one_step_value=float(v[best]),
        efficiency_gap=float(phi.sum() - (v[63] - v[0])),
        shapley=sh,
        interactions=[dict(a=E.AGENTS[i], b=E.AGENTS[j], value=round(float(x), 4),
                           kind="synergy" if x > 0 else "overlap") for (i, j), x in top],
        narrative=narrative(sh, chosen),
        interaction_matrix=_matrix(inter),
        coalition_values=v.round(4).tolist(),
    )
    if q is not None:
        q = np.asarray(q, float)
        out["q_margin"] = q_margins(q)
        out["q_top"] = q_top(q)
    return out
