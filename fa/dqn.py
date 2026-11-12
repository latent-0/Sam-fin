"""Double DQN orchestrator in plain NumPy (paper Sec. IV-E).

A 14-128-128-63 ReLU network learns Q(s, a) over the 63 non-empty agent
subsets. Training uses Adam, the double-estimator target, gamma = 0.9,
Huber-style clipping, batch 256, two updates per tick over 64 parallel
households, a 120,000-transition replay buffer, a target copy every 100 ticks
and a learning rate decayed linearly from 5e-4 to 20% of that.

Exploration is epsilon-greedy with epsilon falling from 1 to 0.05 over the first
60% of training. With rule guidance, an exploratory action is the rule
baseline's subset with up to two random bit flips (probability 0.75),
otherwise a uniform subset.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from . import env as E

LAYERS = (E.OBS_DIM, 128, 128, E.N_ACTIONS)


class QNet:
    def __init__(self, rng: np.random.Generator | None = None, params: dict | None = None):
        if params is not None:
            self.p = {k: np.asarray(v, float) for k, v in params.items()}
            return
        rng = rng or np.random.default_rng(0)
        self.p = {}
        for i, (a, b) in enumerate(zip(LAYERS[:-1], LAYERS[1:]), 1):
            self.p[f"W{i}"] = rng.normal(0, np.sqrt(2.0 / a), (a, b))
            self.p[f"b{i}"] = np.zeros(b)

    @classmethod
    def load(cls, path) -> "QNet":
        with np.load(path) as z:
            return cls(params={k: z[k] for k in z.files})

    def save(self, path) -> None:
        np.savez(path, **self.p)

    def copy(self) -> "QNet":
        return QNet(params={k: v.copy() for k, v in self.p.items()})

    def forward(self, x, cache: bool = False):
        p = self.p
        h1 = np.maximum(x @ p["W1"] + p["b1"], 0)
        h2 = np.maximum(h1 @ p["W2"] + p["b2"], 0)
        q = h2 @ p["W3"] + p["b3"]
        return (q, (x, h1, h2)) if cache else q

    __call__ = forward

    def input_grad(self, x, a: int) -> np.ndarray:
        """dQ(x, a) / dx for one state x; returns shape (OBS_DIM,)."""
        x = np.atleast_2d(np.asarray(x, float))
        p = self.p
        z1 = x @ p["W1"] + p["b1"]
        z2 = np.maximum(z1, 0) @ p["W2"] + p["b2"]
        d2 = p["W3"][:, a] * (z2[0] > 0)
        d1 = (d2 @ p["W2"].T) * (z1[0] > 0)
        return d1 @ p["W1"].T

    def grads(self, cache, dq) -> dict:
        x, h1, h2 = cache
        p = self.p
        g = {"W3": h2.T @ dq, "b3": dq.sum(0)}
        d2 = (dq @ p["W3"].T) * (h2 > 0)
        g["W2"], g["b2"] = h1.T @ d2, d2.sum(0)
        d1 = (d2 @ p["W2"].T) * (h1 > 0)
        g["W1"], g["b1"] = x.T @ d1, d1.sum(0)
        return g


class Adam:
    def __init__(self, params: dict, b1=0.9, b2=0.999, eps=1e-8):
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self.b1, self.b2, self.eps, self.t = b1, b2, eps, 0

    def step(self, params: dict, grads: dict, lr: float):
        self.t += 1
        for k, g in grads.items():
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * g
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * g * g
            mh = self.m[k] / (1 - self.b1 ** self.t)
            vh = self.v[k] / (1 - self.b2 ** self.t)
            params[k] -= lr * mh / (np.sqrt(vh) + self.eps)


def guided_explore(env: E.Env, rng: np.random.Generator, p_guide: float = 0.75) -> np.ndarray:
    """Rule subset with up to two random bit flips (else uniform), as action indices."""
    n = env.n
    rule = E.rule_policy(env)
    flips = rng.integers(0, 3, n)
    m = rule.copy()
    for k in range(2):
        do = flips > k
        m = np.where(do, m ^ (1 << rng.integers(0, E.N_AGENTS, n)), m)
    m = np.where(m == 0, rule, m)
    uni = rng.integers(1, 64, n)
    return np.where(rng.random(n) < p_guide, m, uni) - 1


def train(P: dict, ticks: int = 20000, seed: int = 0, guided: bool = True, n_env: int = 64,
          batch: int = 256, gamma: float = 0.9, lr0: float = 5e-4, buffer: int = 120_000,
          target_every: int = 100, updates_per_tick: int = 2, log_every: int = 0, weights: dict | None = None):
    """Train on the household population P (dict of arrays). Returns (net, curve)."""
    rng = np.random.default_rng(seed)
    net = QNet(rng)
    tgt = net.copy()
    opt = Adam(net.p)
    N = len(P["c"])

    B_s = np.zeros((buffer, E.OBS_DIM))
    B_a = np.zeros(buffer, int)
    B_r = np.zeros(buffer)
    B_s2 = np.zeros((buffer, E.OBS_DIM))
    B_d = np.zeros(buffer)
    ptr = size = 0

    def new_env():
        idx = rng.integers(0, N, n_env)
        return E.Env({k: v[idx] for k, v in P.items()}, seed=int(rng.integers(1 << 31)), weights=weights)

    env = new_env()
    s = env.obs()
    ep_ret = np.zeros(n_env)
    curve = []
    t0 = time.time()
    for tick in range(ticks):
        frac = tick / ticks
        eps = max(0.05, 1 - frac / 0.6 * 0.95)
        lr = lr0 * (1 - 0.8 * frac)
        greedy = net(s).argmax(1)
        explore = rng.random(n_env) < eps
        if explore.any():
            ex = guided_explore(env, rng) if guided else rng.integers(0, 63, n_env)
            greedy = np.where(explore, ex, greedy)
        a = greedy
        r, s2, _ = env.step(E.action_to_mask(a))
        ep_ret += r
        done = float(env.done)
        i = (ptr + np.arange(n_env)) % buffer
        B_s[i], B_a[i], B_r[i], B_s2[i], B_d[i] = s, a, r, s2, done
        ptr, size = (ptr + n_env) % buffer, min(size + n_env, buffer)
        if env.done:
            curve.append(float(ep_ret.mean()))
            ep_ret[:] = 0
            env = new_env()
            s = env.obs()
        else:
            s = s2

        if size >= batch * 4:
            for _ in range(updates_per_tick):
                j = rng.integers(0, size, batch)
                q2_online = net(B_s2[j])
                a2 = q2_online.argmax(1)
                q2 = tgt(B_s2[j])[np.arange(batch), a2]
                y = B_r[j] + gamma * (1 - B_d[j]) * q2
                q, cache = net.forward(B_s[j], cache=True)
                err = np.clip(q[np.arange(batch), B_a[j]] - y, -1, 1)       # Huber gradient
                dq = np.zeros_like(q)
                dq[np.arange(batch), B_a[j]] = err / batch
                opt.step(net.p, net.grads(cache, dq), lr)
        if tick % target_every == 0:
            tgt = net.copy()
        if log_every and tick % log_every == 0:
            print(f"tick {tick:6d}  eps {eps:.2f}  ret {np.mean(curve[-20:]) if curve else 0:.3f}  "
                  f"[{time.time() - t0:.0f}s]", flush=True)
    return net, curve


def policy(net: QNet, bias: np.ndarray | None = None):
    """Greedy policy over masks for use with ``evaluate``."""
    def act(env: E.Env) -> np.ndarray:
        q = net(env.obs())
        if bias is not None:
            q = q + bias
        return E.action_to_mask(q.argmax(1))
    return act


def evaluate(P: dict, act, seed: int = 0, stress: dict | None = None, weights: dict | None = None) -> dict:
    """Run one 12-month episode for every household with common random numbers.

    Also returns the 12-month sum of each reward term and plain outcomes that do not use the reward.
    """
    env = E.Env(P, seed=seed, stress=stress, weights=weights)
    nw0, cov0 = env.net_worth().copy(), np.minimum(env.coverage(), 1.0)
    debt0, goal0 = env.S["D"].copy(), env.S["Gs"].copy()
    terms = None
    rng = np.random.default_rng(10_000 + seed)
    ret = np.zeros(env.n)
    short = np.zeros(env.n)
    calls = np.zeros(env.n)
    usage = np.zeros((env.n, E.N_AGENTS))
    for _ in range(E.HORIZON):
        m = act(env)
        noise = (rng.standard_normal(env.n), rng.standard_normal(env.n))
        r, _, info = env.step(m, noise=noise)
        ret += r
        short += info["shortfall"]
        calls += info["calls"]
        usage += E.mask_to_bits(m)
        terms = {k: v.copy() for k, v in info["terms"].items()} if terms is None else \
            {k: terms[k] + v for k, v in info["terms"].items()}
    S, P = env.S, env.P
    goal_need = np.maximum(P["goal"] - goal0, 1e-9)
    outcomes = dict(
        net_worth_change=env.net_worth() - nw0,
        reserve_coverage_end=np.minimum(env.coverage(), 1.0),
        reserve_coverage_gain=np.minimum(env.coverage(), 1.0) - cov0,
        goal_progress=np.clip((S["Gs"] - goal0) / goal_need, 0, 1),
        debt_paid=debt0 - S["D"],
        shortfall_amount=-terms["shortfall"],
        any_shortfall=(short > 0).astype(float),
    )
    return dict(ret=ret, short=short / E.HORIZON, calls=calls, usage=usage / E.HORIZON, terms=terms,
                outcomes=outcomes)


def save(net: QNet, path: Path) -> None:
    net.save(path)
