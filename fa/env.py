"""Vectorised household-finance simulator used to train and evaluate the RL orchestrator.

All money amounts are expressed in multiples of the household's mean monthly
income, so one network handles any currency/scale; ``scale`` converts back.

Six agents (bit i of the action mask):
  0 Budget      trims discretionary spend by 12% (half the cut is a comfort cost)
  1 Debt        sends surplus to the debt, weight proportional to its interest rate
  2 Emergency   tops the reserve up to a volatility-scaled target clip(3 + 8c, 3, 8) months
  3 Goal        pays the amount needed to reach the goal by its deadline
  4 Risk        income-shock cover: 1.5% premium, pays half of any shortfall below 75%
                of the mean, lowers portfolio volatility by 30%
  5 Investment  invests surplus (penalised if the reserve is thin: suitability)

An action is a non-empty subset (63 of them). The reward is the monthly change
in net worth plus resilience and goal utility, minus shortfall, comfort,
suitability and per-call costs (paper eq. 8).
"""
from __future__ import annotations

import numpy as np

AGENTS = ["budget", "debt", "emergency", "goal", "risk", "investment"]
N_AGENTS = 6
N_ACTIONS = 63                      # masks 1..63
OBS_DIM = 14
CALL_COST = 0.02
PERSONAS = ["gig", "salaried_loan", "student", "near_retiree"]
PERSONA_LABELS = ["Gig worker", "Salaried + loans", "Student / graduate", "Near retiree"]
PERSONA_P = [0.30, 0.35, 0.20, 0.15]
HORIZON = 12
OBS_LABELS = ["Monthly surplus", "Income swing", "Downside swing", "Existing EMI", "Debt size", "Debt rate",
              "Reserve vs target", "Goal gap", "Months to goal", "Risk comfort", "Share invested",
              "Cash on hand", "Age", "Month of year"]

AR_PHI = 0.3
BUDGET_CUT = 0.12
POOL_SHARE = 0.9
RISK_PREMIUM = 0.015
RISK_FLOOR = 0.75
WEIGHTS = {"emergency": 1.5, "goal": 1.0}

# Table IX: uniform ranges per persona (income-normalised units)
PERSONA_RANGES = {
    #               cv          emi          fixed        var          debt     rate         reserve   invest   risk       age
    "gig":           ((0.12, 0.8), (0.00, 0.10), (0.35, 0.50), (0.10, 0.20), (0, 6),  (0.10, 0.30), (0, 1.5), (0, 1),  (0.2, 0.8), (21, 45)),
    "salaried_loan": ((0.04, 0.10), (0.12, 0.30), (0.30, 0.42), (0.08, 0.18), (8, 30), (0.08, 0.16), (0.5, 4), (0, 6),  (0.3, 0.8), (28, 50)),
    "student":       ((0.10, 0.25), (0.04, 0.12), (0.40, 0.55), (0.12, 0.22), (5, 18), (0.07, 0.12), (0, 1),   (0, 0.5), (0.3, 0.9), (20, 27)),
    "near_retiree":  ((0.01, 0.06), (0.00, 0.05), (0.30, 0.42), (0.12, 0.22), (0, 6),  (0.08, 0.14), (3, 10),  (5, 25), (0.1, 0.4), (58, 72)),
}


def mask_to_bits(mask: np.ndarray) -> np.ndarray:
    m = np.asarray(mask)
    return ((m[..., None] >> np.arange(N_AGENTS)) & 1).astype(float)


def action_to_mask(a):
    return np.asarray(a) + 1


def mask_names(mask: int) -> list:
    return [AGENTS[i] for i in range(N_AGENTS) if (int(mask) >> i) & 1]


def names_to_mask(names) -> int:
    return int(sum(1 << AGENTS.index(n) for n in names))


# ---------------------------------------------------------------------------
# population
# ---------------------------------------------------------------------------
def sample_households(n: int, rng: np.random.Generator, persona=None) -> dict:
    """Draw n households from the four personas (30/35/20/15)."""
    pers = np.full(n, persona) if persona is not None else rng.choice(4, n, p=PERSONA_P)
    P = {k: np.zeros(n) for k in ("c", "emi", "fixed", "var", "D", "rd", "R", "I", "rho", "age")}
    for k, name in enumerate(PERSONAS):
        idx = np.nonzero(pers == k)[0]
        m = len(idx)
        if not m:
            continue
        cv, emi, fixed, var, debt, rate, res, inv, risk, age = PERSONA_RANGES[name]
        if name == "gig":
            P["c"][idx] = np.clip(rng.lognormal(np.log(0.35), 0.35, m), *cv)
        else:
            P["c"][idx] = rng.uniform(*cv, m)
        P["emi"][idx] = rng.uniform(*emi, m)
        P["fixed"][idx] = rng.uniform(*fixed, m)
        P["var"][idx] = rng.uniform(*var, m)
        P["D"][idx] = rng.uniform(*debt, m)
        P["rd"][idx] = rng.uniform(*rate, m)
        exp = P["fixed"][idx] + P["var"][idx] + P["emi"][idx]
        P["R"][idx] = rng.uniform(*res, m) * exp
        P["I"][idx] = rng.uniform(*inv, m)
        P["rho"][idx] = rng.uniform(*risk, m)
        P["age"][idx] = rng.uniform(*age, m)
    P["persona"] = pers.astype(float)
    P["A"] = np.where(pers == 0, 0.15, 0.02)
    P["phase"] = rng.uniform(0, 2 * np.pi, n)
    P["t0"] = rng.integers(0, 12, n).astype(float)
    P["L"] = rng.uniform(0.1, 0.6, n)
    P["goal"] = rng.uniform(3, 12, n)
    P["Gs"] = P["goal"] * rng.uniform(0, 0.3, n)
    P["gm"] = rng.integers(6, 37, n).astype(float)
    return P


def profile_to_params(profile: dict, cv: float, mean_income: float) -> tuple[dict, float]:
    """One user profile -> simulator parameters (arrays of length 1) and the scale."""
    s = max(float(mean_income), 1e-9)
    emi = float(profile.get("existing_emi", 0))
    other = max(float(profile["monthly_expenses"]) - emi, 0.0)
    P = dict(
        c=cv, A=0.15 if cv > 0.15 else 0.02, phase=0.0, t0=0.0,
        fixed=0.65 * other / s, var=0.35 * other / s, emi=emi / s,
        D=float(profile.get("debt_balance", 0)) / s, rd=float(profile.get("debt_rate", 0)) / 100.0,
        R=float(profile.get("emergency_fund", 0)) / s, I=float(profile.get("investments", 0)) / s,
        L=float(profile.get("liquid_savings", 0)) / s, rho=float(profile.get("risk_tolerance", 0.5)),
        age=float(profile.get("age", 35)), goal=float(profile.get("goal_amount", 0)) / s,
        Gs=float(profile.get("goal_saved", 0)) / s, gm=float(profile.get("goal_months", 12)), persona=-1.0,
    )
    return {k: np.array([v], float) for k, v in P.items()}, s


def reserve_target(c) -> np.ndarray:
    return np.clip(3 + 8 * np.asarray(c), 3, 8)


# ---------------------------------------------------------------------------
# environment
# ---------------------------------------------------------------------------
class Env:
    def __init__(self, P: dict, seed: int = 0, stress: dict | None = None):
        self.P = {k: np.asarray(v, float).copy() for k, v in P.items()}
        st = stress or {}
        if st:
            self.P["c"] = np.clip(self.P["c"] * st.get("cv", 1.0), 0, 1.2)
            self.P["var"] = self.P["var"] * st.get("var", 1.0)
            self.P["R"] = self.P["R"] * st.get("savings", 1.0)
            self.P["L"] = self.P["L"] * st.get("savings", 1.0)
        self.n = len(self.P["c"])
        self.rng = np.random.default_rng(seed)
        self.reset()

    # -- state --------------------------------------------------------------
    def reset(self):
        P = self.P
        self.S = {k: P[k].copy() for k in ("L", "R", "D", "Gs", "I", "gm")}
        self.t = 0
        self.z = self.rng.normal(0, P["c"])
        return self.obs()

    @property
    def expenses(self):
        return self.P["fixed"] + self.P["var"] + self.P["emi"]

    def net_worth(self):
        S = self.S
        return S["L"] + S["R"] + S["Gs"] + S["I"] - S["D"]

    def coverage(self):
        return self.S["R"] / (reserve_target(self.P["c"]) * self.expenses)

    def obs(self) -> np.ndarray:
        P, S = self.P, self.S
        exp = self.expenses
        assets = S["L"] + S["R"] + S["Gs"] + S["I"]
        gap = np.where(P["goal"] > 0, np.maximum(P["goal"] - S["Gs"], 0) / np.maximum(P["goal"], 1e-9), 0.0)
        return np.stack([
            1 - exp,
            P["c"],
            0.7 * P["c"],
            P["emi"],
            S["D"] / 12,
            P["rd"] / 0.3,
            np.clip(self.coverage(), 0, 2),
            gap,
            S["gm"] / 36,
            P["rho"],
            S["I"] / np.maximum(assets, 1e-9),
            np.clip(S["L"] / exp, 0, 6) / 6,
            P["age"] / 70,
            ((P["t0"] + self.t) % 12) / 12,
        ], axis=1)

    # -- dynamics -------------------------------------------------------------
    def step(self, mask, noise=None):
        P, S = self.P, self.S
        n = self.n
        mask = np.broadcast_to(np.asarray(mask, int), (n,))
        b = mask_to_bits(mask)
        e1, e2 = noise if noise is not None else (self.rng.standard_normal(n), self.rng.standard_normal(n))
        c = P["c"]

        # income
        self.z = AR_PHI * self.z + c * np.sqrt(1 - AR_PHI ** 2) * e1
        season = 1 + P["A"] * np.sin(2 * np.pi * (P["t0"] + self.t) / 12 + P["phase"])
        X = np.exp(self.z - c ** 2 / 2) * season

        nw0, cov0 = self.net_worth(), np.minimum(self.coverage(), 1.0)

        # spending, cover, debt service
        cut = BUDGET_CUT * P["var"] * b[:, 0]
        comfort = 0.5 * cut
        premium = RISK_PREMIUM * X * b[:, 4]
        payout = 0.5 * np.maximum(RISK_FLOOR - X, 0) * b[:, 4]
        S["D"] = S["D"] * (1 + P["rd"] / 12)
        emi_paid = np.minimum(P["emi"], S["D"])
        S["D"] -= emi_paid
        cash = X + payout - P["fixed"] - (P["var"] - cut) - emi_paid - premium

        # investments drift
        mu = 0.04 + 0.08 * P["rho"]
        vol = (0.04 + 0.16 * P["rho"]) * np.where(b[:, 4] > 0, 0.7, 1.0)
        S["I"] = S["I"] * (1 + mu / 12 + vol / np.sqrt(12) * e2)

        # allocation: proportional to weights, capped by need, leftover redistributed
        exp = self.expenses
        tgt = reserve_target(c) * exp
        need = np.stack([
            S["D"] * b[:, 1],
            np.maximum(tgt - S["R"], 0) * b[:, 2],
            np.maximum(P["goal"] - S["Gs"], 0) / np.maximum(S["gm"], 1) * b[:, 3],
            np.where(b[:, 5] > 0, 1e9, 0.0),          # investment absorbs whatever is left
        ], axis=1)
        w = np.stack([np.clip(P["rd"] / 0.10, 0.2, 3.0), np.full(n, WEIGHTS["emergency"]),
                      np.full(n, WEIGHTS["goal"]), 0.5 + P["rho"]], axis=1) * (need > 1e-12)
        pool = POOL_SHARE * np.maximum(cash, 0)
        alloc = np.zeros_like(need)
        rem = pool.copy()
        for _ in range(4):
            open_ = (need - alloc > 1e-12) & (w > 0)
            ws = (w * open_).sum(axis=1)
            share = np.where(ws[:, None] > 0, rem[:, None] * w * open_ / np.maximum(ws[:, None], 1e-12), 0)
            give = np.minimum(share, need - alloc)
            alloc += give
            rem = rem - give.sum(axis=1)
            if np.all(rem < 1e-12):
                break
        a_d, a_e, a_g, a_i = alloc.T

        S["L"] = S["L"] + cash - alloc.sum(axis=1)
        S["D"] -= a_d
        S["R"] += a_e
        S["Gs"] += a_g
        S["I"] += a_i
        # a negative liquid balance draws on the reserve first, then on new debt
        short = np.maximum(-S["L"], 0)
        S["L"] = np.maximum(S["L"], 0)
        from_r = np.minimum(short, S["R"])
        S["R"] -= from_r
        S["D"] += short - from_r
        S["gm"] = np.maximum(S["gm"] - 1, 0)

        unsuitable = a_i * ((S["R"] < 3 * exp) & (c > 0.10))
        d_nw = self.net_worth() - nw0
        d_cov = np.minimum(self.coverage(), 1.0) - cov0
        k = b.sum(axis=1)
        r = (d_nw + 1.5 * (1 + 3 * c) * d_cov + 0.6 * a_g - 3 * short - 0.4 * comfort - 0.8 * unsuitable
             - CALL_COST * k + 3 * a_d * np.maximum(P["rd"] - 0.05, 0))
        self.t += 1
        info = dict(
            income=X, shortfall=short > 1e-12, short_amt=short, calls=k,
            alloc=dict(budget=cut, debt=a_d, emergency=a_e, goal=a_g, risk=premium, investment=a_i),
        )
        return r, self.obs(), info

    @property
    def done(self):
        return self.t >= HORIZON


# ---------------------------------------------------------------------------
# hand-written rule baseline (paper Sec. IV-E)
# ---------------------------------------------------------------------------
def rule_policy(env: Env) -> np.ndarray:
    P, S = env.P, env.S
    exp = env.expenses
    surplus = 1 - exp
    cov = env.coverage()
    gap = np.where(P["goal"] > 0, np.maximum(P["goal"] - S["Gs"], 0) / np.maximum(P["goal"], 1e-9), 0)
    bits = np.zeros((env.n, N_AGENTS), int)
    bits[:, 0] = surplus < 0.15
    bits[:, 1] = (P["rd"] > 0.10) & (S["D"] > 1e-9)
    bits[:, 2] = cov < 1
    bits[:, 3] = (gap > 0.05) & (S["gm"] < 36)
    bits[:, 4] = (P["c"] > 0.20) | ((P["age"] > 55) & (cov < 1))
    bits[:, 5] = (cov >= 1) & (P["rd"] < 0.10) & (surplus > 0.1)
    mask = (bits << np.arange(N_AGENTS)).sum(axis=1)
    return np.where(mask == 0, 1, mask)


POP_SIZE = 12_000
POP_SEED = 2024


def population(n: int = POP_SIZE, seed: int = POP_SEED) -> tuple[dict, dict]:
    """12,000 simulated households: first 80% train, last 20% held-out test."""
    P = sample_households(n, np.random.default_rng(seed))
    cut = int(0.8 * n)
    return {k: v[:cut] for k, v in P.items()}, {k: v[cut:] for k, v in P.items()}
