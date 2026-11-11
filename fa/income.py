"""Income-volatility mathematics.

Everything here works from a plain list of recorded monthly incomes and is
deterministic given a seed, so services and tests agree.

Pieces
------
* Income Stability Index (ISI): a bounded 0-100 score combining coefficient of
  variation, downside semi-deviation, drought fraction and trend (paper eq. 4).
* Shrinkage log-AR(1) income model: short histories (3-6 months) are pulled
  toward a prior so a gig worker with 3 months of data does not get an
  over-confident forecast (paper eq. 2-3). The AR(1) is the exact monthly
  discretisation of an Ornstein-Uhlenbeck process on log-income.
* Income-at-Risk (IaR) and CVaR of monthly income from Monte Carlo paths.
* Ruin-constrained safe EMI: the largest instalment for which the probability
  that cumulative cash goes negative within H months stays below alpha, found
  by bisection on common random numbers (paper eq. 5, Proposition 1).
* Buffer sizing: the smallest cash buffer that keeps ruin below alpha for a
  given EMI (the (1 - alpha) quantile of the running cash deficit).
* Calibrated internal risk target alpha~(n) by history length (Sec. IV-A-e).
"""
from __future__ import annotations

import numpy as np

PRIOR_LOG_SIGMA = 0.25      # prior monthly log-income volatility (sigma_0)
PRIOR_WEIGHT = 4.0          # pseudo-observations behind the prior (kappa)
PRIOR_PHI = 0.3             # prior lag-one coefficient (phi_0)
PHI_CLIP = (-0.5, 0.9)

N_PATHS = 4000
HORIZON = 12
SEED = 7
BISECTION_STEPS = 40

NOMINAL_ALPHA = 0.10
# Leave-one-process-out calibration (scripts/paper/exp_income.py): the largest
# internal level whose mean realised ruin does not exceed the nominal target.
ALPHA_GRID = (0.005, 0.01, 0.02, 0.03, 0.05, 0.07, 0.10)
CALIBRATED_ALPHA = ((24, 0.03), (12, 0.01), (0, 0.005))   # (min history, alpha~)


def _arr(x) -> np.ndarray:
    a = np.asarray(x, dtype=float)
    if a.ndim != 1 or a.size == 0:
        raise ValueError("income history must be a non-empty 1-D list")
    # Scale-safe floor: relative to the level of the series, never an absolute unit.
    floor = 1e-6 * max(float(np.abs(a).mean()), 1e-12)
    return np.clip(a, floor, None)


# ---------------------------------------------------------------------------
# descriptive statistics and stability index
# ---------------------------------------------------------------------------
def stats(history) -> dict:
    x = _arr(history)
    n = x.size
    mean, med = float(x.mean()), float(np.median(x))
    sd = float(x.std(ddof=1)) if n > 1 else 0.0
    cv = sd / mean if mean > 0 else 0.0
    semi = float(np.sqrt(np.mean(np.maximum(med - x, 0.0) ** 2))) / mean
    drought = x < 0.7 * mean
    longest, run = 0, 0
    for d in drought:
        run = run + 1 if d else 0
        longest = max(longest, run)
    trend = 0.0
    if n >= 6:
        trend = float(np.polyfit(np.arange(n), x, 1)[0]) / mean
    return dict(n=int(n), mean=mean, median=med, min=float(x.min()), max=float(x.max()), std=sd,
                cv=cv, semi_dev_ratio=semi, drought_frac=float(drought.mean()),
                longest_drought=int(longest), trend_per_month=trend)


def isi(st: dict) -> float:
    """Income Stability Index in [0, 100] (paper eq. 4)."""
    trend_pen = min(0.8 * max(-6.0 * st["trend_per_month"], 0.0), 0.5)
    z = st["cv"] + 1.2 * st["semi_dev_ratio"] + 0.6 * st["drought_frac"] + trend_pen
    return round(100.0 * float(np.exp(-z)), 1)


def isi_band(score: float) -> str:
    return "stable" if score >= 75 else "moderate" if score >= 50 else "volatile"


def stability(history) -> dict:
    st = stats(history)
    score = isi(st)
    return {"isi": score, "band": isi_band(score), **st}


# ---------------------------------------------------------------------------
# shrinkage log-AR(1) model and Monte Carlo paths
# ---------------------------------------------------------------------------
def fit(history) -> dict:
    """Shrinkage log-AR(1) fit (paper eq. 3).

    ``shrink_weight`` is the weight on the prior, kappa / (n + kappa).
    """
    x = _arr(history)
    y = np.log(x)
    n = y.size
    mu = float(y.mean())
    d = y - mu
    w_prior = PRIOR_WEIGHT / (n + PRIOR_WEIGHT)
    if n >= 4:
        den = float(np.sum(d[:-1] ** 2))
        phi_hat = float(np.clip(np.sum(d[1:] * d[:-1]) / den, *PHI_CLIP)) if den > 0 else PRIOR_PHI
        resid = d[1:] - phi_hat * d[:-1]
        s2 = float(np.var(resid, ddof=1))
    else:
        phi_hat = PRIOR_PHI
        s2 = float(np.var(y, ddof=1)) if n > 1 else PRIOR_LOG_SIGMA ** 2
    sigma = float(np.sqrt((1 - w_prior) * s2 + w_prior * PRIOR_LOG_SIGMA ** 2))
    phi = float((1 - w_prior) * phi_hat + w_prior * PRIOR_PHI)
    season = None
    if n >= 12:
        idx = np.arange(n) % 12
        s = np.array([d[idx == m].mean() for m in range(12)])
        season = (s - s.mean()).tolist()
    return dict(mu=mu, sigma=sigma, phi=phi, shrink_weight=w_prior, mean=float(x.mean()), n=int(n),
                season=season)


def simulate(model: dict, n_paths: int = N_PATHS, horizon: int = HORIZON, seed: int = SEED) -> np.ndarray:
    """Simulated monthly incomes, shape (n_paths, horizon).

    Paths start from the stationary distribution; a level correction makes the
    simulated mean equal the sample mean.
    """
    rng = np.random.default_rng(seed)
    sig, phi = model["sigma"], model["phi"]
    innov = sig * np.sqrt(max(1.0 - phi ** 2, 1e-9))
    z = rng.normal(0.0, sig, n_paths)
    eps = rng.standard_normal((n_paths, horizon))
    Z = np.empty((n_paths, horizon))
    for t in range(horizon):
        z = phi * z + innov * eps[:, t]
        Z[:, t] = z
    if model.get("season"):
        s = np.asarray(model["season"])
        Z = Z + s[(model["n"] + np.arange(horizon)) % 12]
    X = np.exp(Z)
    return X * (model["mean"] / X.mean())


def income_at_risk(paths: np.ndarray, q: float = 0.05) -> dict:
    v = paths.ravel()
    iar = float(np.quantile(v, q))
    return dict(iar=iar, cvar=float(v[v <= iar].mean()), p50=float(np.median(v)),
                p95=float(np.quantile(v, 0.95)))


def fan(paths: np.ndarray, qs=(0.05, 0.25, 0.5, 0.75, 0.95)) -> dict:
    return {str(q): np.round(np.quantile(paths, q, axis=0)).tolist() for q in qs}


# ---------------------------------------------------------------------------
# ruin, safe EMI, buffer
# ---------------------------------------------------------------------------
def ruin_probability(paths: np.ndarray, expenses: float, emi: float, buffer: float) -> float:
    """P(min_t [B + sum_{s<=t}(X_s - e - E)] < 0)  (paper eq. 5)."""
    cash = buffer + np.cumsum(paths - expenses - emi, axis=1)
    return float((cash.min(axis=1) < 0).mean())


def shortfall_probability(paths: np.ndarray, expenses: float, emi: float) -> float:
    """Share of simulated months in which income does not cover outgoings."""
    return float((paths < expenses + emi).mean())


def safe_emi(paths: np.ndarray, expenses: float, buffer: float, alpha: float,
             steps: int = BISECTION_STEPS) -> float:
    """sup{E >= 0 : ruin(E) <= alpha}, by bisection (ruin is monotone in E)."""
    if ruin_probability(paths, expenses, 0.0, buffer) > alpha:
        return 0.0
    lo, hi = 0.0, float(paths.mean()) + buffer / paths.shape[1] + 1.0
    while ruin_probability(paths, expenses, hi, buffer) <= alpha:
        hi *= 2
    for _ in range(steps):
        mid = 0.5 * (lo + hi)
        if ruin_probability(paths, expenses, mid, buffer) <= alpha:
            lo = mid
        else:
            hi = mid
    return lo


def required_buffer(paths: np.ndarray, expenses: float, emi: float, alpha: float) -> float:
    """Minimum B with ruin <= alpha: the (1 - alpha) quantile of max_t(-cumsum)^+."""
    deficit = np.maximum(-np.cumsum(paths - expenses - emi, axis=1), 0.0).max(axis=1)
    return float(np.quantile(deficit, 1.0 - alpha))


def calibrated_alpha(n: int) -> float:
    for min_n, a in CALIBRATED_ALPHA:
        if n >= min_n:
            return a
    return CALIBRATED_ALPHA[-1][1]


def analyse(history, expenses: float, new_emi: float, buffer: float, alpha: float = NOMINAL_ALPHA,
            calibrated: bool = True, seed: int = SEED) -> dict:
    """Full income analysis used by the income service."""
    st = stability(history)
    model = fit(history)
    paths = simulate(model, seed=seed)
    a_eff = calibrated_alpha(st["n"]) if calibrated else alpha
    safe = safe_emi(paths, expenses, buffer, a_eff)
    iar = income_at_risk(paths)
    need = max(expenses + new_emi - iar["iar"], 1e-9)
    return dict(
        stability=st,
        model={k: model[k] for k in ("mu", "sigma", "phi", "shrink_weight")},
        income_at_risk=iar,
        ruin_probability=ruin_probability(paths, expenses, new_emi, buffer),
        shortfall_probability=shortfall_probability(paths, expenses, new_emi),
        safe_emi_total=safe,
        required_buffer=required_buffer(paths, expenses, new_emi, a_eff),
        runway_months=float(buffer / need),
        fan=fan(paths),
        alpha=alpha,
        alpha_effective=a_eff,
        safe_new_emi=safe,
    )


def emi_amount(principal: float, annual_rate: float, months: int) -> float:
    """Standard amortisation formula (paper eq. 1)."""
    if principal <= 0 or months <= 0:
        return 0.0
    r = annual_rate / 1200.0
    if r == 0:
        return principal / months
    return principal * r * (1 + r) ** months / ((1 + r) ** months - 1)
