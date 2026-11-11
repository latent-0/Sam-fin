"""Public credit datasets, schema harmonisation and volatility augmentation.

Sources (fetched from OpenML once and cached under ``data/raw``):

* Give Me Some Credit       150,000 rows   OpenML 46929
* UCI Default of Credit Card 30,000 rows   OpenML 42477
* Statlog German Credit       1,000 rows   OpenML 31

Each source is mapped to eight shared core features. Where a source lacks a
feature, values are bootstrapped from the sources that have it (so imputed
columns carry no information about that source's own label).

None of the datasets records income variation, so six variables are added by
simulation (paper Sec. IV-B, eq. 6-7) and a stress-augmented label is built
from a 12-month ruin simulation.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"

OPENML_IDS = {"gmsc": 46929, "taiwan": 42477, "german": 31}
CORE = ["age", "debt_ratio", "utilization", "d30", "d60", "d90", "open_lines", "dependents"]
VOL = ["income_cv", "semi_dev_ratio", "history_months", "buffer_months", "emi_ratio", "gig", "expense_ratio"]
FEATS = CORE + VOL

PASS_THROUGH = 0.6          # P(default | ruin) for borrowers who were not already defaulters


# ---------------------------------------------------------------------------
# download + cache
# ---------------------------------------------------------------------------
def _fetch(name: str) -> pd.DataFrame:
    RAW.mkdir(parents=True, exist_ok=True)
    path = RAW / f"{name}.csv.gz"
    if path.exists():
        return pd.read_csv(path)
    from sklearn.datasets import fetch_openml
    df = fetch_openml(data_id=OPENML_IDS[name], as_frame=True, parser="auto",
                      data_home=str(RAW / "openml_cache")).frame
    df.to_csv(path, index=False)
    return df


# ---------------------------------------------------------------------------
# harmonise each source to CORE + label y (NaN where the source lacks a feature)
# ---------------------------------------------------------------------------
def _gmsc(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame({
        "age": df["age"].astype(float),
        "debt_ratio": df["DebtRatio"].astype(float),
        "utilization": df["RevolvingUtilizationOfUnsecuredLines"].astype(float),
        "d30": df["NumberOfTime30-59DaysPastDueNotWorse"].astype(float),
        "d60": df["NumberOfTime60-89DaysPastDueNotWorse"].astype(float),
        "d90": df["NumberOfTimes90DaysLate"].astype(float),
        "open_lines": df["NumberOfOpenCreditLinesAndLoans"].astype(float),
        "dependents": df["NumberOfDependents"].astype(float),
    })
    # codes 96/98 are sentinel values in the delinquency counts
    for c in ("d30", "d60", "d90"):
        out.loc[out[c] >= 90, c] = np.nan
    out["utilization"] = out["utilization"].clip(0, 2)
    out["debt_ratio"] = out["debt_ratio"].clip(0, 5)
    out["y"] = (df["FinancialDistressNextTwoYears"].astype(str).str.lower().isin(["yes", "1", "true"])).astype(int)
    return out


def _taiwan(df: pd.DataFrame) -> pd.DataFrame:
    limit = df["x1"].astype(float).clip(lower=1)
    pay = df[[f"x{i}" for i in range(6, 12)]].astype(float).to_numpy()       # repayment status, 6 months
    bills = df[[f"x{i}" for i in range(12, 18)]].astype(float).to_numpy()
    out = pd.DataFrame({
        "age": df["x5"].astype(float),
        "debt_ratio": np.nan,
        "utilization": np.clip(bills.mean(axis=1) / limit, 0, 2),
        "d30": (pay == 1).sum(axis=1).astype(float),
        "d60": (pay == 2).sum(axis=1).astype(float),
        "d90": (pay >= 3).sum(axis=1).astype(float),
        "open_lines": np.nan,
        "dependents": np.nan,
    })
    out["y"] = df["y"].astype(int)
    return out


def _german(df: pd.DataFrame) -> pd.DataFrame:
    hist = df["credit_history"].astype(str)
    out = pd.DataFrame({
        "age": df["age"].astype(float),
        # instalment as % of disposable income, 1..4 -> a coarse debt-ratio proxy
        "debt_ratio": df["installment_commitment"].astype(float) / 8.0,
        "utilization": np.nan,
        "d30": hist.str.contains("delayed").astype(float),
        "d60": hist.str.contains("critical").astype(float),
        "d90": 0.0,
        "open_lines": df["existing_credits"].astype(float),
        "dependents": (df["num_dependents"].astype(float) - 1).clip(lower=0),
    })
    out["y"] = (df["class"].astype(str) == "bad").astype(int)
    return out


def load_sources() -> pd.DataFrame:
    parts = []
    for name, fn in (("gmsc", _gmsc), ("taiwan", _taiwan), ("german", _german)):
        d = fn(_fetch(name))
        d["source"] = name
        parts.append(d)
    return pd.concat(parts, ignore_index=True)


def bootstrap_impute(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Fill missing core values by resampling the observed values of that column."""
    df = df.copy()
    for c in CORE:
        miss = df[c].isna().to_numpy()
        if miss.any():
            pool = df.loc[~miss, c].to_numpy()
            df.loc[miss, c] = rng.choice(pool, miss.sum())
    return df


# ---------------------------------------------------------------------------
# volatility augmentation (paper eq. 6-7)
# ---------------------------------------------------------------------------
def augment(df: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = bootstrap_impute(df, rng)
    m = len(df)
    gig = rng.random(m) < 0.3
    c = np.where(gig, rng.lognormal(np.log(0.35), 0.35, m), rng.lognormal(np.log(0.06), 0.5, m))
    c = np.clip(c, 0.01, 0.9)
    n = rng.integers(3, 25, m)
    buf = np.where(gig, rng.lognormal(np.log(1.2), 0.7, m), rng.lognormal(np.log(2.5), 0.6, m))
    eta = rng.uniform(0.02, 0.45, m)
    rho = np.clip(0.25 + 0.55 * np.minimum(df["debt_ratio"].to_numpy(), 1.0)
                  + 0.02 * df["dependents"].to_numpy() + rng.normal(0, 0.06, m), 0.2, 0.95)
    # the model sees sample variation over n months, not the truth
    c_hat = c * np.sqrt(rng.chisquare(n - 1) / (n - 1))
    semi = c_hat * rng.uniform(0.5, 0.9, m)

    ruin = simulate_ruin(c, buf * rho, rho + eta, rng)
    y_aug = (df["y"].to_numpy() == 1) | (ruin & (rng.random(m) < PASS_THROUGH))

    out = df.copy()
    out["income_cv"] = c_hat
    out["semi_dev_ratio"] = semi
    out["history_months"] = n.astype(float)
    out["buffer_months"] = buf
    out["emi_ratio"] = eta
    out["gig"] = gig.astype(float)
    out["expense_ratio"] = rho
    out["true_cv"] = c
    out["ruin"] = ruin.astype(int)
    out["y_aug"] = y_aug.astype(int)
    return out


def simulate_ruin(c, start_cash, outgoings, rng, months: int = 12, phi: float = 0.3) -> np.ndarray:
    """12 months of AR(1) log-income with mean 1; ruin if cumulative cash < 0."""
    m = len(c)
    z = rng.normal(0, c)
    cash = np.asarray(start_cash, float).copy()
    ruined = np.zeros(m, bool)
    innov = c * np.sqrt(1 - phi ** 2)
    for _ in range(months):
        z = phi * z + innov * rng.standard_normal(m)
        cash += np.exp(z - c ** 2 / 2) - outgoings
        ruined |= cash < 0
    return ruined


def split(df: pd.DataFrame, seed: int = 0, fracs=(0.70, 0.15, 0.15)):
    """Stratified (by source and label) train / calibration / test split."""
    rng = np.random.default_rng(seed)
    part = np.empty(len(df), dtype=np.int8)
    for _, idx in df.groupby(["source", "y_aug"]).indices.items():
        idx = rng.permutation(idx)
        a = int(round(fracs[0] * len(idx)))
        b = a + int(round(fracs[1] * len(idx)))
        part[idx[:a]], part[idx[a:b]], part[idx[b:]] = 0, 1, 2
    return df[part == 0], df[part == 1], df[part == 2]
