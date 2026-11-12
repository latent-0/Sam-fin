"""Five "true" monthly income processes used to stress-test the income models.

Each differs from the fitted log-AR(1): Gaussian AR(1) (well specified), AR(1)
with Student-t4 innovations, a two-state busy/lean regime-switching process, a
seasonal sinusoid with noise, and a process with random near-zero drought
months. Incomes have mean about 1, so results are in multiples of mean income.
"""
from __future__ import annotations

import numpy as np

PROCS = ("ar1_gauss", "ar1_t4", "regime", "seasonal", "droughts")


def draw_process(proc: str, rng):
    if proc in ("ar1_gauss", "ar1_t4"):
        return dict(phi=rng.uniform(0.1, 0.6), sig=rng.uniform(0.1, 0.5))
    if proc == "regime":
        return dict(stay=rng.uniform(0.80, 0.95), lean=rng.uniform(0.5, 0.75), sig=rng.uniform(0.05, 0.15))
    if proc == "seasonal":
        return dict(A=rng.uniform(0.1, 0.4), ph=rng.uniform(0, 2 * np.pi), sig=rng.uniform(0.05, 0.12))
    return dict(q=rng.uniform(0.05, 0.15), sig=rng.uniform(0.08, 0.2))


def simulate_true(proc: str, th: dict, months: int, paths: int, rng, state=None, t0: int = 0):
    """Returns incomes (paths, months) and the final state for continuation."""
    X = np.empty((paths, months))
    if proc in ("ar1_gauss", "ar1_t4"):
        phi, sig = th["phi"], th["sig"]
        z = state if state is not None else rng.normal(0, sig, paths)
        z = np.broadcast_to(z, (paths,)).copy()
        for t in range(months):
            if proc == "ar1_gauss":
                e = rng.standard_normal(paths)
            else:
                e = rng.standard_t(4, paths) / np.sqrt(2.0)        # unit variance
            z = phi * z + sig * np.sqrt(1 - phi ** 2) * e
            X[:, t] = np.exp(z - sig ** 2 / 2)
        return X, z
    if proc == "regime":
        busy = 1 + (1 - th["lean"]) * 0.5
        s = state if state is not None else (rng.random(paths) < 0.5)
        s = np.broadcast_to(s, (paths,)).copy()
        for t in range(months):
            flip = rng.random(paths) > th["stay"]
            s = np.where(flip, ~s, s)
            X[:, t] = np.where(s, busy, th["lean"]) * np.exp(th["sig"] * rng.standard_normal(paths))
        return X, s
    if proc == "seasonal":
        for t in range(months):
            X[:, t] = (1 + th["A"] * np.sin(2 * np.pi * (t0 + t) / 12 + th["ph"])) * \
                      np.exp(th["sig"] * rng.standard_normal(paths) - th["sig"] ** 2 / 2)
        return X, None
    for t in range(months):                                             # droughts
        base = np.exp(th["sig"] * rng.standard_normal(paths) - th["sig"] ** 2 / 2)
        dry = rng.random(paths) < th["q"]
        X[:, t] = np.where(dry, base * rng.uniform(0, 0.15, paths), base / (1 - 0.92 * th["q"]))
    return X, None
