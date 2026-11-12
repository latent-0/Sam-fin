"""RQ2: does the safe-EMI rule keep realised ruin near its target when the income
process is not the fitted one, and how much history is needed?

Households are drawn from five true income processes that differ from the fitted
log-AR(1). For history lengths n in {3, 6, 12, 24} and 100 households per cell,
a rule sees only the n observed months, the outgoing share and a buffer, and
chooses an EMI. 3,000 fresh future paths from the TRUE process (continuing from
the observed state) give the realised 12-month ruin probability.

The internal alpha~(n) is calibrated leave-one-process-out: chosen on four
processes, applied to the fifth.

    python scripts/paper/exp_income.py
Writes paper/results/income.json.
"""
import time

import numpy as np

from common import auc, save
from fa import income as I
from fa.processes import PROCS, draw_process, simulate_true

NS = (3, 6, 12, 24)
REPS = 100
FUTURE = 3000
H = 12
TARGET = I.NOMINAL_ALPHA
METHODS = ("ours", "mle_ar1", "bootstrap", "worst_month", "foir_35")


def realised_ruin(fut: np.ndarray, e: float, emi: float, b: float) -> float:
    return I.ruin_probability(fut, e, emi, b)


# ---------------------------------------------------------------------------
# rules
# ---------------------------------------------------------------------------
def emi_rules(obs: np.ndarray, e: float, b: float, alphas) -> dict:
    model = I.fit(obs)
    paths = I.simulate(model)
    out = {f"ours@{a}": I.safe_emi(paths, e, b, a) for a in alphas}
    raw = dict(model, shrink_weight=0.0)
    y = np.log(np.clip(obs, 1e-6 * obs.mean(), None))
    d = y - y.mean()
    if len(obs) >= 4:
        phi = float(np.clip(np.sum(d[1:] * d[:-1]) / max(np.sum(d[:-1] ** 2), 1e-12), -0.5, 0.9))
        sig = float(np.std(d[1:] - phi * d[:-1], ddof=1))
    else:
        phi, sig = 0.0, float(np.std(y, ddof=1))
    raw.update(phi=phi, sigma=max(sig, 1e-4))
    out["mle_ar1"] = I.safe_emi(I.simulate(raw), e, b, TARGET)
    rng = np.random.default_rng(1)
    boot = rng.choice(obs, (I.N_PATHS, I.HORIZON))
    out["bootstrap"] = I.safe_emi(boot, e, b, TARGET)
    out["worst_month"] = max(float(obs.min()) - e, 0.0)
    out["foir_35"] = 0.35 * float(obs.mean())
    return out


def run_cells(seed: int = 0):
    rng = np.random.default_rng(seed)
    rows = []
    for proc in PROCS:
        for n in NS:
            for _ in range(REPS):
                th = draw_process(proc, rng)
                t0 = int(rng.integers(0, 12))
                obs, st = simulate_true(proc, th, n, 1, rng, t0=t0)
                obs = obs[0]
                e = rng.uniform(0.45, 0.70)
                b = rng.uniform(0.5, 3.0) * e
                fut, _ = simulate_true(proc, th, H, FUTURE, rng, state=st, t0=t0 + n)
                emis = emi_rules(obs, e, b, I.ALPHA_GRID)
                rows.append(dict(proc=proc, n=n, emi={k: v for k, v in emis.items()},
                                 ruin={k: realised_ruin(fut, e, v, b) for k, v in emis.items()}))
    return rows


def agg(rows, key):
    r = np.array([x["ruin"][key] for x in rows])
    m = np.array([x["emi"][key] for x in rows])
    return dict(realised=float(r.mean()), ok=float((r <= TARGET).mean()), emi=float(m.mean()),
                severe=float((r > 2 * TARGET).mean()))


def choose_alpha(rows) -> float:
    ok = [a for a in I.ALPHA_GRID if np.mean([x["ruin"][f"ours@{a}"] for x in rows]) <= TARGET]
    return max(ok) if ok else min(I.ALPHA_GRID)


def ranking(seed: int = 1, n_house: int = 2000, n: int = 12):
    rng = np.random.default_rng(seed)
    isi, cv, mr, lab = [], [], [], []
    for _ in range(n_house):
        proc = PROCS[rng.integers(0, len(PROCS))]
        th = draw_process(proc, rng)
        obs, st = simulate_true(proc, th, n, 1, rng)
        obs = obs[0]
        e, b = rng.uniform(0.45, 0.70), rng.uniform(0.5, 3.0) * 0.6
        fut, _ = simulate_true(proc, th, H, 1000, rng, state=st, t0=n)
        s = I.stability(obs)
        isi.append(-s["isi"]), cv.append(s["cv"]), mr.append(-s["min"] / s["mean"])
        lab.append(realised_ruin(fut, e, 0.20 * obs.mean(), b) > TARGET)
    return dict(isi_low=auc(isi, lab), cv=auc(cv, lab), min_ratio_low=auc(mr, lab))


def main():
    t0 = time.time()
    rows = run_cells()
    print(f"cells [{time.time() - t0:.0f}s]", flush=True)
    keys = {"ours": f"ours@{TARGET}", "mle_ar1": "mle_ar1", "bootstrap": "bootstrap",
            "worst_month": "worst_month", "foir_35": "foir_35"}
    out_agg, by_proc = [], []
    for m, key in keys.items():
        for n in NS:
            out_agg.append(dict(method=m, n=n, **agg([r for r in rows if r["n"] == n], key)))
            for p in PROCS:
                by_proc.append(dict(method=m, proc=p, n=n, **agg([r for r in rows if r["n"] == n and r["proc"] == p], key)))

    # leave-one-process-out calibration of alpha~(n)
    calibrated = []
    for p in PROCS:
        for n in NS:
            a = choose_alpha([r for r in rows if r["n"] == n and r["proc"] != p])
            calibrated.append(dict(proc=p, n=n, chosen_alpha=a,
                                   **agg([r for r in rows if r["n"] == n and r["proc"] == p], f"ours@{a}")))
    cal_agg = []
    for n in NS:
        cs = [c for c in calibrated if c["n"] == n]
        cal_agg.append(dict(n=n, **{k: float(np.mean([c[k] for c in cs])) for k in ("realised", "ok", "emi", "severe")},
                            chosen=choose_alpha([r for r in rows if r["n"] == n])))
        # the deployed table (fa.income.CALIBRATED_ALPHA) is `chosen` on all five processes
    for m in ("ours_cal",):
        for c in cal_agg:
            out_agg.append(dict(method=m, n=c["n"], realised=c["realised"], ok=c["ok"], emi=c["emi"], severe=c["severe"]))

    out = dict(agg=out_agg, by_proc=by_proc, ranking_auc=ranking(), reps=REPS, alpha=TARGET, n_households=2000,
               calibrated=calibrated, calibrated_agg=cal_agg)
    save("income", out)
    print(f"done [{time.time() - t0:.0f}s]")


if __name__ == "__main__":
    main()
