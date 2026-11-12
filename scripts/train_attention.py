"""Train the income pattern reader (fa/attention.py) and compare it with the AR(1) estimate.

Data: histories of 3 to 24 months drawn from the five income processes in
fa/processes.py, with random spending share, instalment and buffer. The label is
the true 12-month ruin probability from 400 future paths of the same process.

    python scripts/train_attention.py
Writes artifacts/attention.npz and paper/results/attention.json.
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from fa import attention as A, income as I  # noqa: E402
from fa.processes import PROCS, draw_process, simulate_true  # noqa: E402

N_TRAIN, N_TEST, FUTURE = 40_000, 5_000, 400
RESULTS = Path(os.getenv("RESULTS_DIR", ROOT / "paper" / "results"))


def make(n_samples: int, seed: int):
    rng = np.random.default_rng(seed)
    X = np.zeros((n_samples, A.T_MAX, A.N_TOK))
    M = np.zeros((n_samples, A.T_MAX))
    G = np.zeros((n_samples, A.N_GLOB))
    y = np.zeros(n_samples)
    meta = []
    for i in range(n_samples):
        proc = PROCS[rng.integers(len(PROCS))]
        th = draw_process(proc, rng)
        n = int(rng.integers(3, A.T_MAX + 1))
        t0 = int(rng.integers(0, 12))
        obs, st = simulate_true(proc, th, n, 1, rng, t0=t0)
        obs = obs[0]
        e, emi, b = rng.uniform(0.40, 0.85), rng.uniform(0.0, 0.45), rng.uniform(0.0, 2.5)
        fut, _ = simulate_true(proc, th, 12, FUTURE, rng, state=st, t0=t0 + n)
        y[i] = I.ruin_probability(fut, e, emi, b)
        X[i], M[i], g = A.encode(obs, e, emi, b)
        G[i] = A.clip_globals(g)
        meta.append((proc, n, obs, e, emi, b))
    return X, M, G, y, meta


def auc(score, label):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(label, score)) if 0 < label.mean() < 1 else None


def main():
    t0 = time.time()
    Xtr, Mtr, Gtr, ytr, mtr = make(N_TRAIN, 0)
    Xte, Mte, Gte, yte, meta = make(N_TEST, 1)
    print(f"data [{time.time() - t0:.0f}s]", flush=True)
    p = A.fit(Xtr, Mtr, Gtr, ytr, log=lambda ep, l: print(f"epoch {ep:2d} loss {l:.4f}", flush=True)
              if ep % 5 == 0 else None)
    out = ROOT / "artifacts" / "attention.npz"
    np.savez(out, **p)
    print(f"saved {out} [{time.time() - t0:.0f}s]", flush=True)

    p_att = A.forward(p, Xte, Mte, Gte)
    p_ar = np.array([I.ruin_probability(I.simulate(I.fit(obs), n_paths=1000, seed=i), e, emi, b)
                     for i, (_, _, obs, e, emi, b) in enumerate(meta)])
    procs = np.array([m[0] for m in meta])
    ns = np.array([m[1] for m in meta])

    def scores(mask):
        lab = yte[mask] > I.NOMINAL_ALPHA
        return dict(n=int(mask.sum()),
                    brier_attention=float(np.mean((p_att[mask] - yte[mask]) ** 2)),
                    brier_ar1=float(np.mean((p_ar[mask] - yte[mask]) ** 2)),
                    mae_attention=float(np.mean(np.abs(p_att[mask] - yte[mask]))),
                    mae_ar1=float(np.mean(np.abs(p_ar[mask] - yte[mask]))),
                    auc_attention=auc(p_att[mask], lab), auc_ar1=auc(p_ar[mask], lab))

    res = dict(
        overall=scores(np.ones(len(yte), bool)),
        by_process={pr: scores(procs == pr) for pr in PROCS},
        by_history={str(k): scores((ns >= lo) & (ns <= hi)) for k, (lo, hi) in
                    {"3-5": (3, 5), "6-11": (6, 11), "12-24": (12, 24)}.items()},
        n_train=N_TRAIN, n_test=N_TEST, future_paths=FUTURE,
        architecture=dict(heads=A.HEADS, d_model=A.D, d_head=A.DH, hidden=A.HIDDEN, max_months=A.T_MAX),
    )
    # leave one process family out: train on four, test on the unseen fifth
    tr_procs = np.array([m[0] for m in mtr])
    lopo = {}
    for pr in PROCS:
        keep = tr_procs != pr
        q = A.fit(Xtr[keep], Mtr[keep], Gtr[keep], ytr[keep], epochs=20, seed=1)
        msk = procs == pr
        lab = yte[msk] > I.NOMINAL_ALPHA
        pa = A.forward(q, Xte[msk], Mte[msk], Gte[msk])
        lopo[pr] = dict(brier_attention=float(np.mean((pa - yte[msk]) ** 2)),
                        brier_ar1=float(np.mean((p_ar[msk] - yte[msk]) ** 2)),
                        auc_attention=auc(pa, lab), auc_ar1=auc(p_ar[msk], lab))
        print(f"unseen {pr:10s} brier att {lopo[pr]['brier_attention']:.4f} ar1 {lopo[pr]['brier_ar1']:.4f}", flush=True)
    res["leave_one_process_out"] = lopo

    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "attention.json").write_text(json.dumps(res, indent=1))
    for k, v in res["by_process"].items():
        print(f"{k:10s} brier att {v['brier_attention']:.4f} ar1 {v['brier_ar1']:.4f}   "
              f"auc att {v['auc_attention']} ar1 {v['auc_ar1']}")
    print(f"done [{time.time() - t0:.0f}s]")


if __name__ == "__main__":
    main()
