"""RQ5 / RQ6: are counterfactuals and history projections usable; does the feedback
rule adapt without harming plan value; is the service latency acceptable?

* Counterfactuals: 400 variable-income applicants with PD > 0.10 from a held-out
  augmented draw; share with a counterfactual, its cost, levers, and the
  36-month history projection.
* Feedback: simulated raters score a plan by 0.6 q + 0.4 (1 - (|S|-1)/5) plus
  noise (sigma = 0.4), with q the plan's normalised six-month value among all 63
  subsets; 600 gig-worker households arrive sequentially, with and without updates.
* Latency: all seven services on one machine (started here if not running),
  75 end-to-end requests (five presets x 15) and 45 direct calls per service.

    python scripts/paper/exp_system.py [--no-latency]
Writes paper/results/system.json.
"""
import json
import subprocess
import sys
import time

import numpy as np

from common import ROOT, save
from fa import credit as C, data as D, dqn as Q, env as E, feedback as F, xai as X

N_CF, CF_HORIZON = 400, 36
N_FB = 600


# ---------------------------------------------------------------------------
def counterfactual_study(seed: int = 99):
    model, meta = C.load()
    aug = D.augment(D.load_sources(), seed=seed)
    gig = aug[aug.gig == 1]
    p = model.predict(gig[D.FEATS].to_numpy(float))
    cand = gig[p > meta["approve_pd"]]
    rows = cand.sample(N_CF, random_state=seed)
    found, costs, nlev, months = 0, [], [], []
    freq = {k: 0 for k in ("emi_ratio", "buffer_months", "utilization", "history_months")}
    pd_before = []
    old_h = C.HISTORY_HORIZON
    C.HISTORY_HORIZON = CF_HORIZON
    try:
        for _, r in rows.iterrows():
            f = {k: float(r[k]) for k in D.FEATS}
            pd_before.append(float(model.predict(np.array([[f[k] for k in D.FEATS]]))[0]))
            cfs = C.counterfactuals(model, f, meta["approve_pd"], k=1)
            if cfs:
                found += 1
                costs.append(cfs[0]["cost"])
                nlev.append(len(cfs[0]["changes"]))
                for k in cfs[0]["changes"]:
                    freq[k] += 1
            saving = max(1 - f["expense_ratio"] - f["emi_ratio"], 0.0)
            hp = C.history_path(model, f, saving, max(f["expense_ratio"], 1e-9), meta["approve_pd"])
            months.append(hp["months_to_target"])
    finally:
        C.HISTORY_HORIZON = old_h
    reach = [m for m in months if m is not None]
    return dict(n=N_CF, validity=found / N_CF, cost_mean=float(np.mean(costs)), cost_median=float(np.median(costs)),
                n_changed_mean=float(np.mean(nlev)), feature_freq={k: v / N_CF for k, v in freq.items()},
                months_reachable_36=len(reach) / N_CF,
                months_median=float(np.median(reach)) if reach else None,
                months_p90=float(np.quantile(reach, 0.9)) if reach else None,
                mean_pd_before=float(np.mean(pd_before)))


# ---------------------------------------------------------------------------
def feedback_study(seed: int = 5):
    net = Q.QNet.load(ROOT / "artifacts" / "dqn.npz")
    P = E.sample_households(N_FB, np.random.default_rng(seed), persona=0)
    sizes = np.array([bin(m).count("1") for m in range(1, 64)])
    out = {}
    for mode in ("feedback", "frozen"):
        prefs = F.Preferences()
        rng = np.random.default_rng(seed + 1)
        ratings, qs, size = [], [], []
        for i in range(N_FB):
            P1 = {k: v[i:i + 1] for k, v in P.items()}
            bucket = F.bucket_of(float(P1["c"][0]))
            q = net(E.Env(P1).obs())[0] + prefs.bias[bucket]
            mask = int(q.argmax() + 1)
            v = X.coalition_values(P1, draws=8, seed=i)[1:]
            qn = (v[mask - 1] - v.min()) / max(v.max() - v.min(), 1e-9)
            score = 0.6 * qn + 0.4 * (1 - (sizes[mask - 1] - 1) / 5)
            rating = float(np.clip(np.round(1 + 4 * score + rng.normal(0, 0.4)), 1, 5))
            if mode == "feedback":
                prefs.update(bucket, mask, rating)
            ratings.append(rating), qs.append(qn), size.append(sizes[mask - 1])
        out[mode] = dict(first100=float(np.mean(ratings[:100])), last100=float(np.mean(ratings[-100:])),
                         q_first=float(np.mean(qs[:100])), q_last=float(np.mean(qs[-100:])),
                         size_first=float(np.mean(size[:100])), size_last=float(np.mean(size[-100:])))
    return out


# ---------------------------------------------------------------------------
def latency_study():
    import httpx
    sys.path.insert(0, str(ROOT))
    from fa import income as I
    from services.base import PORTS

    base = f"http://127.0.0.1:{PORTS['gateway']}"
    proc = None
    try:
        httpx.get(base + "/health", timeout=1)
    except Exception:  # noqa: BLE001
        proc = subprocess.Popen([sys.executable, str(ROOT / "run_all.py")], cwd=ROOT,
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        for _ in range(60):
            try:
                httpx.get(base + "/health", timeout=1)
                break
            except Exception:  # noqa: BLE001
                time.sleep(0.5)
    presets = json.loads((ROOT / "frontend" / "presets.json").read_text())
    bodies = [{k: v for k, v in p.items() if k != "label"} for p in presets.values()]

    def pct(xs):
        return dict(p50=float(np.percentile(xs, 50)), p95=float(np.percentile(xs, 95)), n=len(xs))

    out = {}
    try:
        with httpx.Client(timeout=30) as c:
            c.post(base + "/api/advise", json=bodies[0])                      # warm-up
            e2e = []
            for _ in range(15):
                for b in bodies:
                    t = time.perf_counter()
                    c.post(base + "/api/advise", json=b).raise_for_status()
                    e2e.append((time.perf_counter() - t) * 1000)
            out["gateway_e2e"] = pct(e2e)
            full = [c.post(base + "/api/advise", json=b).json() for b in bodies]
            url = {s: f"http://127.0.0.1:{PORTS[s]}" for s in ("income", "credit", "rl", "agents", "xai")}
            calls = {
                "income": lambda b, r: (url["income"] + "/analyse", dict(
                    income_history=b["income_history"], monthly_expenses=b["monthly_expenses"],
                    existing_emi=b["existing_emi"], new_emi=r["new_emi"],
                    buffer=b["liquid_savings"] + b["emergency_fund"])),
                "credit": lambda b, r: (url["credit"] + "/assess", dict(profile=b, new_emi=r["new_emi"])),
                "rl": lambda b, r: (url["rl"] + "/select", dict(params=_params(b, I))),
                "agents": lambda b, r: (url["agents"] + "/plan", dict(params=_params(b, I), scale=r["scale"],
                                                                      mask=r["orchestrator"]["mask"])),
                "xai": lambda b, r: (url["xai"] + "/explain", dict(params=_params(b, I),
                                                                   mask=r["orchestrator"]["mask"])),
            }
            for name, mk in calls.items():
                ts = []
                for _ in range(9):
                    for b, r in zip(bodies, full):
                        u, payload = mk(b, r)
                        t = time.perf_counter()
                        c.post(u, json=payload).raise_for_status()
                        ts.append((time.perf_counter() - t) * 1000)
                out[name] = pct(ts)
    finally:
        if proc:
            proc.terminate()
    return out


def _params(b, I):
    st = I.stats(b["income_history"])
    P1, _ = E.profile_to_params(b, st["cv"], st["mean"])
    return {k: [float(x) for x in v] for k, v in P1.items()}


def main():
    t0 = time.time()
    out = dict(counterfactual=counterfactual_study())
    print(f"counterfactuals [{time.time() - t0:.0f}s]", flush=True)
    out["feedback"] = feedback_study()
    print(f"feedback [{time.time() - t0:.0f}s]", flush=True)
    if "--no-latency" not in sys.argv:
        out["latency"] = latency_study()
    save("system", out)
    print(f"done [{time.time() - t0:.0f}s]")


if __name__ == "__main__":
    main()
