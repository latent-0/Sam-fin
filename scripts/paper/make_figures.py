"""Draw the paper figures from paper/results/*.json (and rl_returns.npz when present).

    python scripts/paper/make_figures.py
"""
import numpy as np

from common import FIGURES, RESULTS, load

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

C = dict(blue="#0072B2", sky="#56B4E9", orange="#D55E00", grey="#999999", pink="#CC79A7", gold="#E69F00",
         green="#009E73")
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150,
                     "axes.grid": True, "grid.alpha": 0.25})
AGENTS = ["budget", "debt", "emerg", "goal", "risk", "invest"]


def _save(fig, name):
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIGURES / f"{name}.png", bbox_inches="tight")
    plt.close(fig)
    print("figure", name)


def credit():
    d = load("credit")
    s = d["per_seed_summary"]
    names = [("full", "Full"), ("nomono", "No\nmonotone"), ("novol", "No\nvolatility"), ("core", "Core\nonly"),
             ("logreg", "Logistic\nreg.")]
    cols = [C["blue"], C["sky"], C["orange"], C["grey"], C["pink"]]
    fig, ax = plt.subplots(1, 3, figsize=(10, 2.8))
    for a, (m, title) in zip(ax, (("auc", "AUC (higher better)"), ("brier", "Brier score (lower better)"),
                                  ("ece", "ECE (lower better)"))):
        a.bar([n[1] for n in names], [s[k][m][0] for k, _ in names], yerr=[s[k][m][1] for k, _ in names],
              color=cols, capsize=3)
        a.set_title(title)
        if m == "auc":
            a.set_ylim(0.7, 0.9)
    _save(fig, "fig_credit_ablation")

    cf = d["conformal"]
    keys = ["overall", "class0", "class1", "steady", "gig", "gmsc", "taiwan", "german"]
    fig, ax = plt.subplots(1, 2, figsize=(8, 2.8), gridspec_kw={"width_ratios": [2.2, 1]})
    ax[0].bar(keys, [cf[k][0] for k in keys], yerr=[cf[k][1] for k in keys], color=C["blue"], capsize=3)
    ax[0].axhline(0.9, ls="--", c="k", lw=0.8)
    ax[0].set_ylim(0.7, 1)
    ax[0].set_title("(a) empirical coverage, target 0.90")
    ax[0].tick_params(axis="x", rotation=40)
    ax[1].bar(["singleton", "both labels", "empty"], [cf["single"][0], cf["both"][0], cf["empty"][0]],
              color=[C["green"], C["orange"], C["grey"]])
    ax[1].set_title("(b) prediction-set size")
    _save(fig, "fig_conformal")


def income():
    d = load("income")
    A = {(a["method"], a["n"]): a for a in d["agg"]}
    ns = [3, 6, 12, 24]
    meth = [("ours", "Shrinkage AR(1) (ours)", C["blue"]), ("ours_cal", "Shrinkage AR(1) + calibrated alpha",
                                                              C["orange"]),
            ("mle_ar1", "AR(1), no shrinkage", C["sky"]), ("bootstrap", "iid bootstrap", C["green"]),
            ("worst_month", "Worst month", C["gold"]), ("foir_35", "35% of income rule", C["pink"])]
    fig, ax = plt.subplots(1, 2, figsize=(9, 3))
    for m, lab, col in meth:
        if (m, 3) not in A:
            continue
        ax[0].plot(ns, [A[m, n]["realised"] for n in ns], "o-", c=col, label=lab, ms=3)
        ax[1].plot(ns, [A[m, n]["ok"] for n in ns], "o-", c=col, ms=3)
    ax[0].axhline(d["alpha"], ls="--", c="k", lw=0.8)
    ax[0].set(title="(a) mean realised ruin, nominal 0.10", xlabel="months of history", xticks=ns)
    ax[1].set(title="(b) share with realised ruin <= 0.10", xlabel="months of history", xticks=ns)
    ax[0].legend(fontsize=7)
    _save(fig, "fig_safe_emi")

    procs = sorted({b["proc"] for b in d["by_proc"]})
    B = {(b["method"], b["proc"], b["n"]): b["realised"] for b in d["by_proc"]}
    Cc = {(c["proc"], c["n"]): c["realised"] for c in d["calibrated"]}
    fig, ax = plt.subplots(1, len(procs), figsize=(12, 2.6), sharey=True)
    for a, p in zip(ax, procs):
        for m, lab, col in meth:
            ys = [Cc[p, n] for n in ns] if m == "ours_cal" else [B.get((m, p, n), np.nan) for n in ns]
            a.plot(ns, ys, "o-", c=col, ms=3)
        a.axhline(0.1, ls="--", c="k", lw=0.8)
        a.set(title=p, xticks=ns)
    ax[0].set_ylabel("realised ruin")
    _save(fig, "fig_ruin_by_process")


def rl():
    d = load("rl")
    h = d["headline"]
    sel = [("random", "Random", C["grey"]), ("all", "All six", C["pink"]), ("rule", "Rule", C["gold"]),
           ("dqn", "DQN", C["blue"])]
    fig, ax = plt.subplots(1, 2, figsize=(8, 2.8))
    for i, (k, lab, col) in enumerate(sel):
        ax[0].bar(lab, h[k]["mean"], color=col, yerr=[[h[k]["mean"] - h[k]["ci"][0]], [h[k]["ci"][1] - h[k]["mean"]]],
                  capsize=3)
        ax[1].bar(lab, h[k]["cvar5"], color=col,
                  yerr=[[h[k]["cvar5"] - h[k]["cvar_ci"][0]], [h[k]["cvar_ci"][1] - h[k]["cvar5"]]], capsize=3)
    ax[0].set_title("mean 12-month return [95% CI]")
    ax[1].set_title("CVaR 5% of return")
    _save(fig, "fig_selectors")

    bp = d["by_persona"]
    labs = list(bp)
    x = np.arange(len(labs))
    fig, ax = plt.subplots(figsize=(7, 2.8))
    for j, (k, lab, col) in enumerate(sel[0:1] + sel[2:]):
        ax.bar(x + (j - 1) * 0.27, [bp[p][k]["mean"] for p in labs], 0.27, color=col, label=lab)
    ax.set_xticks(x, labs)
    ax.legend(fontsize=7)
    ax.set_title("mean return by persona")
    _save(fig, "fig_persona")

    fig, ax = plt.subplots(1, 3, figsize=(12, 2.8))
    for a, (k, title) in zip(ax, (("rule", "Rule"), ("dqn", "DQN"), ("oracle", "Myopic oracle (300 hh)"))):
        u = np.array(d["usage"][k])
        im = a.imshow(u, vmin=0, vmax=1, cmap="viridis")
        for i in range(u.shape[0]):
            for j in range(u.shape[1]):
                a.text(j, i, f"{u[i, j]:.2f}", ha="center", va="center", fontsize=7,
                       color="k" if u[i, j] > 0.6 else "w")
        a.set_xticks(range(6), AGENTS)
        a.set_yticks(range(4), labs if k == "rule" else [""] * 4)
        a.set_title(title)
        a.grid(False)
    fig.colorbar(im, ax=ax, shrink=0.8, label="selection frequency")
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / "fig_usage.png", bbox_inches="tight")
    plt.close(fig)

    npz = RESULTS / "rl_returns.npz"
    if npz.exists():
        z = np.load(npz)
        fig, ax = plt.subplots(1, 2, figsize=(10, 3))
        parts = ax[0].violinplot([z[f"ret_{k}"] for k, _, _ in sel], showmedians=True)
        for b, (_, _, col) in zip(parts["bodies"], sel):
            b.set_facecolor(col)
        ax[0].set_xticks(range(1, 5), [s[1] for s in sel])
        ax[0].set_title("12-month return per household")
        for key, lab, col in (("g", "rule-guided exploration", C["blue"]), ("u", "uniform exploration",
                                                                             C["orange"])):
            cs = [z[f"curve_{key}{s}"] for s in d["seeds"] if f"curve_{key}{s}" in z]
            if not cs:
                continue
            L = min(map(len, cs))
            arr = np.array([np.convolve(c[:L], np.ones(25) / 25, mode="valid") for c in cs])
            xs = np.arange(arr.shape[1])
            ax[1].plot(xs, arr.mean(0), c=col, label=lab)
            ax[1].fill_between(xs, arr.min(0), arr.max(0), color=col, alpha=0.25)
        ax[1].set(title="learning curves (min-max over seeds)", xlabel="training episodes", ylabel="episode return")
        ax[1].legend(fontsize=7)
        _save(fig, "fig_returns_and_learning")


def xai():
    d = load("xai")
    labs = list(d["mean_phi_by_persona"])
    M = np.array([d["mean_phi_by_persona"][p] for p in labs])
    fig, ax = plt.subplots(1, 2 if "mean_interaction" in d else 1, figsize=(10, 3), squeeze=False)
    v = np.abs(M).max()
    im = ax[0, 0].imshow(M, cmap="RdBu_r", vmin=-v, vmax=v)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax[0, 0].text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=7)
    ax[0, 0].set_xticks(range(6), AGENTS)
    ax[0, 0].set_yticks(range(len(labs)), labs)
    ax[0, 0].set_title("(a) mean Shapley value by persona")
    ax[0, 0].grid(False)
    fig.colorbar(im, ax=ax[0, 0], shrink=0.8)
    if "mean_interaction" in d:
        I = np.array(d["mean_interaction"])
        v = np.abs(I).max()
        im = ax[0, 1].imshow(I, cmap="PuOr", vmin=-v, vmax=v)
        ax[0, 1].set_xticks(range(6), AGENTS)
        ax[0, 1].set_yticks(range(6), AGENTS)
        ax[0, 1].set_title("(b) mean pairwise interaction")
        ax[0, 1].grid(False)
        fig.colorbar(im, ax=ax[0, 1], shrink=0.8)
    _save(fig, "fig_shapley")

    fig, ax = plt.subplots(1, 2, figsize=(8, 2.6))
    dl = d["deletion"]
    ax[0].bar(["highest phi", "random", "lowest phi"], [dl["top"], dl["random"], dl["low"]],
              color=[C["blue"], C["grey"], C["orange"]])
    ax[0].set_title(f"(a) value lost on removal, n={d['n_deletion']}")
    ks = sorted(map(int, d["convergence_sd"]))
    ax[1].plot(ks, [d["convergence_sd"][str(k)] for k in ks], "o-", c=C["blue"])
    ax[1].set_xscale("log", base=2)
    ax[1].set(title="(b) sd of phi over 5 seeds", xlabel="Monte Carlo draws per coalition")
    _save(fig, "fig_xai_faithfulness")


def system():
    d = load("system")
    cf = d["counterfactual"]
    fig, ax = plt.subplots(1, 2, figsize=(8, 2.6))
    ff = cf["feature_freq"]
    ax[0].bar(["EMI\nreduction", "buffer\nincrease", "more\nhistory", "utilisation\ncut"],
              [ff["emi_ratio"], ff["buffer_months"], ff["history_months"], ff["utilization"]], color=C["green"])
    ax[0].set_title(f"(a) levers used ({cf['validity'] * 100:.0f}% have a plan)")
    fb = d["feedback"]
    x = np.arange(3)
    for j, (k, col) in enumerate((("feedback", C["blue"]), ("frozen", C["grey"]))):
        ax[1].bar(x + (j - 0.5) * 0.35, [fb[k]["last100"] / 5, fb[k]["q_last"], fb[k]["size_last"] / 6], 0.35,
                  color=col, label="with feedback updates" if k == "feedback" else "frozen policy")
    ax[1].set_xticks(x, ["rating / 5", "plan value", "agents / 6"])
    ax[1].set_title("(b) last 100 interactions")
    ax[1].legend(fontsize=7)
    _save(fig, "fig_counterfactual_feedback")

    if "latency" in d:
        L = d["latency"]
        keys = [k for k in ("income", "credit", "rl", "agents", "xai", "gateway_e2e") if k in L]
        x = np.arange(len(keys))
        fig, ax = plt.subplots(figsize=(5, 2.6))
        ax.bar(x - 0.2, [L[k]["p50"] for k in keys], 0.4, label="p50", color=C["blue"])
        ax.bar(x + 0.2, [L[k]["p95"] for k in keys], 0.4, label="p95", color=C["orange"])
        ax.set_yscale("log")
        ax.set_xticks(x, [k.replace("gateway_e2e", "end-to-end") for k in keys])
        ax.set_ylabel("latency (ms)")
        ax.legend(fontsize=7)
        _save(fig, "fig_latency")


if __name__ == "__main__":
    for name, fn in (("credit", credit), ("income", income), ("rl", rl), ("xai", xai), ("system", system)):
        if (RESULTS / f"{name}.json").exists():
            fn()
