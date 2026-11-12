"""Is "DQN beats the rule" an artefact of our reward weights?

The reward (paper eq. 8) is a design choice, so no single weighting can be called
optimal. Three checks show how much the conclusion depends on it:

A. Plain outcomes. Compare rule and DQN on quantities that do not use the reward
   at all: change in net worth, reserve coverage, goal progress, debt paid,
   cash shortfall, and the share of households that ever run short.
B. Re-weighting. Keep the same 12-month trajectories and re-score them under
   2,000 random weightings (each weight scaled by a factor between 1/4 and 4),
   plus one-at-a-time removals. Tests whether the ranking flips.
C. Retraining. Train a fresh DQN under four different reward philosophies and
   compare it with the (unchanged) rule under that same reward.

    python scripts/paper/exp_reward.py [ticks]
Writes paper/results/reward.json.
"""
import sys
import time

import numpy as np

from common import ROOT, bootstrap_ci, paired, save
from fa import dqn as Q, env as E

EVAL_SEEDS = (0, 1, 2, 3, 4)
N_DRAWS = 2000
ALTERNATIVES = {
    "cash first": dict(shortfall=6.0, resilience=3.0, call=0.04),
    "growth first": dict(resilience=0.5, goal=0.3, shortfall=1.5, debt_bonus=1.0),
    "net worth and shortfall only": dict(resilience=0.0, goal=0.0, comfort=0.0, unsuitable=0.0, debt_bonus=0.0),
    "all weights equal": dict(resilience=1.0, goal=1.0, shortfall=1.0, comfort=1.0, unsuitable=1.0, call=1.0 * E.CALL_COST,
                              debt_bonus=1.0),
}


def run(P, act, weights=None):
    outs = [Q.evaluate(P, act, seed=s, weights=weights) for s in EVAL_SEEDS]
    avg = lambda key: {k: np.mean([o[key][k] for o in outs], axis=0) for k in outs[0][key]}  # noqa: E731
    return dict(ret=np.mean([o["ret"] for o in outs], axis=0), terms=avg("terms"), outcomes=avg("outcomes"))


def score(terms, w):
    return terms["net_worth"] + sum(w[k] * v for k, v in terms.items() if k != "net_worth")


def main(ticks: int = 20000):
    t0 = time.time()
    train, test = E.population()
    net = Q.QNet.load(ROOT / "artifacts" / "dqn.npz")
    rule = run(test, E.rule_policy)
    dqn = run(test, Q.policy(net))
    print(f"evaluated [{time.time() - t0:.0f}s]", flush=True)

    # A. plain outcomes, higher is better except the shortfall rows
    outcomes = {}
    for k in rule["outcomes"]:
        a, b = dqn["outcomes"][k], rule["outcomes"][k]
        outcomes[k] = dict(rule=float(b.mean()), dqn=float(a.mean()), **paired(a, b))
    nw_d = dqn["outcomes"]["net_worth_change"] - rule["outcomes"]["net_worth_change"]
    sh_d = dqn["outcomes"]["shortfall_amount"] - rule["outcomes"]["shortfall_amount"]
    dominance = dict(
        dqn_dominates=float(((nw_d >= 0) & (sh_d <= 0) & ((nw_d > 0) | (sh_d < 0))).mean()),
        rule_dominates=float(((nw_d <= 0) & (sh_d >= 0) & ((nw_d < 0) | (sh_d > 0))).mean()),
    )
    dominance["trade_off"] = 1 - dominance["dqn_dominates"] - dominance["rule_dominates"]

    # B. re-weight the same trajectories
    base = dict(E.REWARD_WEIGHTS)
    rng = np.random.default_rng(0)
    diffs = []
    for _ in range(N_DRAWS):
        w = {k: v * float(np.exp(rng.uniform(np.log(0.25), np.log(4)))) for k, v in base.items()}
        diffs.append(score(dqn["terms"], w).mean() - score(rule["terms"], w).mean())
    diffs = np.array(diffs)
    one_at_a_time = {}
    for k in base:
        for label, f in (("removed", 0.0), ("x4", 4.0)):
            w = dict(base, **{k: base[k] * f})
            one_at_a_time[f"{k} {label}"] = float(score(dqn["terms"], w).mean() - score(rule["terms"], w).mean())
    contributions = {k: float(((dqn["terms"][k] - rule["terms"][k]) * (1 if k == "net_worth" else base[k])).mean())
                     for k in rule["terms"]}
    reweight = dict(n_draws=N_DRAWS, range="each weight x [1/4, 4], log-uniform",
                    share_dqn_better=float((diffs > 0).mean()), diff_median=float(np.median(diffs)),
                    diff_p5=float(np.quantile(diffs, 0.05)), diff_p95=float(np.quantile(diffs, 0.95)),
                    one_at_a_time=one_at_a_time, gain_by_term=contributions)
    print(f"re-weighting [{time.time() - t0:.0f}s]", flush=True)

    # C. retrain under alternative rewards
    retrained = {}
    for name, alt in ALTERNATIVES.items():
        q, _ = Q.train(train, ticks=ticks, seed=0, weights=alt)
        a = run(test, Q.policy(q), weights=alt)
        b = run(test, E.rule_policy, weights=alt)
        d = a["ret"] - b["ret"]
        retrained[name] = dict(weights=dict(base, **alt), rule=float(b["ret"].mean()), dqn=float(a["ret"].mean()),
                               mean_diff=float(d.mean()), ci=bootstrap_ci(d), share_dqn_better=float((d > 0).mean()),
                               net_worth_change_diff=float((a["outcomes"]["net_worth_change"]
                                                            - b["outcomes"]["net_worth_change"]).mean()),
                               shortfall_amount_diff=float((a["outcomes"]["shortfall_amount"]
                                                            - b["outcomes"]["shortfall_amount"]).mean()))
        print(f"retrained '{name}': dqn {retrained[name]['dqn']:.3f} rule {retrained[name]['rule']:.3f} "
              f"[{time.time() - t0:.0f}s]", flush=True)

    save("reward", dict(weights=base, outcomes=outcomes, dominance=dominance, reweight=reweight,
                        retrained=retrained, n_test=len(test["c"]), eval_seeds=list(EVAL_SEEDS), ticks=ticks))
    print(f"done [{time.time() - t0:.0f}s]")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 20000)
