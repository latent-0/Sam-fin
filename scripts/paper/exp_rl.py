"""RQ3: does the orchestrator beat simple selectors in mean and tail, per persona and under shift?

Trains three DQN seeds with rule-guided exploration and three with uniform
exploration, then evaluates every selector on the same 2,400 held-out
households with common random numbers over five evaluation seeds.

    python scripts/paper/exp_rl.py [ticks]
Writes paper/results/rl.json and paper/results/rl_returns.npz.
"""
import sys
import time

import numpy as np

from common import RESULTS, bootstrap_ci, cvar, paired, save, wilcoxon_p
from fa import dqn as Q, env as E

TRAIN_SEEDS = (0, 1, 2)
EVAL_SEEDS = (0, 1, 2, 3, 4)
STRESS = dict(cv=1.5, var=1.08, savings=0.6)
N_ORACLE, ORACLE_DRAWS = 300, 8


def run_selector(P, act, stress=None):
    """Mean over evaluation seeds, per household."""
    outs = [Q.evaluate(P, act, seed=s, stress=stress) for s in EVAL_SEEDS]
    return {k: np.mean([o[k] for o in outs], axis=0) for k in outs[0]}


def random_policy(seed):
    rng = np.random.default_rng(seed)
    return lambda env: rng.integers(1, 64, env.n)


def oracle_policy(draws: int = ORACLE_DRAWS):
    """Myopic oracle: the subset with the best mean one-step reward over common-random-number draws."""
    def act(env: E.Env):
        n, reps = env.n, 63 * draws
        P = {k: np.repeat(v, reps) for k, v in env.P.items()}
        sim = E.Env(P, seed=0)
        sim.S = {k: np.repeat(v, reps) for k, v in env.S.items()}
        sim.t = env.t
        sim.z = np.repeat(env.z, reps)
        masks = np.tile(np.repeat(np.arange(1, 64), draws), n)
        rng = np.random.default_rng(env.t)
        e1 = np.tile(rng.standard_normal(draws), 63 * n)
        e2 = np.tile(rng.standard_normal(draws), 63 * n)
        r, _, _ = sim.step(masks, noise=(e1, e2))
        return r.reshape(n, 63, draws).mean(axis=2).argmax(axis=1) + 1
    return act


def summary(ret, short, calls, seed=0):
    return dict(mean=float(ret.mean()), ci=bootstrap_ci(ret, seed=seed), cvar5=cvar(ret),
                cvar_ci=bootstrap_ci(ret, stat=cvar, reps=1000, seed=seed), short=float(short.mean()),
                calls=float(calls.mean()), p10=float(np.quantile(ret, 0.1)), median=float(np.median(ret)))


def usage_by_persona(P, usage):
    return [usage[P["persona"] == k].mean(axis=0).round(4).tolist() for k in range(4)]


def main(ticks: int = 20000):
    t0 = time.time()
    train, test = E.population()
    nets, curves = {}, {}
    for guided in (True, False):
        for s in TRAIN_SEEDS:
            key = ("g" if guided else "u") + str(s)
            net, curve = Q.train(train, ticks=ticks, seed=s, guided=guided)
            nets[key], curves[key] = net, np.array(curve)
            if guided:
                net.save(RESULTS / f"dqn_seed{s}.npz")
            print(f"trained {key} [{time.time() - t0:.0f}s]", flush=True)

    res = {
        "random": run_selector(test, random_policy(123)),
        "all": run_selector(test, lambda env: np.full(env.n, 63)),
        "rule": run_selector(test, E.rule_policy),
    }
    per_seed = {"dqn": [], "dqn_noguide": [], "dqn_cvar5": [], "dqn_noguide_cvar5": []}
    for tag, prefix in (("dqn", "g"), ("dqn_noguide", "u")):
        runs = [run_selector(test, Q.policy(nets[prefix + str(s)])) for s in TRAIN_SEEDS]
        per_seed[tag] = [float(r["ret"].mean()) for r in runs]
        per_seed[tag + "_cvar5"] = [cvar(r["ret"]) for r in runs]
        res[tag] = {k: np.mean([r[k] for r in runs], axis=0) for k in runs[0]}
    print(f"evaluated [{time.time() - t0:.0f}s]", flush=True)

    head = {k: summary(v["ret"], v["short"], v["calls"]) for k, v in res.items()}
    d_rule = res["dqn"]["ret"] - res["rule"]["ret"]
    pv = paired(res["dqn"]["ret"], res["rule"]["ret"])
    pv.update(share_dqn_better=float((d_rule > 0).mean()), pct_gain=float(100 * d_rule.mean() / head["rule"]["mean"]))

    by_persona = {}
    for k, lab in enumerate(E.PERSONA_LABELS):
        m = test["persona"] == k
        by_persona[lab] = {s: dict(mean=float(res[s]["ret"][m].mean()), cvar5=cvar(res[s]["ret"][m]), n=int(m.sum()))
                           for s in ("rule", "random", "all", "dqn")}
        by_persona[lab]["diff_ci"] = bootstrap_ci(d_rule[m], seed=k)

    # myopic oracle on the first 300 households (costly)
    sub = {k: v[:N_ORACLE] for k, v in test.items()}
    orc = run_selector(sub, oracle_policy())
    oracle = dict(n=N_ORACLE, oracle=float(orc["ret"].mean()), rule=float(res["rule"]["ret"][:N_ORACLE].mean()),
                  dqn=float(res["dqn"]["ret"][:N_ORACLE].mean()), random=float(res["random"]["ret"][:N_ORACLE].mean()))
    oracle["gap_closed_vs_rule"] = (oracle["dqn"] - oracle["rule"]) / (oracle["oracle"] - oracle["rule"])
    oracle.update(oracle_shortfall=float(orc["short"].mean()), oracle_calls=float(orc["calls"].mean()))
    print(f"oracle [{time.time() - t0:.0f}s]", flush=True)

    # distribution shift, no retraining
    shift_runs = {"random": run_selector(test, random_policy(123), STRESS),
                  "all": run_selector(test, lambda env: np.full(env.n, 63), STRESS),
                  "rule": run_selector(test, E.rule_policy, STRESS)}
    dq = [run_selector(test, Q.policy(nets["g" + str(s)]), STRESS) for s in TRAIN_SEEDS]
    shift_runs["dqn"] = {k: np.mean([r[k] for r in dq], axis=0) for k in dq[0]}
    shift = {k: dict(mean=float(v["ret"].mean()), cvar5=cvar(v["ret"]), short=float(v["short"].mean()))
             for k, v in shift_runs.items()}
    shift["paired_dqn_minus_rule"] = paired(shift_runs["dqn"]["ret"], shift_runs["rule"]["ret"])

    out = dict(
        headline=head, per_seed=per_seed, paired_vs_rule=pv,
        paired_vs_random={k: v for k, v in paired(res["dqn"]["ret"], res["random"]["ret"]).items()},
        paired_guided=paired(res["dqn"]["ret"], res["dqn_noguide"]["ret"]),
        by_persona=by_persona, oracle=oracle, shift=shift,
        usage=dict(rule=usage_by_persona(test, res["rule"]["usage"]),
                   dqn=usage_by_persona(test, res["dqn"]["usage"]),
                   oracle=usage_by_persona(sub, orc["usage"])),
        ticks=ticks, seeds=list(TRAIN_SEEDS), eval_seeds=list(EVAL_SEEDS),
        n_test=len(test["c"]), n_train=len(train["c"]),
    )
    out["paired_vs_random"]["wilcoxon_p"] = wilcoxon_p(res["dqn"]["ret"] - res["random"]["ret"])
    save("rl", out)
    np.savez(RESULTS / "rl_returns.npz", persona=test["persona"],
             **{f"ret_{k}": v["ret"] for k, v in res.items()},
             **{f"shift_{k}": v["ret"] for k, v in shift_runs.items()},
             **{f"curve_{k}": v for k, v in curves.items()})
    print(f"done [{time.time() - t0:.0f}s]")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 20000)
