"""RQ4: are agent-Shapley explanations faithful to the simulated outcome?

For 50 test households per persona the DQN chooses a subset. Shapley values
come from 16 draws; v is then re-evaluated with an INDEPENDENT set of 64 draws
and we measure the value lost when the chosen agent with the highest phi, the
lowest phi, or a random chosen agent is removed. Also checks efficiency and the
stability of phi as the number of Monte Carlo draws grows.

    python scripts/paper/exp_xai.py
Writes paper/results/xai.json.
"""
import time
from pathlib import Path

import numpy as np

from common import ROOT, save, wilcoxon_p
from fa import dqn as Q, env as E, xai as X

PER_PERSONA = 50
PHI_DRAWS, EVAL_DRAWS = 16, 64
CONV_KS, CONV_SEEDS, CONV_HH = (4, 8, 16, 32, 64), 5, 40


def household(P, i):
    return {k: v[i:i + 1] for k, v in P.items()}


def main(weights: Path = ROOT / "artifacts" / "dqn.npz"):
    t0 = time.time()
    _, test = E.population()
    net = Q.QNet.load(weights)
    rng = np.random.default_rng(0)
    idx = np.concatenate([np.nonzero(test["persona"] == k)[0][:PER_PERSONA] for k in range(4)])

    eff, top, low, rnd = [], [], [], []
    pos_chosen, nonpos_unchosen = [], []
    phi_by_persona = {k: [] for k in range(4)}
    inter_sum = np.zeros((6, 6))
    for i in idx:
        P1 = household(test, i)
        mask = int(net(E.Env(P1).obs())[0].argmax() + 1)
        v = X.coalition_values(P1, draws=PHI_DRAWS, seed=int(i))
        phi = X.shapley(v)
        eff.append(abs(phi.sum() - (v[63] - v[0])))
        for (a, b), x in X.interactions(v).items():
            inter_sum[a, b] += x
            inter_sum[b, a] += x
        phi_by_persona[int(test["persona"][i])].append(phi)
        v_ind = X.coalition_values(P1, draws=EVAL_DRAWS, seed=100_000 + int(i))
        chosen = [k for k in range(6) if mask >> k & 1]
        pos_chosen += [phi[k] > 0 for k in chosen]
        nonpos_unchosen += [phi[k] <= 0 for k in range(6) if k not in chosen]
        if len(chosen) >= 2:
            def loss(k):
                return v_ind[mask] - v_ind[mask & ~(1 << k)]
            top.append(loss(max(chosen, key=lambda k: phi[k])))
            low.append(loss(min(chosen, key=lambda k: phi[k])))
            rnd.append(loss(chosen[rng.integers(len(chosen))]))
    top, low, rnd = map(np.array, (top, low, rnd))
    print(f"deletion [{time.time() - t0:.0f}s]", flush=True)

    conv = {}
    for k in CONV_KS:
        sds = []
        for i in idx[:: max(1, len(idx) // CONV_HH)][:CONV_HH]:
            P1 = household(test, i)
            phis = np.array([X.shapley(X.coalition_values(P1, draws=k, seed=1000 * s + int(i)))
                             for s in range(CONV_SEEDS)])
            sds.append(phis.std(axis=0, ddof=1).mean())
        conv[k] = float(np.mean(sds))

    out = dict(
        n=len(idx), efficiency_residual_max=float(max(eff)), n_deletion=len(top),
        deletion=dict(top=float(top.mean()), low=float(low.mean()), random=float(rnd.mean()),
                      top_gt_low=float((top > low).mean()), wilcoxon_p_top_vs_low=wilcoxon_p(top - low),
                      wilcoxon_p_top_vs_rand=wilcoxon_p(top - rnd)),
        chosen_positive_share=float(np.mean(pos_chosen)),
        unchosen_nonpositive_share=float(np.mean(nonpos_unchosen)),
        convergence_sd=conv,
        mean_phi_by_persona={E.PERSONA_LABELS[k]: np.mean(v, axis=0).round(4).tolist()
                             for k, v in phi_by_persona.items()},
        mean_interaction=(inter_sum / len(idx)).round(4).tolist(),
    )
    save("xai", out)
    print(f"done [{time.time() - t0:.0f}s]")


if __name__ == "__main__":
    main()
