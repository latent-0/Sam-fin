"""One command to rebuild every deployed artifact.

1. downloads the three public credit datasets once (data/raw/), augments them and
   trains the monotone LightGBM model  -> artifacts/credit_model.txt, credit_meta.json
2. simulates 12,000 households and trains the Double-DQN orchestrator -> artifacts/dqn.npz
3. a quick held-out evaluation of the selectors -> artifacts/evaluation.json

    python scripts/train_all.py [dqn_ticks]       # ~2-3 minutes on a desktop CPU

The full paper experiments live in scripts/paper/ (see scripts/paper/run_all.py).
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import train_credit  # noqa: E402
import train_dqn  # noqa: E402
from fa import dqn as Q, env as E  # noqa: E402


def quick_eval(net) -> dict:
    _, test = E.population()
    rng = np.random.default_rng(123)
    sel = {"random": lambda env: rng.integers(1, 64, env.n), "all": lambda env: np.full(env.n, 63),
           "rule": E.rule_policy, "dqn": Q.policy(net)}
    out = {}
    for k, act in sel.items():
        r = Q.evaluate(test, act, seed=0)
        ret = np.sort(r["ret"])
        out[k] = dict(mean=float(ret.mean()), cvar5=float(ret[: len(ret) // 20].mean()),
                      shortfall=float(r["short"].mean()), calls=float(r["calls"].mean()))
    return out


def main(ticks: int = 20000):
    t0 = time.time()
    print("== credit model")
    train_credit.main(0)
    print("== orchestrator")
    net, _ = train_dqn.main(ticks, 0)
    print("== evaluation on 2,400 held-out households")
    ev = quick_eval(net)
    (ROOT / "artifacts" / "evaluation.json").write_text(json.dumps(ev, indent=1))
    for k, v in ev.items():
        print(f"  {k:7s} mean {v['mean']:.3f}  cvar5 {v['cvar5']:.3f}  calls/yr {v['calls']:.1f}")
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 20000)
