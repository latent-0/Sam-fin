"""Train the Double-DQN orchestrator and write artifacts/dqn.npz.

    python scripts/train_dqn.py [ticks] [seed] [--uniform]
"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from fa import dqn as Q, env as E  # noqa: E402


def main(ticks: int = 20000, seed: int = 0, guided: bool = True, out: Path | None = None):
    train, _ = E.population()
    t0 = time.time()
    net, curve = Q.train(train, ticks=ticks, seed=seed, guided=guided, log_every=max(ticks // 10, 1))
    out = out or ROOT / "artifacts" / "dqn.npz"
    out.parent.mkdir(exist_ok=True)
    net.save(out)
    print(f"saved {out} after {time.time() - t0:.0f}s")
    return net, curve


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    main(int(args[0]) if args else 20000, int(args[1]) if len(args) > 1 else 0, "--uniform" not in sys.argv)
