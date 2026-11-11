"""Run every paper experiment, then draw the figures.

    python scripts/paper/run_all.py            # writes paper/results/*.json and paper/figures/*.png

Set RESULTS_DIR / FIGURES_DIR to write somewhere other than the published results.
Order matters: exp_xai and exp_system use artifacts/dqn.npz and the credit model,
so run scripts/train_all.py first.
"""
import runpy
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

STEPS = ["exp_credit.py", "exp_income.py", "exp_rl.py", "exp_xai.py", "exp_system.py", "make_figures.py"]

if __name__ == "__main__":
    t0 = time.time()
    for s in STEPS:
        print(f"\n===== {s}", flush=True)
        runpy.run_path(str(HERE / s), run_name="__main__")
    print(f"\nall experiments done in {time.time() - t0:.0f}s")
