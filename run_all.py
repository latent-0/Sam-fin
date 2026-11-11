"""Start all seven services locally, one uvicorn process per service.

    python run_all.py          # then open http://127.0.0.1:8100
"""
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from services.base import PORTS  # noqa: E402

MODULES = {"income": "income_svc", "credit": "credit_svc", "agents": "agents_svc", "rl": "rl_svc",
           "xai": "xai_svc", "feedback": "feedback_svc", "gateway": "gateway"}


def main():
    procs = []
    for name, mod in MODULES.items():
        cmd = [sys.executable, "-m", "uvicorn", f"services.{mod}:app", "--host", "127.0.0.1",
               "--port", str(PORTS[name]), "--log-level", "warning"]
        procs.append(subprocess.Popen(cmd, cwd=ROOT))
        print(f"{name:9s} http://127.0.0.1:{PORTS[name]}")
    print("\nOpen http://127.0.0.1:8100  (Ctrl+C to stop)")

    def stop(*_):
        for p in procs:
            p.terminate()
        sys.exit(0)

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    while all(p.poll() is None for p in procs):
        time.sleep(1)
    stop()


if __name__ == "__main__":
    main()
