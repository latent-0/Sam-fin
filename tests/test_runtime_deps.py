"""The Vercel function installs only requirements.txt; the app must import without the training stack."""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

CODE = """
import sys
for m in ("pandas", "sklearn", "scipy", "lightgbm", "matplotlib"):
    sys.modules[m] = None          # any import of these now raises ImportError
import os
os.environ["INPROCESS"] = "1"
os.environ["FEEDBACK_DB"] = os.path.join(sys.argv[1], "fb.db")
import app
from fastapi.testclient import TestClient
import json
body = json.load(open("frontend/presets.json"))["gig"]
body.pop("label")
r = TestClient(app.app).post("/api/advise", json=body)
assert r.status_code == 200, r.text
print("ok")
"""


def test_app_runs_with_runtime_requirements_only(tmp_path):
    out = subprocess.run([sys.executable, "-c", CODE, str(tmp_path)], cwd=ROOT, capture_output=True, text=True)
    assert out.returncode == 0, out.stderr[-2000:]
    assert out.stdout.strip().endswith("ok")
