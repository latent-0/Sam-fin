"""End-to-end: the gateway in serverless mode (all services in-process), as deployed on Vercel."""
import json
import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
pytestmark = pytest.mark.skipif(not (ROOT / "artifacts" / "dqn.npz").exists()
                                or not (ROOT / "artifacts" / "credit_model.txt").exists(),
                                reason="run scripts/train_all.py first")


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    os.environ["INPROCESS"] = "1"
    os.environ["FEEDBACK_DB"] = str(tmp_path_factory.mktemp("fb") / "feedback.db")
    from fastapi.testclient import TestClient
    import app as entry
    with TestClient(entry.app) as c:
        yield c


PRESETS = json.loads((ROOT / "frontend" / "presets.json").read_text())


def body(name):
    return {k: v for k, v in PRESETS[name].items() if k != "label"}


def test_static_and_meta_routes(client):
    assert client.get("/").status_code == 200
    assert client.get("/health").json() == {"service": "gateway", "status": "ok"}
    assert set(client.get("/api/presets").json()) == set(PRESETS)
    assert client.get("/api/services").json() == {k: "ok" for k in
                                                  ("income", "credit", "agents", "rl", "xai", "feedback")}
    assert {"rl", "income", "credit", "xai", "system", "attention"} <= set(client.get("/api/results").json())


@pytest.mark.parametrize("name", list(PRESETS))
def test_advise_contract(client, name):
    r = client.post("/api/advise", json=body(name))
    assert r.status_code == 200, r.text
    d = r.json()
    for k in ("headline", "new_emi", "total_repayment", "mean_income", "bucket", "income", "credit",
              "orchestrator", "plan", "explain", "scale", "total_ms"):
        assert k in d
    assert d["credit"]["decision"] in ("likely", "borderline", "unlikely")
    assert 1 <= d["orchestrator"]["mask"] <= 63
    assert len(d["income"]["requirements"]) == 9
    assert len(d["plan"]["projection"]["with_plan"]) == 12
    assert abs(d["explain"]["efficiency_gap"]) < 1e-9


def test_stream_emits_six_steps_then_done(client):
    with client.stream("POST", "/api/advise/stream", json=body("gig")) as r:
        text = "".join(r.iter_text())
    events = [blk.split("\n")[0][7:] for blk in text.strip().split("\n\n")]
    assert events.count("step") == 6 and events[-1] == "done"
    ids = {json.loads(b.split("data: ", 1)[1])["id"] for b in text.strip().split("\n\n")[:-1]}
    assert ids == {"feedback", "income", "credit", "rl", "agents", "xai"}


def test_feedback_roundtrip(client):
    d = client.post("/api/advise", json=body("gig")).json()
    r = client.post("/api/feedback", json={"bucket": d["bucket"], "mask": d["orchestrator"]["mask"], "rating": 5})
    assert r.status_code == 200 and r.json()["n"] == 1


def test_validation(client):
    assert client.post("/api/advise", json={"income_history": [1000], "monthly_expenses": 500}).status_code == 422
    assert client.post("/api/advise", json={"income_history": [0, 0], "monthly_expenses": 500}).status_code == 422


def test_new_response_blocks(client):
    d = client.post("/api/advise", json=body("gig")).json()
    assert len(d["income"]["pattern"]["focus"]) == 3
    sal = d["explain"]["saliency"]
    assert len(sal["features"]) == 14 and len(sal["rows"]) == 5 and len(sal["rows"][0]["values"]) == 14
    assert len(d["explain"]["interaction_matrix"]) == 6
    assert "saliency" not in d["orchestrator"]


def test_shock_route(client):
    b = body("freelancer")
    r = client.post("/api/shock", json=dict(income_history=b["income_history"], monthly_expenses=b["monthly_expenses"],
                                           new_emi=10000, buffer=130000, drop=0.5, months=3, start=2))
    assert r.status_code == 200
    s = r.json()
    assert s["ruin"] >= s["ruin_base"] and s["shock"]["start"] == 2
    assert client.post("/api/shock", json=dict(income_history=[1, 2], monthly_expenses=1, drop=2)).status_code == 422
