"""API gateway + orchestration. Calls the six services and serves the web UI.

/api/advise         one-shot JSON
/api/advise/stream  server-sent events: one event per real pipeline step with its
                    measured duration, then the full result

Pipeline (paper Sec. III-A):
  1. income, credit and feedback bias in parallel
  2. RL orchestrator picks the agent subset from the household state plus the bias
  3. plan builder (agents) and Shapley explainer (xai) in parallel on the chosen subset
"""
import asyncio
import json
import time
from pathlib import Path
from typing import List, Optional

import httpx
from fastapi import HTTPException
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from services.base import INPROCESS, ROOT, SERVICES, URLS, make_app
from fa import env as E, feedback as F, income as I

ROOT = Path(ROOT)
FRONTEND = ROOT / "frontend"
RESULTS = ROOT / "paper" / "results"
app = make_app("gateway")

FRIENDLY = {"budget": "budgeting", "debt": "debt paydown", "emergency": "emergency fund", "goal": "goal saving",
            "risk": "income protection", "investment": "investing"}
STEP_LABELS = {"feedback": "Past feedback", "income": "Income analysis", "credit": "Loan readiness",
               "rl": "Choosing helpers", "agents": "Building the plan", "xai": "Explaining the choice"}
DISCLAIMER = "This is a preparation check, not a lender decision."
HEADLINES = {
    "likely": "Loan readiness looks good under these figures.",
    "borderline": "Loan readiness is borderline under these figures. A slow month could squeeze repayments; "
                  "the checklist shows what to strengthen.",
    "unlikely": "Loan readiness is weak under these figures. Strengthen cash flow before relying on this loan.",
}


# ---------------------------------------------------------------------------
# transport: real HTTP between services, or in-memory ASGI in serverless mode
# ---------------------------------------------------------------------------
if INPROCESS:
    # Every service is still its own FastAPI app with its own contract, but they are called through
    # in-memory ASGI transports instead of TCP, because a function instance cannot open six ports.
    from services import agents_svc, credit_svc, feedback_svc, income_svc, rl_svc, xai_svc

    class _Router(httpx.AsyncBaseTransport):
        def __init__(self):
            mods = dict(income=income_svc, credit=credit_svc, agents=agents_svc, rl=rl_svc, xai=xai_svc,
                        feedback=feedback_svc)
            self.t = {f"{n}.local": httpx.ASGITransport(app=m.app) for n, m in mods.items()}

        async def handle_async_request(self, request):
            return await self.t[request.url.host].handle_async_request(request)

    _transport = _Router()
else:
    # one pooled keep-alive client: a new connection per call cost ~200 ms end to end
    _transport = httpx.AsyncHTTPTransport(retries=1)

client = httpx.AsyncClient(transport=_transport, timeout=httpx.Timeout(30.0),
                           limits=httpx.Limits(max_keepalive_connections=32, max_connections=64))


async def call(svc: str, path: str, payload: Optional[dict] = None) -> dict:
    url = URLS[svc] + path
    r = await (client.post(url, json=payload) if payload is not None else client.get(url))
    if r.status_code >= 400:
        raise HTTPException(502, f"{svc} service error: {r.text[:300]}")
    return r.json()


# ---------------------------------------------------------------------------
# request model
# ---------------------------------------------------------------------------
class Loan(BaseModel):
    amount: float = Field(0, ge=0)
    annual_rate: float = Field(12, ge=0, le=100)
    months: int = Field(36, ge=1, le=480)


class Profile(BaseModel):
    income_history: List[float] = Field(..., min_length=2, max_length=120)
    monthly_expenses: float = Field(..., gt=0)
    existing_emi: float = Field(0, ge=0)
    loan: Loan = Loan()
    liquid_savings: float = Field(0, ge=0)
    emergency_fund: float = Field(0, ge=0)
    investments: float = Field(0, ge=0)
    debt_balance: float = Field(0, ge=0)
    debt_rate: float = Field(0, ge=0, le=100)
    goal_amount: float = Field(0, ge=0)
    goal_saved: float = Field(0, ge=0)
    goal_months: int = Field(12, ge=0, le=600)
    risk_tolerance: float = Field(0.5, ge=0, le=1)
    age: float = Field(35, ge=16, le=100)
    dependents: int = Field(0, ge=0, le=20)
    utilization: float = Field(0.3, ge=0, le=2)
    late_30: int = Field(0, ge=0)
    late_60: int = Field(0, ge=0)
    late_90: int = Field(0, ge=0)
    open_lines: int = Field(2, ge=0)
    missed_payments: Optional[int] = Field(None, ge=0)


def _jsonable(P: dict) -> dict:
    return {k: [float(x) for x in v] for k, v in P.items()}


# ---------------------------------------------------------------------------
# pipeline
# ---------------------------------------------------------------------------
async def pipeline(p: Profile, emit=None) -> dict:
    t0 = time.perf_counter()
    prof = p.model_dump()
    hist = [max(float(x), 0.0) for x in p.income_history]
    if sum(hist) <= 0:
        raise HTTPException(422, "Income history must contain at least one positive month.")
    prof["income_history"] = hist
    new_emi = I.emi_amount(p.loan.amount, p.loan.annual_rate, p.loan.months)
    st = I.stats(hist)
    mean = st["mean"]
    bucket = F.bucket_of(st["cv"])
    buffer = p.liquid_savings + p.emergency_fund

    async def step(sid: str, coro, detail):
        s = time.perf_counter()
        out = await coro
        now = time.perf_counter()
        if emit:
            await emit({"id": sid, "label": STEP_LABELS[sid], "detail": detail(out),
                        "ms": round((now - s) * 1000, 1), "at": round((now - t0) * 1000, 1)})
        return out

    # 1. income, credit, feedback in parallel
    inc, cr, fb = await asyncio.gather(
        step("income", call("income", "/analyse", dict(
            income_history=hist, monthly_expenses=p.monthly_expenses, existing_emi=p.existing_emi,
            new_emi=new_emi, buffer=buffer, missed_payments=p.missed_payments)),
            lambda r: (f"Simulated 4,000 possible years from {r['stability']['n']} months of income. "
                       f"Swing {r['stability']['cv'] * 100:.0f}%, stability {r['stability']['isi']:.0f}/100, "
                       f"pattern reader ruin estimate {r['pattern']['ruin_probability'] * 100:.0f}%.")),
        step("credit", call("credit", "/assess", dict(profile=prof, new_emi=new_emi)),
             lambda r: (f"Scored the loan with a model trained on 181,000 public borrower records. "
                        f"Estimated default risk {r['pd'] * 100:.1f}%, band: {r['decision']}.")),
        step("feedback", call("feedback", f"/bias/{bucket}"),
             lambda r: f"Looked up how similar households rated past plans ({r['learned']} learned preferences)."),
    )

    # 2. orchestrator
    P1, scale = E.profile_to_params(prof, st["cv"], mean)
    params = _jsonable(P1)
    rl = await step("rl", call("rl", "/select", dict(params=params, bias=fb["bias"])),
                    lambda r: ("Compared all 63 helper combinations. The learned selector chose: "
                               + ", ".join(FRIENDLY[a] for a in r["agents"]) + "."))
    mask = rl["mask"]

    # 3. plan + explanation in parallel
    plan, xai = await asyncio.gather(
        step("agents", call("agents", "/plan", dict(params=params, scale=scale, mask=mask)),
             lambda r: (f"Built a 12-month plan with {len(r['actions'])} actions and {len(r['conflicts'])} "
                        f"conflicts resolved.")),
        step("xai", call("xai", "/explain", dict(params=params, mask=mask, q=rl["q"])),
             lambda r: ("Tested all 64 helper combinations to measure each one's contribution. Most valuable here: "
                        + FRIENDLY[max(r["shapley"], key=r["shapley"].get)] + ".")),
    )

    if p.loan.amount > 0:
        headline = f"{HEADLINES[cr['decision']]} {DISCLAIMER}"
    else:
        headline = f"No new loan entered; the plan focuses on cash-flow resilience. {DISCLAIMER}"
    orchestrator = {k: v for k, v in rl.items() if k not in ("q", "saliency")}
    return dict(headline=headline, new_emi=new_emi, total_repayment=new_emi * p.loan.months if new_emi else 0.0,
                mean_income=mean, bucket=bucket, income=inc, credit=cr, orchestrator=orchestrator, plan=plan,
                explain=dict(xai, saliency=rl["saliency"]), scale=scale, total_ms=round((time.perf_counter() - t0) * 1000, 1))


# ---------------------------------------------------------------------------
# routes
# ---------------------------------------------------------------------------
@app.post("/api/advise")
async def advise(p: Profile):
    return await pipeline(p)


@app.post("/api/advise/stream")
async def advise_stream(p: Profile):
    q: asyncio.Queue = asyncio.Queue()

    async def emit(ev):
        await q.put(("step", ev))

    async def run():
        try:
            await q.put(("done", await pipeline(p, emit)))
        except HTTPException as e:
            await q.put(("error", {"detail": e.detail}))
        except Exception as e:  # noqa: BLE001 - surfaced to the client as an error event
            await q.put(("error", {"detail": f"{type(e).__name__}: {e}"}))

    async def gen():
        task = asyncio.create_task(run())
        while True:
            kind, data = await q.get()
            yield f"event: {kind}\ndata: {json.dumps(data)}\n\n"
            if kind in ("done", "error"):
                break
        await task

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


class Shock(BaseModel):
    income_history: List[float] = Field(..., min_length=2, max_length=120)
    monthly_expenses: float = Field(..., gt=0)
    new_emi: float = Field(0, ge=0)
    buffer: float = Field(0, ge=0)
    drop: float = Field(0.3, ge=0, le=1)
    months: int = Field(3, ge=1, le=12)
    start: int = Field(1, ge=1, le=12)


@app.post("/api/shock")
async def shock(s: Shock):
    return await call("income", "/shock", s.model_dump())


class Rating(BaseModel):
    bucket: str
    mask: int = Field(..., ge=1, le=63)
    rating: float = Field(..., ge=1, le=5)


@app.post("/api/feedback")
async def feedback(r: Rating):
    return await call("feedback", "/feedback", r.model_dump())


@app.get("/api/services")
async def services():
    async def one(s):
        try:
            return s, (await client.get(URLS[s] + "/health", timeout=3.0)).json().get("status", "down")
        except Exception:  # noqa: BLE001
            return s, "down"
    return dict(await asyncio.gather(*(one(s) for s in SERVICES)))


@app.get("/api/results")
def results():
    out = {}
    for k in ("rl", "income", "credit", "xai", "system", "attention"):
        f = RESULTS / f"{k}.json"
        if f.exists():
            out[k] = json.loads(f.read_text())
    return JSONResponse(out)


@app.get("/api/presets")
def presets():
    return FileResponse(FRONTEND / "presets.json")


@app.get("/")
def index():
    return FileResponse(FRONTEND / "index.html")
