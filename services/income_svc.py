"""Income service: stability, Monte Carlo ruin, safe EMI, the readiness checklist,
the attention pattern reader and the income shock test."""
from typing import List, Optional

from pydantic import BaseModel, Field

from services.base import make_app
from fa import attention as At, income as I, requirements as Rq

app = make_app("income")


class IncomeRequest(BaseModel):
    income_history: List[float] = Field(..., min_length=2)
    monthly_expenses: float = Field(..., gt=0)
    existing_emi: float = 0.0
    new_emi: float = 0.0
    buffer: float = 0.0
    missed_payments: Optional[int] = None
    alpha: float = I.NOMINAL_ALPHA


@app.post("/analyse")
def analyse(req: IncomeRequest):
    r = I.analyse(req.income_history, req.monthly_expenses, req.new_emi, req.buffer, alpha=req.alpha)
    r["requirements"] = Rq.checklist(
        req.income_history, req.monthly_expenses, req.existing_emi, req.new_emi, req.buffer,
        r["stability"], r["ruin_probability"], r["safe_new_emi"], r["required_buffer"],
        missed_payments=req.missed_payments, alpha=req.alpha)
    r["pattern"] = At.read(req.income_history, req.monthly_expenses, req.new_emi, req.buffer)
    return r


class ShockRequest(BaseModel):
    income_history: List[float] = Field(..., min_length=2)
    monthly_expenses: float = Field(..., gt=0)
    new_emi: float = Field(0.0, ge=0)
    buffer: float = Field(0.0, ge=0)
    drop: float = Field(0.3, ge=0, le=1)
    months: int = Field(3, ge=1, le=12)
    start: int = Field(1, ge=1, le=12)


@app.post("/shock")
def shock(req: ShockRequest):
    return I.shock_test(req.income_history, req.monthly_expenses, req.new_emi, req.buffer,
                        req.drop, req.months, req.start)
