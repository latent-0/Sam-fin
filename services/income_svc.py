"""Income service: stability, Monte Carlo ruin, safe EMI and the readiness checklist."""
from typing import List, Optional

from pydantic import BaseModel, Field

from services.base import make_app
from fa import income as I, requirements as Rq

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
    return r
