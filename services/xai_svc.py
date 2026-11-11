"""XAI service: exact agent-Shapley values, interactions and Q-margins."""
from typing import List, Optional

from pydantic import BaseModel, Field

from services.base import make_app, to_params
from fa import xai as X

app = make_app("xai")


class ExplainRequest(BaseModel):
    params: dict
    mask: int = Field(..., ge=1, le=63)
    q: Optional[List[float]] = None
    draws: int = Field(X.DRAWS, ge=1, le=256)


@app.post("/explain")
def explain(req: ExplainRequest):
    return X.explain(to_params(req.params), req.mask, req.q, draws=req.draws)
