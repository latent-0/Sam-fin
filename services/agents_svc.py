"""Agent service: six agents, conflict resolution and the 12-month projection."""
from pydantic import BaseModel, Field

from services.base import make_app, to_params
from fa import agents as A

app = make_app("agents")


class PlanRequest(BaseModel):
    params: dict
    scale: float = Field(..., gt=0)
    mask: int = Field(..., ge=1, le=63)
    horizon: int = Field(12, ge=1, le=60)


@app.post("/plan")
def plan(req: PlanRequest):
    return A.build_plan(to_params(req.params), req.scale, req.mask, req.horizon)
