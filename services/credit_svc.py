"""Credit service: monotone GBM score, conformal set, SHAP factors, counterfactuals."""
from pydantic import BaseModel

from services.base import make_app
from fa import credit as C, income as I

app = make_app("credit")


class CreditRequest(BaseModel):
    profile: dict
    new_emi: float = 0.0


@app.post("/assess")
def assess(req: CreditRequest):
    st = I.stats(req.profile["income_history"])
    return C.assess(req.profile, st, req.new_emi)


@app.get("/meta")
def meta():
    return C.load()[1]
