"""RL orchestrator service: Double-DQN forward pass plus the feedback bias."""
from functools import lru_cache
from pathlib import Path
from typing import List, Optional

import numpy as np
from pydantic import BaseModel

from services.base import ROOT, make_app, to_params
from fa import dqn as Q, env as E, xai as X

app = make_app("rl")
WEIGHTS = Path(ROOT) / "artifacts" / "dqn.npz"


@lru_cache(maxsize=1)
def net() -> Q.QNet:
    return Q.QNet.load(WEIGHTS)


class SelectRequest(BaseModel):
    params: dict
    bias: Optional[List[float]] = None


@app.post("/select")
def select(req: SelectRequest):
    env = E.Env(to_params(req.params), seed=0)
    q = net()(env.obs())[0]
    mask = int(np.argmax(q) + 1)
    shifted = False
    if req.bias is not None and len(req.bias) == E.N_ACTIONS:
        qb = q + np.asarray(req.bias, float)
        mask_b = int(np.argmax(qb) + 1)
        shifted, mask = mask_b != mask, mask_b
    return dict(mask=mask, agents=E.mask_names(mask), rule_agents=E.mask_names(int(E.rule_policy(env)[0])),
                q_top=X.q_top(q), q=q.round(5).tolist(), feedback_shifted=shifted)
