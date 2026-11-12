"""RL orchestrator service: Double-DQN forward pass, the feedback bias and input saliency."""
from functools import lru_cache
from pathlib import Path
from typing import List, Optional

import numpy as np
from pydantic import BaseModel

from services.base import ROOT, make_app, to_params
from fa import dqn as Q, env as E, xai as X

app = make_app("rl")
WEIGHTS = Path(ROOT) / "artifacts" / "dqn.npz"
SALIENCY_ROWS = 5


@lru_cache(maxsize=1)
def net() -> Q.QNet:
    return Q.QNet.load(WEIGHTS)


class SelectRequest(BaseModel):
    params: dict
    bias: Optional[List[float]] = None


def saliency(qnet: Q.QNet, s: np.ndarray, q: np.ndarray) -> dict:
    """Gradient x input of Q(s, a) for the top-scoring subsets: which state features moved each score."""
    rows = []
    for a in np.argsort(-q)[:SALIENCY_ROWS]:
        g = qnet.input_grad(s, int(a)) * s
        rows.append(dict(agents=E.mask_names(int(a) + 1), q=round(float(q[a]), 3), values=g.round(4).tolist()))
    return dict(features=E.OBS_LABELS, state=s.round(4).tolist(), rows=rows)


@app.post("/select")
def select(req: SelectRequest):
    env = E.Env(to_params(req.params), seed=0)
    s = env.obs()[0]
    qnet = net()
    q = qnet(s[None])[0]
    mask = int(np.argmax(q) + 1)
    shifted = False
    if req.bias is not None and len(req.bias) == E.N_ACTIONS:
        mask_b = int(np.argmax(q + np.asarray(req.bias, float)) + 1)
        shifted, mask = mask_b != mask, mask_b
    return dict(mask=mask, agents=E.mask_names(mask), rule_agents=E.mask_names(int(E.rule_policy(env)[0])),
                q_top=X.q_top(q), q=q.round(5).tolist(), feedback_shifted=shifted, saliency=saliency(qnet, s, q))
