"""Income pattern reader: a small multi-head attention model over the raw months.

The log-AR(1) model in ``income.py`` assumes one smooth process. It does badly
when income alternates between long busy and lean spells, or has occasional
near-zero months. This model reads the recorded months directly and estimates
the 12-month ruin probability for a given spending level, instalment and buffer.

Architecture (NumPy, trained by ``scripts/train_attention.py``):

* each month becomes a token: [x/mean - 1, log(x/mean), x < 0.7 mean, months before the latest]
* a query built from a learned vector plus the household's spending, instalment and buffer
  attends over the month tokens with H heads (scaled dot-product, padding masked)
* the heads are concatenated, projected, added back to the query (residual) and passed
  with the global features through a small ReLU layer to a ruin probability

Because the query depends on the spending and instalment, the attention weights show
which months the model reads for this particular loan. The model is trained on
simulated histories from the five processes in ``processes.py``, so it is a second
opinion next to the Monte Carlo estimate, not a replacement for it.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np

T_MAX = 24
N_TOK = 4
N_GLOB = 6
D = 16
HEADS = 2
DH = 8
HIDDEN = 32
ROOT = Path(__file__).resolve().parent.parent
WEIGHTS = ROOT / "artifacts" / "attention.npz"

# training ranges, in multiples of the household's mean income
RANGES = dict(expenses=(0.35, 0.95), emi=(0.0, 0.5), buffer=(0.0, 3.0))


# ---------------------------------------------------------------------------
# features
# ---------------------------------------------------------------------------
def encode(history, expenses: float, emi: float, buffer: float):
    """One household -> tokens (T_MAX, N_TOK), mask (T_MAX,), globals (N_GLOB,).

    Money amounts are in the same currency as ``history``; the latest T_MAX months are used.
    """
    x = np.asarray(history, float)[-T_MAX:]
    n = len(x)
    m = max(float(x.mean()), 1e-9)
    r = np.clip(x / m, 1e-3, None)
    tok = np.zeros((T_MAX, N_TOK))
    tok[:n, 0] = r - 1
    tok[:n, 1] = np.clip(np.log(r), -5, 1)
    tok[:n, 2] = r < 0.7
    tok[:n, 3] = (n - 1 - np.arange(n)) / (T_MAX - 1)
    mask = np.zeros(T_MAX)
    mask[:n] = 1
    cv = float(r.std(ddof=1)) if n > 1 else 0.0
    g = np.array([expenses / m, emi / m, buffer / m, n / T_MAX, cv, float(r.min())])
    return tok, mask, g


def clip_globals(g):
    g = g.copy()
    g[0] = np.clip(g[0], *RANGES["expenses"])
    g[1] = np.clip(g[1], *RANGES["emi"])
    g[2] = np.clip(g[2], *RANGES["buffer"])
    return g


# ---------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------
def init(seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)

    def w(*shape, fan):
        return rng.normal(0, 1 / np.sqrt(fan), shape)

    return dict(
        We=w(N_TOK, D, fan=N_TOK), be=np.zeros(D),
        cls=w(D, fan=D), Wg=w(N_GLOB, D, fan=N_GLOB),
        Wq=w(HEADS, D, DH, fan=D), Wk=w(HEADS, D, DH, fan=D), Wv=w(HEADS, D, DH, fan=D),
        Wo=w(HEADS * DH, D, fan=HEADS * DH),
        W1=w(D + N_GLOB, HIDDEN, fan=D + N_GLOB), b1=np.zeros(HIDDEN),
        w2=w(HIDDEN, fan=HIDDEN), b2=np.zeros(1),
    )


def forward(p: dict, X, M, G, cache: bool = False):
    """X (B,T,F), M (B,T), G (B,g) -> probability (B,), [cache]."""
    E = X @ p["We"] + p["be"]                                   # (B,T,D)
    c = p["cls"] + G @ p["Wg"]                                  # (B,D)
    neg = (1 - M) * -1e9
    heads = []
    for h in range(HEADS):
        q = c @ p["Wq"][h]                                      # (B,DH)
        K = E @ p["Wk"][h]                                      # (B,T,DH)
        V = E @ p["Wv"][h]
        s = np.einsum("bd,btd->bt", q, K) / np.sqrt(DH) + neg
        s -= s.max(axis=1, keepdims=True)
        a = np.exp(s)
        a /= a.sum(axis=1, keepdims=True)
        o = np.einsum("bt,btd->bd", a, V)
        heads.append((q, K, V, a, o))
    O = np.concatenate([hd[4] for hd in heads], axis=1)         # (B,H*DH)
    u = c + O @ p["Wo"]
    f = np.concatenate([u, G], axis=1)
    z1 = f @ p["W1"] + p["b1"]
    h1 = np.maximum(z1, 0)
    logit = h1 @ p["w2"] + p["b2"][0]
    prob = 1 / (1 + np.exp(-np.clip(logit, -30, 30)))
    if not cache:
        return prob
    return prob, dict(X=X, M=M, G=G, E=E, c=c, heads=heads, O=O, u=u, f=f, h1=h1)


def backward(p: dict, prob, y, C) -> tuple[float, dict]:
    """Binary cross-entropy with soft labels; returns (loss, grads)."""
    B = len(y)
    pc = np.clip(prob, 1e-7, 1 - 1e-7)
    loss = float(-np.mean(y * np.log(pc) + (1 - y) * np.log(1 - pc)))
    g = {k: np.zeros_like(v) for k, v in p.items()}
    dlogit = (prob - y) / B
    g["w2"] = C["h1"].T @ dlogit
    g["b2"] = np.array([dlogit.sum()])
    dz1 = np.outer(dlogit, p["w2"]) * (C["h1"] > 0)
    g["W1"] = C["f"].T @ dz1
    g["b1"] = dz1.sum(0)
    df = dz1 @ p["W1"].T
    du = df[:, :D]
    dc = du.copy()
    g["Wo"] = C["O"].T @ du
    dO = du @ p["Wo"].T
    dE = np.zeros_like(C["E"])
    E, c = C["E"], C["c"]
    for h, (q, K, V, a, o) in enumerate(C["heads"]):
        do = dO[:, h * DH:(h + 1) * DH]
        da = np.einsum("bd,btd->bt", do, V)
        dV = a[:, :, None] * do[:, None, :]
        ds = a * (da - (a * da).sum(axis=1, keepdims=True)) / np.sqrt(DH)
        dq = np.einsum("bt,btd->bd", ds, K)
        dK = ds[:, :, None] * q[:, None, :]
        g["Wq"][h] = c.T @ dq
        dc += dq @ p["Wq"][h].T
        g["Wk"][h] = np.einsum("btd,bte->de", E, dK)
        g["Wv"][h] = np.einsum("btd,bte->de", E, dV)
        dE += dK @ p["Wk"][h].T + dV @ p["Wv"][h].T
    g["cls"] = dc.sum(0)
    g["Wg"] = C["G"].T @ dc
    g["We"] = np.einsum("btf,btd->fd", C["X"], dE)
    g["be"] = dE.sum(axis=(0, 1))
    return loss, g


class Adam:
    def __init__(self, p, lr=3e-3, b1=0.9, b2=0.999):
        self.m = {k: np.zeros_like(v) for k, v in p.items()}
        self.v = {k: np.zeros_like(v) for k, v in p.items()}
        self.lr, self.b1, self.b2, self.t = lr, b1, b2, 0

    def step(self, p, g, lr=None):
        self.t += 1
        lr = lr or self.lr
        for k in p:
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * g[k]
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * g[k] ** 2
            p[k] -= lr * (self.m[k] / (1 - self.b1 ** self.t)) / (np.sqrt(self.v[k] / (1 - self.b2 ** self.t)) + 1e-8)


def fit(X, M, G, y, epochs: int = 30, batch: int = 256, lr: float = 3e-3, seed: int = 0, log=None) -> dict:
    p = init(seed)
    opt = Adam(p, lr)
    rng = np.random.default_rng(seed)
    n = len(y)
    for ep in range(epochs):
        idx = rng.permutation(n)
        tot = 0.0
        lr_ep = lr * (0.1 + 0.9 * (1 - ep / epochs))
        for i in range(0, n, batch):
            j = idx[i:i + batch]
            prob, C = forward(p, X[j], M[j], G[j], cache=True)
            loss, g = backward(p, prob, y[j], C)
            opt.step(p, g, lr_ep)
            tot += loss * len(j)
        if log:
            log(ep, tot / n)
    return p


# ---------------------------------------------------------------------------
# inference
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def load() -> dict:
    with np.load(WEIGHTS) as z:
        return {k: z[k] for k in z.files}


def token_attention(p: dict, X, M):
    """Month-to-month attention per head, each month used as a query (for display)."""
    E = X @ p["We"] + p["be"]
    n = int(M.sum())
    out = []
    for h in range(HEADS):
        Q = E[:n] @ p["Wq"][h]
        K = E[:n] @ p["Wk"][h]
        s = Q @ K.T / np.sqrt(DH)
        s -= s.max(axis=1, keepdims=True)
        a = np.exp(s)
        out.append((a / a.sum(axis=1, keepdims=True)).round(4).tolist())
    return out


def read(history, expenses: float, emi: float, buffer: float, p: dict | None = None) -> dict:
    p = p or load()
    tok, mask, g = encode(history, expenses, emi, buffer)
    gc = clip_globals(g)
    prob, C = forward(p, tok[None], mask[None], gc[None], cache=True)
    n = int(mask.sum())
    offset = max(len(history) - T_MAX, 0)
    heads = [hd[3][0, :n] for hd in C["heads"]]
    focus = np.mean(heads, axis=0)
    return dict(
        ruin_probability=float(prob[0]),
        months_used=n,
        first_month=offset + 1,
        query_attention=[h.round(4).tolist() for h in heads],
        focus=focus.round(4).tolist(),
        top_months=[int(offset + i + 1) for i in np.argsort(-focus)[:3]],
        head_maps=token_attention(p, tok, mask),
        outside_training_range=bool(np.any(np.abs(gc - g) > 1e-9)),
    )
