"""Dependency-free evaluator for a LightGBM text model.

The serverless deployment ships only numpy, so the trained booster is read from
its text dump (``artifacts/credit_model.txt``) and evaluated here:

* ``predict_raw``  vectorised log-odds for many rows (used for the 1,152-row
  counterfactual grid and the history projection);
* ``contributions`` exact path-dependent TreeSHAP (Lundberg et al., 2018),
  equivalent to LightGBM's ``pred_contrib`` output.

``tests/test_credit.py`` checks both against the lightgbm package.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


class _Tree:
    __slots__ = ("feat", "thr", "left", "right", "leaf", "icount", "lcount", "shrink", "leaf_only")

    def __init__(self, kv: dict):
        nl = int(kv["num_leaves"])
        self.leaf = np.array(kv["leaf_value"].split(), float)
        self.leaf_only = nl == 1
        if self.leaf_only:
            return
        self.feat = np.array(kv["split_feature"].split(), int)
        self.thr = np.array(kv["threshold"].split(), float)
        self.left = np.array(kv["left_child"].split(), int)
        self.right = np.array(kv["right_child"].split(), int)
        self.icount = np.array(kv["internal_count"].split(), float)
        self.lcount = np.array(kv["leaf_count"].split(), float)
        dt = np.array(kv.get("decision_type", " ".join(["2"] * len(self.feat))).split(), int)
        if np.any(dt & 1):
            raise ValueError("categorical splits are not supported")

    # vectorised prediction ------------------------------------------------
    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.leaf_only:
            return np.full(len(X), self.leaf[0])
        node = np.zeros(len(X), int)
        out = np.empty(len(X))
        active = np.ones(len(X), bool)
        while active.any():
            idx = np.nonzero(active)[0]
            nd = node[idx]
            go_left = X[idx, self.feat[nd]] <= self.thr[nd]
            nxt = np.where(go_left, self.left[nd], self.right[nd])
            leaf = nxt < 0
            out[idx[leaf]] = self.leaf[~nxt[leaf]]
            active[idx[leaf]] = False
            node[idx[~leaf]] = nxt[~leaf]
        return out

    def expected(self) -> float:
        if self.leaf_only:
            return float(self.leaf[0])
        return float(np.dot(self.lcount, self.leaf) / self.lcount.sum())

    # TreeSHAP ---------------------------------------------------------------
    def _cover(self, j):
        return self.lcount[~j] if j < 0 else self.icount[j]

    def shap(self, x: np.ndarray, phi: np.ndarray) -> None:
        if self.leaf_only:
            return

        def extend(m, pz, po, pi):
            d, z, o, w = (list(a) for a in m)
            ln = len(d)
            d.append(pi); z.append(pz); o.append(po); w.append(1.0 if ln == 0 else 0.0)
            for i in range(ln - 1, -1, -1):
                w[i + 1] += po * w[i] * (i + 1) / (ln + 1)
                w[i] = pz * w[i] * (ln - i) / (ln + 1)
            return d, z, o, w

        def unwind(m, i):
            d, z, o, w = (list(a) for a in m)
            ln = len(d) - 1
            n = w[ln]
            for j in range(ln - 1, -1, -1):
                if o[i] != 0:
                    t = w[j]
                    w[j] = n * (ln + 1) / ((j + 1) * o[i])
                    n = t - w[j] * z[i] * (ln - j) / (ln + 1)
                else:
                    w[j] = w[j] * (ln + 1) / (z[i] * (ln - j))
            for j in range(i, ln):
                d[j], z[j], o[j] = d[j + 1], z[j + 1], o[j + 1]
            return d[:ln], z[:ln], o[:ln], w[:ln]

        def unwound_sum(m, i):
            d, z, o, w = m
            ln = len(d) - 1
            n, tot = w[ln], 0.0
            for j in range(ln - 1, -1, -1):
                if o[i] != 0:
                    t = n * (ln + 1) / ((j + 1) * o[i])
                    tot += t
                    n = w[j] - t * z[i] * (ln - j) / (ln + 1)
                else:
                    tot += w[j] * (ln + 1) / (z[i] * (ln - j))
            return tot

        def recurse(j, m, pz, po, pi):
            m = extend(m, pz, po, pi)
            if j < 0:
                v = self.leaf[~j]
                d, z, o, _ = m
                for i in range(1, len(d)):
                    phi[d[i]] += unwound_sum(m, i) * (o[i] - z[i]) * v
                return
            f = self.feat[j]
            hot, cold = (self.left[j], self.right[j]) if x[f] <= self.thr[j] else (self.right[j], self.left[j])
            iz = io = 1.0
            d = m[0]
            if f in d[1:]:
                k = d.index(f, 1)
                iz, io = m[1][k], m[2][k]
                m = unwind(m, k)
            rj = self._cover(j)
            recurse(hot, m, iz * self._cover(hot) / rj, io, f)
            recurse(cold, m, iz * self._cover(cold) / rj, 0.0, f)

        recurse(0, ([], [], [], []), 1.0, 1.0, -1)


class TreeModel:
    def __init__(self, text: str):
        blocks = text.split("\nTree=")
        head = dict(line.split("=", 1) for line in blocks[0].splitlines() if "=" in line)
        self.feature_names = head["feature_names"].split()
        self.trees = []
        for b in blocks[1:]:
            body = b.split("\nend of trees")[0]
            kv = dict(line.split("=", 1) for line in body.splitlines()[1:] if "=" in line)
            self.trees.append(_Tree(kv))
        self.base_value = float(sum(t.expected() for t in self.trees))

    @classmethod
    def load(cls, path) -> "TreeModel":
        return cls(Path(path).read_text())

    def predict_raw(self, X) -> np.ndarray:
        X = np.atleast_2d(np.asarray(X, float))
        return np.sum([t.predict(X) for t in self.trees], axis=0)

    def predict(self, X) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-self.predict_raw(X)))

    def contributions(self, x) -> np.ndarray:
        """Per-feature log-odds contributions; they sum to raw score - base_value."""
        x = np.asarray(x, float).ravel()
        phi = np.zeros(len(self.feature_names))
        for t in self.trees:
            t.shap(x, phi)
        return phi
