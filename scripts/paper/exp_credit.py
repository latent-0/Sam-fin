"""RQ1: does the monotone volatility-aware model rank and calibrate risk, and does
it identify when it is uncertain?

Each of 5 seeds redraws the augmentation and uses a stratified 70/15/15 split.
Ablations remove (a) monotone constraints, (b) the four volatility features,
(c) all non-core features; a logistic regression gives a linear reference.
Also reports AUC on the ORIGINAL public labels per source, class-conditional
conformal coverage by group, and a leave-one-dataset-out transfer test.

    python scripts/paper/exp_credit.py
Writes paper/results/credit.json and paper/results/credit_shap.json.
"""
import time

import numpy as np

from common import mean_sd, save
from fa import data as D, gbm as G

SEEDS = (0, 1, 2, 3, 4)
VOL4 = ("income_cv", "semi_dev_ratio", "gig", "history_months")
SOURCES = ("gmsc", "taiwan", "german")


def logreg(tr, te, feats, seed):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=seed))
    m.fit(tr[feats].to_numpy(float), tr["y_aug"].to_numpy())
    return m.predict_proba(te[feats].to_numpy(float))[:, 1]


def logloss(p, y):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def main():
    t0 = time.time()
    src = D.load_sources()
    variants = {
        "full": (D.FEATS, True),
        "nomono": (D.FEATS, False),
        "novol": ([f for f in D.FEATS if f not in VOL4], True),
        "core": (D.CORE, True),
    }
    per = {k: {m: [] for m in ("auc", "ap", "brier", "logloss", "ece")} for k in list(variants) + ["logreg"]}
    orig = {s: [] for s in SOURCES}
    conf = {k: [] for k in ("overall", "class0", "class1", "gig", "steady", *SOURCES, "single", "both", "empty")}
    lodo = {s: {"auc_aug": [], "auc_orig": []} for s in SOURCES}
    shap_sample = None

    for seed in SEEDS:
        aug = D.augment(src, seed=seed)
        tr, cal, te = D.split(aug, seed=seed)
        y_te = te["y_aug"].to_numpy()
        for name, (feats, mono) in variants.items():
            b = G.train(tr[feats], tr["y_aug"], feats=feats, monotone=mono, seed=seed)
            p = b.predict(te[feats].to_numpy(float))
            m = G.metrics(p, y_te)
            for k, key in (("auc", "auc"), ("ap", "pr_auc"), ("brier", "brier"), ("ece", "ece")):
                per[name][k].append(m[key])
            per[name]["logloss"].append(logloss(p, y_te))
            if name == "full":
                for s in SOURCES:
                    msk = (te.source == s).to_numpy()
                    orig[s].append(G.metrics(p[msk], te["y"].to_numpy()[msk])["auc"])
                pc = b.predict(cal[feats].to_numpy(float))
                q = G.conformal_qhat(pc, cal["y_aug"].to_numpy())
                sets = G.conformal_sets(p, q)
                cov = sets[np.arange(len(y_te)), y_te]
                size = sets.sum(axis=1)
                conf["overall"].append(cov.mean())
                conf["class0"].append(cov[y_te == 0].mean())
                conf["class1"].append(cov[y_te == 1].mean())
                conf["gig"].append(cov[te.gig.to_numpy() == 1].mean())
                conf["steady"].append(cov[te.gig.to_numpy() == 0].mean())
                for s in SOURCES:
                    conf[s].append(cov[(te.source == s).to_numpy()].mean())
                conf["single"].append((size == 1).mean())
                conf["both"].append((size == 2).mean())
                conf["empty"].append((size == 0).mean())
                if seed == 0:
                    idx = np.random.default_rng(0).choice(len(te), 2000, replace=False)
                    contrib = b.predict(te[feats].to_numpy(float)[idx], pred_contrib=True)[:, :-1]
                    shap_sample = dict(features=feats, values=te[feats].to_numpy(float)[idx].round(4).tolist(),
                                       contrib=contrib.round(5).tolist(),
                                       mean_abs=dict(zip(feats, np.abs(contrib).mean(axis=0).round(5).tolist())))
        p = logreg(tr, te, D.FEATS, seed)
        m = G.metrics(p, y_te)
        for k, key in (("auc", "auc"), ("ap", "pr_auc"), ("brier", "brier"), ("ece", "ece")):
            per["logreg"][k].append(m[key])
        per["logreg"]["logloss"].append(logloss(p, y_te))

        # leave one dataset out: train on two sources, test on the third
        for s in SOURCES:
            trn, tst = aug[aug.source != s], aug[aug.source == s]
            b = G.train(trn[D.FEATS], trn["y_aug"], seed=seed)
            p = b.predict(tst[D.FEATS].to_numpy(float))
            lodo[s]["auc_aug"].append(G.metrics(p, tst["y_aug"].to_numpy())["auc"])
            lodo[s]["auc_orig"].append(G.metrics(p, tst["y"].to_numpy())["auc"])
        print(f"seed {seed} [{time.time() - t0:.0f}s]", flush=True)

    summary = {k: {m: mean_sd(v) for m, v in d.items()} for k, d in per.items()}
    for s in SOURCES:
        summary[f"orig_auc_{s}"] = mean_sd(orig[s])
    aug0 = D.augment(src, seed=0)
    out = dict(
        per_seed_summary=summary,
        conformal={k: mean_sd(v) for k, v in conf.items()},
        lodo={s: {k: [round(x, 4) for x in mean_sd(v)] for k, v in d.items()} for s, d in lodo.items()},
        data=dict(
            base_rate_orig={s: float(src.y[src.source == s].mean()) for s in SOURCES},
            n={s: int((src.source == s).sum()) for s in SOURCES},
            aug_rate_gig=float(aug0.y_aug[aug0.gig == 1].mean()),
            aug_rate_steady=float(aug0.y_aug[aug0.gig == 0].mean()),
            ruin_rate=float(aug0.ruin.mean()),
            aug_rate=float(aug0.y_aug.mean()),
            orig_rate=float(aug0.y.mean()),
        ),
        seeds=list(SEEDS),
    )
    save("credit", out)
    save("credit_shap", shap_sample)
    print(f"done [{time.time() - t0:.0f}s]")


if __name__ == "__main__":
    main()
