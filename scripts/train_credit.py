"""Train the deployed loan-readiness model.

Writes artifacts/credit_model.txt (LightGBM text dump) and artifacts/credit_meta.json
(features, conformal quantiles, thresholds and headline metrics).

    python scripts/train_credit.py [seed]
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from fa import data as D, gbm as G  # noqa: E402


def main(seed: int = 0):
    t0 = time.time()
    src = D.load_sources()
    aug = D.augment(src, seed=seed)
    tr, cal, te = D.split(aug, seed=seed)
    booster = G.train(tr[D.FEATS], tr["y_aug"], seed=seed)
    p_cal = booster.predict(cal[D.FEATS].to_numpy(float))
    p_te = booster.predict(te[D.FEATS].to_numpy(float))
    qhat = G.conformal_qhat(p_cal, cal["y_aug"].to_numpy())
    sets = G.conformal_sets(p_te, qhat)
    y_te = te["y_aug"].to_numpy()
    covered = sets[np.arange(len(y_te)), y_te]

    m = G.metrics(p_te, y_te)
    per_src = {s: round(G.metrics(p_te[te.source == s], te.y[te.source == s])["auc"], 4)
               for s in ("german", "gmsc", "taiwan")}
    no_vol = [f for f in D.FEATS if f not in ("income_cv", "semi_dev_ratio", "gig", "history_months")]
    b2 = G.train(tr[no_vol], tr["y_aug"], feats=no_vol, seed=seed)
    auc_novol = G.metrics(b2.predict(te[no_vol].to_numpy(float)), y_te)["auc"]

    meta = dict(
        feats=D.FEATS,
        qhat={"0": qhat[0], "1": qhat[1]},
        metrics=dict(
            auc_test_aug=round(m["auc"], 4),
            brier_test=round(m["brier"], 4),
            auc_test_original_label=round(G.metrics(p_te, te["y"].to_numpy())["auc"], 4),
            per_source_auc_original=per_src,
            n_train=len(tr), n_cal=len(cal), n_test=len(te),
            datasets={s: int((src.source == s).sum()) for s in ("gmsc", "taiwan", "german")},
            base_rate_aug=round(float(aug.y_aug.mean()), 4),
            auc_test_aug_no_volatility_features=round(auc_novol, 4),
            conformal_coverage=round(float(covered.mean()), 4),
        ),
        approve_pd=G.APPROVE_PD,
        review_pd=G.REVIEW_PD,
    )
    out = ROOT / "artifacts"
    out.mkdir(exist_ok=True)
    booster.save_model(str(out / "credit_model.txt"))
    (out / "credit_meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta["metrics"], indent=1), f"\n[{time.time() - t0:.0f}s]")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 0)
