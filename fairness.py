"""
FAIRNESS AUDIT: false-positive rate (human flagged as AI) for non-native English writers.

Data: liang_toefl_nonnative.csv: 91 TOEFL essays from Liang et al. (2023), "GPT detectors are
biased against non-native English writers" (Patterns). Their finding: perplexity-based
detectors over-flag non-native writing because it is more predictable. We add perplexity
features, so we must test exactly that risk.

Comparison group: native human essays (PERSUADE) as close in length to TOEFL as the pool allows.
TOEFL median is ~104 words; the pool's shortest native essays start at ~139 words (pool minimum
100), so we use native essays <= 200 words (the two lowest length bins). Scored OUT-OF-FOLD
(GroupKFold by topic) so no essay is judged by a model trained on it. The residual length gap
(TOEFL shorter) is stated as a limitation.
Reported: FPR with Wilson 95% CIs, Fisher exact test, n stated (n=91 is small: wide CI).

Mitigation experiments (same untuned model family as the gate so differences are attributable):
  M0  baseline: stylometric + LM features
  M1  drop LM (perplexity) features            -> does perplexity drive the bias?
  M2  add non-native human essays to training,  5-fold CV over TOEFL (each TOEFL essay is scored
      by a model that never saw it)            -> does a little in-group data fix it?
For each: TOEFL FPR, native FPR, and essay-domain AI recall (cost of the mitigation).
"""
import json
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact
from sklearn.model_selection import GroupKFold, KFold
from xgboost import XGBClassifier

SEED = 42
XGB = dict(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
           eval_metric="logloss", n_jobs=2, random_state=SEED)


def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (c - h, c + h)


def load():
    full = pd.read_parquet("artifacts/corpus_full.parquet")
    pool = pd.read_parquet("artifacts/corpus_matched.parquet")
    S = pd.read_parquet("artifacts/features_stylo.parquet")
    L = pd.read_parquet("artifacts/features_lm.parquet")
    F = S.merge(L, on="uid")
    ess = pool[pool.domain == "essay"].merge(F, on="uid")
    toefl = full[full.dataset == "toefl"].merge(F, on="uid")
    stylo = [c for c in S.columns if c != "uid"]
    lm = [c for c in L.columns if c != "uid"]
    return ess, toefl, stylo, lm


def oof_native_and_recall(ess, cols, lo, hi, extra=None):
    """Out-of-fold predictions on the essay pool; returns native-human FPR on the TOEFL-length
    band and overall AI recall."""
    p = np.zeros(len(ess))
    for tr, te in GroupKFold(n_splits=5).split(ess, ess.label, ess.topic):
        trd = ess.iloc[tr]
        if extra is not None:
            trd = pd.concat([trd, extra])
        p[te] = XGBClassifier(**XGB).fit(trd[cols], trd.label).predict_proba(ess.iloc[te][cols])[:, 1]
    band = (ess.label == 0) & ess.n_words.between(lo, hi)
    k, n = int((p[band] >= 0.5).sum()), int(band.sum())
    return k, n, float((p[ess.label == 1] >= 0.5).mean())


def run(ess, toefl, cols, name, augment=False):
    lo, hi = 0, 200   # shortest native essays available in the pool (see docstring)
    if not augment:
        pt = XGBClassifier(**XGB).fit(ess[cols], ess.label).predict_proba(toefl[cols])[:, 1]
        kn, nn, rec = oof_native_and_recall(ess, cols, lo, hi)
    else:
        pt = np.zeros(len(toefl))
        for tr, te in KFold(5, shuffle=True, random_state=SEED).split(toefl):
            trd = pd.concat([ess, toefl.iloc[tr]])
            pt[te] = XGBClassifier(**XGB).fit(trd[cols], trd.label).predict_proba(toefl.iloc[te][cols])[:, 1]
        kn, nn, rec = oof_native_and_recall(ess, cols, lo, hi, extra=toefl.sample(frac=0.8, random_state=SEED))
    kt, nt = int((pt >= 0.5).sum()), len(pt)
    odds, pv = fisher_exact([[kt, nt - kt], [kn, nn - kn]])
    r = {"model": name, "toefl_fpr": kt / nt, "toefl_ci": wilson(kt, nt), "toefl_k": kt, "toefl_n": nt,
         "native_fpr": kn / nn, "native_ci": wilson(kn, nn), "native_k": kn, "native_n": nn,
         "fisher_p": pv, "odds_ratio": odds, "ai_recall": rec, "native_len_band": [float(lo), float(hi)]}
    print(f"{name:34s} TOEFL FPR {kt}/{nt}={kt/nt:.3f} [{r['toefl_ci'][0]:.3f},{r['toefl_ci'][1]:.3f}] | "
          f"native (len {lo:.0f}-{hi:.0f}w, OOF) {kn}/{nn}={kn/nn:.3f} [{r['native_ci'][0]:.3f},{r['native_ci'][1]:.3f}] | "
          f"Fisher p={pv:.3g} | AI recall {rec:.3f}")
    return r


def main():
    ess, toefl, stylo, lm = load()
    print(f"essay pool with LM features: {len(ess)} | TOEFL: {len(toefl)} (median {toefl.n_words.median():.0f} words)")
    res = [run(ess, toefl, stylo + lm, "M0 baseline (stylo + LM)"),
           run(ess, toefl, stylo, "M1 drop LM/perplexity features"),
           run(ess, toefl, stylo + lm, "M2 + non-native data (5-fold CV)", augment=True)]
    json.dump(res, open("reports/fairness.json", "w"), indent=2, default=float)


if __name__ == "__main__":
    main()
