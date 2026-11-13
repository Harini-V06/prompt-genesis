"""Out-of-scope / fairness checks of the DEPLOYED Guard (artifacts/guard_final.joblib) on human text it
never trained on: TOEFL non-native essays (n=91) and RAID adult human prose (150-1,200 words).
Reports: flagged AI (p >= 0.5) and flagged CONFIDENTLY (outside the 'insufficient evidence' band),
with Wilson CIs."""
import json, sys
import joblib, numpy as np, pandas as pd
sys.path.insert(0, "src")
from fairness import wilson
from portable import load as pload
g = pload("artifacts/guard_final.joblib"); cols = g["features"]
full = pd.read_parquet("artifacts/corpus_full.parquet")
F = pd.read_parquet("artifacts/features_stylo.parquet").merge(pd.read_parquet("artifacts/features_lm.parquet"), on="uid")
out = {}
for name, q in [("toefl_nonnative", full.dataset.eq("toefl")),
                ("adult_prose_raid", full.dataset.eq("raid") & full.label.eq(0) & full.n_words.between(150, 1200))]:
    d = full[q].merge(F, on="uid")
    p = g["iso"].predict(g["model"].predict_proba(d[cols])[:, 1])
    k1, k2, n = int((p >= .5).sum()), int((p >= .5 + g["band"]).sum()), len(d)
    out[name] = {"n": n, "flagged": k1 / n, "flagged_ci": wilson(k1, n), "flagged_confident": k2 / n, "flagged_confident_ci": wilson(k2, n)}
    print(name, out[name])
json.dump(out, open("reports/deployed_checks.json", "w"), indent=2, default=float)
