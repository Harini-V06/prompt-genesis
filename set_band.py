"""Deployment abstention band for the Guard ("insufficient evidence").
The nested inner-OOF band (guard.py) is 0.00: inside training the model is almost never wrong, so it
would never abstain. The deployment band is instead set on the cross-validated leave-one-generator-out
predictions (each made by a model that never saw that generator), with the target fixed IN ADVANCE:
abstain on the 10% least-confident essays. This is a choice about the deployed app, not a reported
test result; RESULTS.md reports the whole accuracy-vs-coverage curve, not a cherry-picked point."""
import joblib, json
import numpy as np, pandas as pd
import sys
P = pd.read_parquet(sys.argv[1] if len(sys.argv) > 1 else "reports/guard_oof_logo_stylo_lm_essay.parquet")
c = (P.p_guard - 0.5).abs()
band = float(np.quantile(c, 0.10))
ok = (P.p_guard >= 0.5) == P.label
g = joblib.load("artifacts/guard_final.joblib")
g["band_inner"] = g["band"]; g["band"] = band; g["band_rule"] = "10% least-confident LOGO OOF"
joblib.dump(g, "artifacts/guard_final.joblib")
r = {"band": band, "abstain_if_p_between": [0.5 - band, 0.5 + band], "coverage": float((c >= band).mean()),
     "acc_answered": float(ok[c >= band].mean()), "acc_in_band": float(ok[c < band].mean())}
json.dump(r, open("reports/guard_band.json", "w"), indent=2); print(r)
