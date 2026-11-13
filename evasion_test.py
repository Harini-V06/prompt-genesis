"""
Does the "make it sound human" instruction actually change the text? Per generator.
For each AI model: Mann-Whitney U on every feature, evasion=sound_human vs evasion=none
(factorial-style prompts only; plain prompts excluded), BH-FDR within generator, rank-biserial r.
If a model's texts barely change, NO classifier can recover the setting for that model:
this separates "the Detective is weak" from "the instruction was ignored".
"""
import sys
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests
sys.path.insert(0, "src")
from detective import load

df, cols = load(sys.argv[1] if len(sys.argv) > 1 else "stylo+lm")
df = df[df.evasion.isin(["none", "sound_human"])]
if "gen_kind" in df:
    df = df[df.gen_kind.fillna("factorial") != "plain"]
rows = []
for g, d in df.groupby("generator"):
    a, b = d[d.evasion == "sound_human"], d[d.evasion == "none"]
    ps, rs = [], []
    for c in cols:
        x, y = a[c].dropna(), b[c].dropna()
        if x.nunique() < 2 and y.nunique() < 2:
            ps.append(1.0); rs.append(0.0); continue
        u, p = mannwhitneyu(x, y, alternative="two-sided")
        ps.append(p); rs.append(1 - 2 * u / (len(x) * len(y)))
    rej = multipletests(ps, method="fdr_bh")[0]
    r = np.abs(rs)
    top = [cols[i] for i in np.argsort(-r)[:3]]
    rows.append({"generator": g, "n_sound_human": len(a), "n_none": len(b), "sig_features": int(rej.sum()),
                 "of": len(cols), "max_abs_r": float(r.max()), "median_abs_r": float(np.median(r)),
                 "n_medium_effect(|r|>=0.3)": int((r >= 0.3).sum()), "top_features": ", ".join(top)})
R = pd.DataFrame(rows).sort_values("max_abs_r", ascending=False)
R.to_csv("reports/evasion_effect_by_generator.csv", index=False)
print(R.to_string(index=False))
