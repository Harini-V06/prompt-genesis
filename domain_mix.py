"""
Threat check + mitigation: does the essay-only Guard treat polished ADULT human writing as AI?
Compares two training sets with the SAME untuned model (gate settings), so the difference is the data:
  E   = essay pool only (the deployed setting)
  E+P = essay pool + RAID prose pool (adult human prose + 11 generators' prose)
Evaluated on:
  (1) essay leave-one-generator-out accuracy (the 6 essay generators; human test topics held out)
  (2) adult human prose false-positive rate, out-of-fold (GroupKFold by RAID source over prose)
  (3) TOEFL non-native false-positive rate (n=91)
All with Wilson CIs. Prose groups never appear in both train and test.
"""
import json, sys
import numpy as np, pandas as pd
from sklearn.model_selection import GroupKFold
from xgboost import XGBClassifier
sys.path.insert(0, "src")
from fairness import wilson
XGB = dict(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
           eval_metric="logloss", n_jobs=2, random_state=42)
pool = pd.read_parquet("artifacts/corpus_matched.parquet"); full = pd.read_parquet("artifacts/corpus_full.parquet")
F = pd.read_parquet("artifacts/features_stylo.parquet").merge(pd.read_parquet("artifacts/features_lm.parquet"), on="uid")
cols = [c for c in F.columns if c != "uid"]
E = pool[pool.domain == "essay"].merge(F, on="uid"); P = pool[pool.domain == "prose"].merge(F, on="uid")
T = full[full.dataset == "toefl"].merge(F, on="uid")
P["grp"] = P.topic
print(f"essay {len(E)} | prose {len(P)} (human {int((P.label==0).sum())}) | toefl {len(T)}")

def w(df):  # balance domain x label so prose cannot swamp essays
    k = df.domain.astype(str) + df.label.astype(str); return (1.0 / k.map(k.value_counts())).values * len(df) / k.nunique()

def fit(tr):
    return XGBClassifier(**XGB).fit(tr[cols], tr.label, sample_weight=w(tr))

out = {}
for name, use_prose in (("E", False), ("E+P", True)):
    r = {}
    # (1) essay LOGO
    acc = []
    rng = np.random.default_rng(42)
    for gen in sorted(E.loc[E.label == 1, "generator"].unique()):
        te_ai = E[E.generator == gen]
        topics = E.loc[E.label == 0, "topic"].unique().copy(); rng.shuffle(topics)
        take, n = [], 0
        for t in topics:
            take.append(t); n += int(((E.label == 0) & (E.topic == t)).sum())
            if n >= len(te_ai): break
        te = pd.concat([te_ai, E[(E.label == 0) & E.topic.isin(take)]])
        tr = E[(E.generator != gen) & ~E.topic.isin(take)]
        if use_prose: tr = pd.concat([tr, P])
        p = fit(tr).predict_proba(te[cols])[:, 1]
        acc.append(((p >= .5) == te.label).values)
        r[f"logo_{gen}"] = float(acc[-1].mean())
    a = np.concatenate(acc); r["logo_pooled"] = float(a.mean()); r["logo_ci"] = wilson(int(a.sum()), len(a))
    # (2) adult human prose FPR, out-of-fold over prose groups
    ph = np.zeros(len(P))
    for tri, tei in GroupKFold(5).split(P, P.label, P.grp):
        tr = pd.concat([E, P.iloc[tri]]) if use_prose else E
        ph[tei] = fit(tr).predict_proba(P.iloc[tei][cols])[:, 1]
    hm = (P.label == 0).values; k = int((ph[hm] >= .5).sum())
    r["prose_human_fpr"] = k / hm.sum(); r["prose_human_fpr_ci"] = wilson(k, int(hm.sum()))
    r["prose_ai_recall"] = float((ph[~hm] >= .5).mean())
    # (3) TOEFL FPR
    m = fit(pd.concat([E, P]) if use_prose else E); pt = m.predict_proba(T[cols])[:, 1]
    k = int((pt >= .5).sum()); r["toefl_fpr"] = k / len(T); r["toefl_ci"] = wilson(k, len(T))
    out[name] = r
    print(f"{name:4s} essay LOGO {r['logo_pooled']:.3f} [{r['logo_ci'][0]:.3f},{r['logo_ci'][1]:.3f}] "
          f"{ {g[5:]: round(v,3) for g,v in r.items() if g.startswith('logo_') and g not in ('logo_pooled','logo_ci')} }\n"
          f"     adult human prose FPR {r['prose_human_fpr']:.3f} [{r['prose_human_fpr_ci'][0]:.3f},{r['prose_human_fpr_ci'][1]:.3f}] "
          f"(prose AI recall {r['prose_ai_recall']:.3f}) | TOEFL FPR {r['toefl_fpr']:.3f} [{r['toefl_ci'][0]:.3f},{r['toefl_ci'][1]:.3f}]")
json.dump(out, open("reports/domain_mix.json", "w"), indent=2, default=float)
