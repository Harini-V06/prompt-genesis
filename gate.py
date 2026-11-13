"""
FEASIBILITY GATE: quick Guard baseline on the CLEANED (normalized, length-matched,
dedup-grouped) data, no tuning. Reports with 95% bootstrap CIs:

 (a) LOPO  leave-one-prompt-out over the 15 PERSUADE prompts (essay domain)
 (b) LOGO  leave-one-generator-out over the AI generators present
           test = held-out generator's AI rows + equal-size HELD-OUT human rows of the
           same domain (those humans removed from training). When a generator is the
           only AI source in its domain (chatgpt in QA), training uses only the other
           domain, so this fold is also a cross-domain test (reported as such).
 (c) OOD   raw_llm_essays.json (58 AI essays, topics outside the 15): detection rate

Usage: python src/gate.py [--features stylo|stylo+lm]
"""
import argparse
import json
import sys
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from xgboost import XGBClassifier

SEED = 42
DOMAIN = None
XGB = dict(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
           eval_metric="logloss", n_jobs=2, random_state=SEED)


def boot_ci(y, p, fn, n=1000, seed=SEED):
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if fn is roc_auc_score and len(set(y[i])) < 2:
            continue
        vals.append(fn(y[i], p[i]))
    return np.percentile(vals, [2.5, 97.5])


def score(y, prob):
    y, prob = np.asarray(y), np.asarray(prob)
    pred = (prob >= 0.5).astype(int)
    acc = accuracy_score(y, pred)
    lo, hi = boot_ci(y, pred, accuracy_score)
    out = {"n": len(y), "acc": acc, "acc_lo": lo, "acc_hi": hi, "f1": f1_score(y, pred, zero_division=0)}
    out["auc"] = roc_auc_score(y, prob) if len(set(y)) == 2 else np.nan
    return out


def load(feats):
    m = pd.read_parquet("artifacts/corpus_matched.parquet")
    full = pd.read_parquet("artifacts/corpus_full.parquet")
    F = pd.read_parquet("artifacts/features_stylo.parquet")
    if "lm" in feats:
        F = F.merge(pd.read_parquet("artifacts/features_lm.parquet"), on="uid")
    cols = [c for c in F.columns if c != "uid"]
    df = m.merge(F, on="uid")
    if DOMAIN:
        df = df[df.domain == DOMAIN]
    global TOEFL
    TOEFL = full[full.dataset == "toefl"].merge(F, on="uid")
    return df, full[full.dataset == "raw_llm"].merge(F, on="uid"), cols


def fit_predict(tr, te, cols):
    clf = XGBClassifier(**XGB).fit(tr[cols], tr.label)
    return clf.predict_proba(te[cols])[:, 1]


def tfidf_predict(tr, te):
    v = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=50000, sublinear_tf=True)
    m = LogisticRegression(max_iter=2000, C=1.0).fit(v.fit_transform(tr.text), tr.label)
    return m.predict_proba(v.transform(te.text))[:, 1]


def lopo(df, cols):
    ess = df[df.domain == "essay"]
    ys, ps, pb = [], [], []
    for topic in sorted(ess.topic.unique()):
        te, tr = ess[ess.topic == topic], df[df.topic != topic]
        ys += te.label.tolist(); ps += list(fit_predict(tr, te, cols)); pb += list(tfidf_predict(tr, te))
    return score(ys, ps), score(ys, pb)


def logo(df, cols, seed=SEED):
    rng = np.random.default_rng(seed)
    res = {}
    for gen in sorted(df[df.label == 1].generator.unique()):
        ai_te = df[df.generator == gen]
        dom = ai_te.domain.iloc[0]
        hum = df[(df.label == 0) & (df.domain == dom)]
        grp_col = "topic" if dom == "essay" else "cluster_id"
        groups = hum[grp_col].unique()
        rng.shuffle(groups)
        take, n = [], 0
        for g in groups:                       # hold out whole human groups until balanced
            take.append(g); n += (hum[grp_col] == g).sum()
            if n >= len(ai_te):
                break
        hum_te = hum[hum[grp_col].isin(take)]
        te = pd.concat([ai_te, hum_te])
        tr = df[(df.generator != gen) & ~df.uid.isin(hum_te.uid)]
        other_ai_in_dom = ((tr.label == 1) & (tr.domain == dom)).any()
        if not other_ai_in_dom:                # no AI left in this domain: train on other domains only
            tr = tr[tr.domain != dom]
        res[gen] = {"xgb": score(te.label, fit_predict(tr, te, cols)),
                    "tfidf_lr": score(te.label, tfidf_predict(tr, te)),
                    "cross_domain": not other_ai_in_dom}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="stylo")
    ap.add_argument("--domain", default=None, help="restrict pool to one domain (e.g. essay)")
    a = ap.parse_args()
    global DOMAIN
    DOMAIN = a.domain
    df, ood, cols = load(a.features)
    print(f"features={a.features} ({len(cols)} cols) | pool={len(df)}")
    out = {"features": a.features, "n_features": len(cols)}

    x, b = lopo(df, cols)
    out["lopo"] = {"xgb": x, "tfidf_lr": b}
    out["logo"] = logo(df, cols)
    p_ood = fit_predict(df, ood, cols)
    out["ood_raw_llm"] = {"n": len(ood), "detection_rate": float((p_ood >= 0.5).mean()),
                          "by_genre": ood.assign(p=p_ood >= 0.5).groupby("genre").p.mean().round(3).to_dict()}
    tag = a.features.replace('+', '_') + (f"_{a.domain}" if a.domain else "")
    p_t = fit_predict(df, TOEFL, cols)
    out["toefl_fpr"] = {"n": len(TOEFL), "fpr": float((p_t >= 0.5).mean())}
    json.dump(out, open(f"reports/gate_{tag}.json", "w"), indent=2, default=float)

    f = lambda s: f"acc {s['acc']:.3f} [{s['acc_lo']:.3f},{s['acc_hi']:.3f}] auc {s['auc']:.3f} n={s['n']}"
    print("(a) LOPO  XGB    :", f(x)); print("    LOPO  TFIDF  :", f(b))
    for g, r in out["logo"].items():
        tag = " (cross-domain)" if r["cross_domain"] else ""
        print(f"(b) LOGO {g:16s} XGB {f(r['xgb'])}{tag}")
        print(f"    LOGO {g:16s} TFIDF {f(r['tfidf_lr'])}")
    print(f"(f) TOEFL human false-positive rate: {out['toefl_fpr']['fpr']:.3f} (n={len(TOEFL)})")
    print(f"(c) OOD raw_llm detection rate: {out['ood_raw_llm']['detection_rate']:.3f}  {out['ood_raw_llm']['by_genre']}")


if __name__ == "__main__":
    main()
