"""
GUARD: human vs AI, with nested tuning, calibration and abstention.

OUTER protocols (test data never touches any choice):
  lopo  leave-one-prompt-out over the 15 PERSUADE prompts (essay domain)
  logo  leave-one-generator-out (test = held-out generator's AI + held-out human groups
        of the same domain; those humans are removed from training)
INNER (on outer-train only):
  - random search over XGBoost and LightGBM configs, 3-fold GroupKFold (groups = topic,
    which also keeps every near-duplicate cluster together); best model type+config wins
  - isotonic calibration fitted on the inner out-of-fold predictions
  - abstention band [0.5-d, 0.5+d]: smallest d whose inner-OOF accuracy on answered
    cases reaches TARGET_ACC
  - domain x label balanced sample weights
Baselines (outer, same folds): TF-IDF + LogReg; perplexity threshold (if LM features).

Usage:
  python src/guard.py --protocol lopo --features stylo
  python src/guard.py --protocol logo --features stylo+lm
  python src/guard.py --final --features stylo+lm      # fit + save deployable model
"""
import argparse
import json
import time
import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import GroupKFold
from xgboost import XGBClassifier

SEED = 42
TARGET_ACC = 0.90
N_CONFIGS = 6
rng_global = np.random.default_rng(SEED)


FIXED = False   # --fixed: no search, one pre-set config (the untuned gate model)


def configs(n=N_CONFIGS, seed=SEED):
    if FIXED:
        return [("xgb", dict(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8))]
    r = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        out.append(("xgb", dict(n_estimators=int(r.choice([200, 400, 600])), max_depth=int(r.choice([3, 4, 6])),
                                learning_rate=float(r.choice([0.03, 0.05, 0.1])), subsample=float(r.choice([0.7, 0.85, 1.0])),
                                colsample_bytree=float(r.choice([0.6, 0.8, 1.0])), min_child_weight=float(r.choice([1, 5, 10])),
                                reg_lambda=float(r.choice([1, 5, 10])))))
        out.append(("lgbm", dict(n_estimators=int(r.choice([200, 400, 600])), num_leaves=int(r.choice([15, 31, 63])),
                                 learning_rate=float(r.choice([0.03, 0.05, 0.1])), subsample=float(r.choice([0.7, 0.85, 1.0])),
                                 subsample_freq=1, colsample_bytree=float(r.choice([0.6, 0.8, 1.0])),
                                 min_child_samples=int(r.choice([10, 30, 60])), reg_lambda=float(r.choice([0, 1, 5])))))
    return out


def make(kind, params):
    if kind == "xgb":
        return XGBClassifier(**params, eval_metric="logloss", n_jobs=2, random_state=SEED, tree_method="hist")
    return LGBMClassifier(**params, n_jobs=2, random_state=SEED, verbose=-1)


def weights(df):
    cnt = df.groupby(["domain", "label"]).uid.transform("size")
    w = 1.0 / cnt
    return (w / w.mean()).values


def ece(y, p, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    return float(sum(abs(y[idx == b].mean() - p[idx == b].mean()) * (idx == b).mean()
                     for b in range(bins) if (idx == b).any()))


def pick_band(y, p, target=TARGET_ACC):
    """Smallest half-width d so that accuracy on |p-0.5| >= d reaches target."""
    for d in np.arange(0, 0.5, 0.01):
        m = np.abs(p - 0.5) >= d
        if m.sum() >= max(20, 0.05 * len(y)) and ((p[m] >= 0.5) == y[m]).mean() >= target:
            return float(d)
    return 0.49


def inner_splits(tr, seed=SEED):
    """Validation that mirrors deployment: inner leave-one-GENERATOR-out when the training
    fold has >= 3 AI generators (val = one generator's AI rows + an equal number of held-out
    human topic groups; those humans are removed from the inner train). Otherwise falls back
    to 3-fold GroupKFold by topic. Tuning/calibrating on known-generator folds made the model
    overconfident on unseen generators (reports/guard_logo_stylo_lm_essay.json)."""
    gens = [x for x in tr.loc[tr.label == 1, "generator"].value_counts().index]
    if len(gens) < 3 or INNER == "gkf":
        return list(GroupKFold(n_splits=3).split(tr, tr.label.values, tr.grp.values))
    r = np.random.default_rng(seed)
    idx = np.arange(len(tr))
    splits = []
    for gen in gens:
        ai = idx[(tr.generator == gen).values]
        dom = tr.domain.values[ai[0]]
        hum_groups = np.array(tr.loc[(tr.label == 0) & (tr.domain == dom), "grp"].unique(), dtype=object)
        r.shuffle(hum_groups)
        take, n = set(), 0
        for gg in hum_groups:
            take.add(gg); n += int(((tr.grp == gg) & (tr.label == 0)).sum())
            if n >= len(ai):
                break
        hum = idx[((tr.label == 0) & tr.grp.isin(take)).values]
        val = np.concatenate([ai, hum])
        trn = np.setdiff1d(idx, val)
        splits.append((trn, val))
    return splits


def fit_tuned(tr, cols, log=print):
    """Inner random search + isotonic calibration + abstention band, on outer-train only."""
    y, w, g = tr.label.values, weights(tr), tr.grp.values
    inner = inner_splits(tr)
    best = (None, -1, None)
    covered = np.zeros(len(tr), bool)
    for _, b in inner:
        covered[b] = True
    for kind, params in configs():
        oof = np.full(len(tr), np.nan)
        for a, b in inner:
            m = make(kind, params).fit(tr.iloc[a][cols], y[a], sample_weight=w[a])
            oof[b] = m.predict_proba(tr.iloc[b][cols])[:, 1]
        auc = roc_auc_score(y[covered], oof[covered])
        if auc > best[1]:
            best = ((kind, params), auc, oof)
    (kind, params), auc, oof = best
    iso = IsotonicRegression(out_of_bounds="clip").fit(oof[covered], y[covered])
    band = pick_band(y[covered], iso.predict(oof[covered]))
    model = make(kind, params).fit(tr[cols], y, sample_weight=w)
    return {"kind": kind, "params": params, "inner_auc": auc, "model": model, "iso": iso, "band": band}


def metrics(y, p, band=None):
    y, p = np.asarray(y), np.asarray(p)
    pred = (p >= 0.5).astype(int)
    out = {"n": int(len(y)), "acc": accuracy_score(y, pred), "precision": precision_score(y, pred, zero_division=0),
           "recall": recall_score(y, pred, zero_division=0), "f1": f1_score(y, pred, zero_division=0),
           "confusion": confusion_matrix(y, pred, labels=[0, 1]).tolist()}
    if len(set(y)) == 2:
        out.update(roc_auc=roc_auc_score(y, p), pr_auc=average_precision_score(y, p), ece=ece(y, p))
    r = np.random.default_rng(SEED)
    accs = [accuracy_score(y[i], pred[i]) for i in (r.integers(0, len(y), len(y)) for _ in range(1000))]
    out["acc_ci"] = list(np.percentile(accs, [2.5, 97.5]))
    if band is not None:
        m = np.abs(p - 0.5) >= band
        out["abstain_band"] = band
        out["coverage"] = float(m.mean())
        out["acc_answered"] = float((pred[m] == y[m]).mean()) if m.any() else float("nan")
    return out


def tfidf_lr(tr, te):
    v = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=50000, sublinear_tf=True)
    m = LogisticRegression(max_iter=2000).fit(v.fit_transform(tr.text), tr.label, sample_weight=weights(tr))
    return m.predict_proba(v.transform(te.text))[:, 1]


def ppl_threshold(tr, te, col="lm_obs_ppl_log"):
    """Zero-feature-engineering baseline: low perplexity => AI; threshold fit on train only."""
    cands = np.quantile(tr[col], np.linspace(0.02, 0.98, 97))
    best = max(cands, key=lambda t: accuracy_score(tr.label, (tr[col] < t).astype(int)))
    return (te[col] < best).astype(float).values


DOMAIN = None
INNER = "logo"   # ablation switch: "gkf" = old inner 3-fold GroupKFold by topic


def SELF_REF():
    """uid -> the id hard negatives use to point at their source ("factorial_api_row<i>", i = row in
    factorial_essays_real_api_final.csv; build_corpus.py keeps all rows in file order)."""
    full = pd.read_parquet("artifacts/corpus_full.parquet", columns=["uid", "dataset", "text_raw"])
    fa = full[full.dataset == "factorial_api"].reset_index(drop=True)
    a = pd.read_csv("data/factorial_essays_real_api_final.csv", usecols=["text"])
    assert len(fa) == len(a) and (fa.text_raw.values == a.text.values).all(), "row order changed"
    return pd.Series([f"factorial_api_row{i}" for i in range(len(fa))], index=fa.uid.values)


def load(feats):
    pool = pd.read_parquet("artifacts/corpus_matched.parquet")
    if DOMAIN:
        pool = pool[pool.domain == DOMAIN]
    F = pd.read_parquet("artifacts/features_stylo.parquet")
    if "lm" in feats:
        F = F.merge(pd.read_parquet("artifacts/features_lm.parquet"), on="uid")
    cols = [c for c in F.columns if c != "uid"]
    df = pool.merge(F, on="uid")
    df["grp"] = df.topic
    # id each row would be referred to by in source_ref (generated hard negatives point at
    # "factorial_api_row<i>" etc.); built in build_corpus.py as the `ref` of the source row
    full = pd.read_parquet("artifacts/corpus_full.parquet", columns=["uid", "source_ref"])
    df["source_ref"] = df.uid.map(full.set_index("uid").source_ref)
    df["source_ref_self"] = df.uid.map(SELF_REF())
    return df, cols


def folds(df, protocol):
    if protocol == "lopo":
        for t in sorted(df[df.domain == "essay"].topic.unique()):
            yield t, df[df.topic != t], df[(df.topic == t)], False
    else:
        r = np.random.default_rng(SEED)
        for gen in sorted(df[df.label == 1].generator.unique()):
            ai = df[df.generator == gen]
            dom = ai.domain.iloc[0]
            hum = df[(df.label == 0) & (df.domain == dom)]
            groups = np.array(hum.grp.unique(), dtype=object); r.shuffle(groups)
            take, n = [], 0
            for g in groups:
                take.append(g); n += (hum.grp == g).sum()
                if n >= len(ai):
                    break
            hte = hum[hum.grp.isin(take)]
            # hard negatives (another model paraphrasing / lightly editing an essay) carry their SOURCE
            # essay's content: if the source was written by the held-out generator, drop them from
            # training too, otherwise the held-out essays' content leaks in via the paraphrase.
            src_held = df.source_ref.notna() & df.source_ref.isin(df.loc[df.generator == gen, "source_ref_self"].dropna())
            tr = df[(df.generator != gen) & ~df.grp.isin(take) & ~src_held]
            cross = not ((tr.label == 1) & (tr.domain == dom)).any()
            if cross:
                tr = tr[tr.domain != dom]
            yield gen, tr, pd.concat([ai, hte]), cross


def evaluate(protocol, feats):
    df, cols = load(feats)
    t0 = time.time()
    rows, per_fold = [], {}
    for name, tr, te, cross in folds(df, protocol):
        f = fit_tuned(tr, cols)
        p = f["iso"].predict(f["model"].predict_proba(te[cols])[:, 1])
        base = tfidf_lr(tr, te)
        ppl = ppl_threshold(tr, te) if "lm_obs_ppl_log" in cols else np.full(len(te), np.nan)
        per_fold[name] = {"guard": metrics(te.label, p, f["band"]), "tfidf_lr": metrics(te.label, base),
                          "chosen": f["kind"], "inner_auc": f["inner_auc"], "cross_domain": cross}
        if not np.isnan(ppl).all():
            per_fold[name]["ppl_threshold"] = metrics(te.label, ppl)
        rows.append(te[["uid", "label", "domain", "generator", "topic"]].assign(
            fold=name, p_guard=p, p_tfidf=base, p_ppl=ppl, band=f["band"]))
        g = per_fold[name]["guard"]
        print(f"[{protocol}] {str(name)[:28]:28s} {f['kind']:4s} acc {g['acc']:.3f} auc {g.get('roc_auc', float('nan')):.3f} "
              f"| abstain cov {g['coverage']:.2f} acc@cov {g['acc_answered']:.3f} | tfidf {per_fold[name]['tfidf_lr']['acc']:.3f}"
              f"{' (cross-domain)' if cross else ''}  [{(time.time()-t0)/60:.1f}m]", flush=True)
    P = pd.concat(rows)
    tag = feats.replace('+', '_') + (f"_{DOMAIN}" if DOMAIN else "") + ("_innergkf" if INNER == "gkf" else "") + ("_fixed" if FIXED else "")
    P.to_parquet(f"reports/guard_oof_{protocol}_{tag}.parquet", index=False)
    pooled = {"guard": metrics(P.label, P.p_guard), "tfidf_lr": metrics(P.label, P.p_tfidf)}
    m = np.abs(P.p_guard - 0.5) >= P.band
    pooled["guard"]["coverage"] = float(m.mean())
    pooled["guard"]["acc_answered"] = float(((P.p_guard[m] >= 0.5) == P.label[m]).mean())
    if P.p_ppl.notna().all():
        pooled["ppl_threshold"] = metrics(P.label, P.p_ppl)
    out = {"protocol": protocol, "features": feats, "n_features": len(cols), "pooled": pooled, "per_fold": per_fold}
    json.dump(out, open(f"reports/guard_{protocol}_{tag}.json", "w"), indent=2, default=float)
    g = pooled["guard"]
    print(f"POOLED {protocol}: acc {g['acc']:.3f} [{g['acc_ci'][0]:.3f},{g['acc_ci'][1]:.3f}] auc {g['roc_auc']:.3f} "
          f"pr_auc {g['pr_auc']:.3f} ece {g['ece']:.3f} | answered {g['coverage']:.2f} at acc {g['acc_answered']:.3f} "
          f"| tfidf acc {pooled['tfidf_lr']['acc']:.3f}")


def final(feats):
    """Fit the deployable Guard on the whole pool (same nested-style tuning on inner
    folds), save it in the format src/pipeline.py loads, plus per-feature training
    medians used to phrase the plain-English SHAP reasons."""
    df, cols = load(feats)
    f = fit_tuned(df, cols)
    art = {"model": f["model"], "iso": f["iso"], "band": f["band"], "features": cols, "kind": f["kind"],
           "params": f["params"], "inner_auc": f["inner_auc"], "feature_set": feats, "domain": DOMAIN,
           "n_train": len(df), "fixed_config": FIXED}
    tag = feats.replace('+', '_') + (f"_{DOMAIN}" if DOMAIN else "") + ("_fixed" if FIXED else "")
    joblib.dump(art, f"artifacts/guard_{tag}.joblib")
    joblib.dump(art, "artifacts/guard_final.joblib")
    from portable import to_portable
    to_portable(f"artifacts/guard_{tag}.joblib"); to_portable("artifacts/guard_final.joblib")
    json.dump({c: float(df[c].median()) for c in cols}, open("artifacts/guard_feature_medians.json", "w"))
    print(f"saved final guard ({tag}): {f['kind']} inner AUC {f['inner_auc']:.3f} band {f['band']:.2f} n={len(df)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", choices=["lopo", "logo"], default="lopo")
    ap.add_argument("--features", default="stylo")
    ap.add_argument("--final", action="store_true")
    ap.add_argument("--domain", default=None)
    ap.add_argument("--inner", choices=["logo", "gkf"], default="logo")
    ap.add_argument("--fixed", action="store_true", help="no hyper-parameter search (single preset config)")
    a = ap.parse_args()
    DOMAIN, INNER, FIXED = a.domain, a.inner, a.fixed
    final(a.features) if a.final else evaluate(a.protocol, a.features)
