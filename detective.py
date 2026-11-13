"""
DETECTIVE: predicts the prompt settings that produced an AI essay.
One XGBoost classifier per label: tone, persona, evasion, length.
Trained on factorial AI essays only (Claude boilerplate already stripped in
build_corpus.py). 5-fold GroupKFold with groups = topic: every near-duplicate
cluster lives inside one topic, so a topic AND its clusters are always held out
together. CIs are cluster bootstrap (resample clusters, not rows), because
near-duplicates are not independent.

Length head only: n_words is allowed as a feature. "concise" vs "elaborate" IS a
length instruction, so word count is the direct evidence for the target, not a
confound (the Guard never sees length).

--tune: nested model selection. For every outer fold, a small random search over XGBoost /
LightGBM configs is scored on INNER GroupKFold(3) by topic (outer-train only); the winner is
refit on the outer-train and scored once on the outer test. Test folds never influence a choice.

Usage: python src/detective.py [--features stylo|stylo+lm] [--tune] [--save]
"""
import argparse
import json
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

sys.path.insert(0, "src")
SEED = 42
LABELS = ["tone", "persona", "evasion", "length"]
XGB = dict(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
           n_jobs=2, random_state=SEED, eval_metric="mlogloss")


def cluster_boot(y, pred, clusters, n=1000, seed=SEED):
    rng = np.random.default_rng(seed)
    cl = pd.Series(clusters)
    idx_by = cl.groupby(cl).indices
    keys = np.array(list(idx_by))
    accs = []
    for _ in range(n):
        pick = rng.choice(keys, len(keys), replace=True)
        ii = np.concatenate([idx_by[k] for k in pick])
        accs.append((y[ii] == pred[ii]).mean())
    return np.percentile(accs, [2.5, 97.5])


def configs(n=4, seed=SEED):
    r = np.random.default_rng(seed)
    out = [("xgb", dict(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8))]
    for _ in range(n):
        out.append(("xgb", dict(n_estimators=int(r.choice([200, 400, 600])), max_depth=int(r.choice([3, 4, 6])),
                                learning_rate=float(r.choice([0.03, 0.05, 0.1])), subsample=float(r.choice([0.7, 0.85, 1.0])),
                                colsample_bytree=float(r.choice([0.6, 0.8, 1.0])), min_child_weight=float(r.choice([1, 5]))))) 
        out.append(("lgbm", dict(n_estimators=int(r.choice([200, 400, 600])), num_leaves=int(r.choice([15, 31, 63])),
                                 learning_rate=float(r.choice([0.03, 0.05, 0.1])), subsample=float(r.choice([0.7, 0.85, 1.0])),
                                 subsample_freq=1, colsample_bytree=float(r.choice([0.6, 0.8, 1.0])),
                                 min_child_samples=int(r.choice([10, 30])))))
    return out


def make(kind, p):
    if kind == "xgb":
        return XGBClassifier(**p, n_jobs=2, random_state=SEED, eval_metric="mlogloss", tree_method="hist")
    from lightgbm import LGBMClassifier
    return LGBMClassifier(**p, n_jobs=2, random_state=SEED, verbose=-1)


def select(X, y, groups):
    """Inner GroupKFold(3) by topic on outer-train only; returns best (kind, params) by accuracy."""
    best, best_acc = None, -1
    for kind, p in configs():
        pr = np.zeros_like(y)
        for a, b in GroupKFold(n_splits=3).split(X, y, groups):
            pr[b] = make(kind, p).fit(X.iloc[a], y[a]).predict(X.iloc[b])
        acc = (pr == y).mean()
        if acc > best_acc:
            best, best_acc = (kind, p), acc
    return best


TARGET_SETTING_ACC = 0.80


def conf_threshold(y, proba, target=TARGET_SETTING_ACC, min_cov=0.10):
    """Smallest max-probability t such that out-of-fold accuracy on predictions with
    max prob >= t reaches `target` (with >= min_cov of essays still answered).
    The app shows a setting as 'confident' only above t; below it, the setting is
    shown as uncertain and left out of the reconstructed prompt. Chosen on OOF
    predictions only (never on a test set)."""
    conf, pred = proba.max(1), proba.argmax(1)
    for t in np.round(np.arange(0.34, 1.0, 0.01), 2):
        m = conf >= t
        if m.mean() >= min_cov and (pred[m] == y[m]).mean() >= target:
            return float(t), float(m.mean()), float((pred[m] == y[m]).mean())
    return 1.01, 0.0, float("nan")


def load(feats):
    full = pd.read_parquet("artifacts/corpus_full.parquet")
    # factorial essays + generated essays with KNOWN prompts (plain prompts carry the
    # "unstyled" label). The out-of-topic generated essays (ood_generated) stay out.
    df = full[(full.prompt_type == "factorial") |
              (full.dataset.eq("generated") & full.prompt_type.eq("generated"))].copy()
    F = pd.read_parquet("artifacts/features_stylo.parquet")
    if "lm" in feats:
        F = F.merge(pd.read_parquet("artifacts/features_lm.parquet"), on="uid")
    cols = [c for c in F.columns if c != "uid"]
    return df.merge(F, on="uid"), cols


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="stylo")
    ap.add_argument("--save", action="store_true", help="fit final models on all data and save")
    ap.add_argument("--tune", action="store_true", help="nested model selection on inner folds")
    a = ap.parse_args()
    df, cols = load(a.features)
    print(f"factorial essays: {len(df)} | effective clusters: {df.cluster_id.nunique()} | features: {len(cols)}")
    out = {}
    gkf = GroupKFold(n_splits=5)
    all_df = df
    for lab in LABELS:
        df = all_df[all_df[lab].notna()].reset_index(drop=True)   # plain prompts have no length label
        use = cols + (["n_words"] if lab == "length" else [])
        le = LabelEncoder().fit(df[lab])
        y = le.transform(df[lab])
        pred = np.zeros_like(y)
        proba = np.zeros((len(y), len(le.classes_)))
        chosen = []
        for tr, te in gkf.split(df, y, groups=df.topic):
            if a.tune:
                kind, p = select(df.iloc[tr][use].reset_index(drop=True), y[tr], df.topic.values[tr])
                clf = make(kind, p).fit(df.iloc[tr][use], y[tr]); chosen.append(kind)
            else:
                clf = XGBClassifier(**XGB).fit(df.iloc[tr][use], y[tr])
            proba[te] = clf.predict_proba(df.iloc[te][use])
            pred[te] = proba[te].argmax(1)
        acc = accuracy_score(y, pred)
        lo, hi = cluster_boot(y, pred, df.cluster_id.values)
        majority = pd.Series(y).value_counts(normalize=True).max()
        per_gen = {g: round(float((y[m] == pred[m]).mean()), 3)
                   for g, m in ((g, (df.generator == g).values) for g in df.generator.unique())}
        out[lab] = {"classes": list(le.classes_), "acc": acc, "acc_ci": [lo, hi],
                    "macro_f1": f1_score(y, pred, average="macro"), "chance_uniform": 1 / len(le.classes_),
                    "chance_majority": majority, "per_generator_acc": per_gen,
                    "confusion": confusion_matrix(y, pred).tolist(), "chosen_per_fold": chosen}
        t, cov, acc_c = conf_threshold(y, proba)
        out[lab].update(conf_threshold=t, coverage_at_threshold=cov, acc_at_threshold=acc_c)
        print(f"         confident when p >= {t:.2f}: answers {cov:.0%} of essays at {acc_c:.1%} accuracy")
        print(f"{lab:8s} acc {acc:.3f} [{lo:.3f},{hi:.3f}]  macroF1 {out[lab]['macro_f1']:.3f}  "
              f"chance {1/len(le.classes_):.3f} (majority {majority:.3f})  per-gen {per_gen}")
        if a.save:
            import joblib
            if a.tune:
                kind, p = select(df[use].reset_index(drop=True), y, df.topic.values)
                clf = make(kind, p).fit(df[use], y)
            else:
                clf = XGBClassifier(**XGB).fit(df[use], y)
            joblib.dump({"model": clf, "encoder": le, "features": use, "conf_threshold": t}, f"artifacts/detective_{lab}.joblib")
    json.dump(out, open(f"reports/detective_{a.features.replace('+', '_')}{'_tuned' if a.tune else ''}.json", "w"), indent=2, default=float)


if __name__ == "__main__":
    main()
