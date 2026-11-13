"""
Interpretability + error analysis for the final Guard.

1. Global SHAP (TreeExplainer on the deployed model, essay pool): mean |SHAP| per feature
   and per feature GROUP, plus a beeswarm of the top 15. Saved to reports/.
2. Error analysis on the leave-one-generator-out OUT-OF-FOLD predictions (never training data):
   - error rate by held-out generator, true label, length bin, and prompt settings (AI side)
   - human false positives by topic and length
   - the most confident mistakes (printed with ids so they can be read), so the error modes
     can be described in words in RESULTS.md

  python src/explain.py
"""
import json
import sys
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "src")
RED = "#dc2626"
NAMES = {"sent": "sentence length/rhythm", "lex": "vocabulary", "pos": "parts of speech", "syntax": "syntax",
         "pron": "pronouns/contractions", "disc": "hedges/discourse markers", "punct": "punctuation",
         "read": "readability", "func": "function words"}


def group_of(c):
    """Same groups as features_stylo.GROUPS (used for the ablation), plus the two LM groups."""
    from features_stylo import group_of as g
    if c.startswith("emb_"):
        return "LM: coherence (MiniLM)"
    if c.startswith("lm_"):
        return "LM: predictability (gpt2)"
    return NAMES.get(g(c), g(c))


def shap_global():
    import shap
    from portable import load as pload
    art = pload("artifacts/guard_final.joblib")
    cols = art["features"]
    pool = pd.read_parquet("artifacts/corpus_matched.parquet")
    pool = pool[pool.domain == "essay"]
    F = pd.read_parquet("artifacts/features_stylo.parquet").merge(pd.read_parquet("artifacts/features_lm.parquet"), on="uid")
    X = pool.merge(F, on="uid")[cols].sample(n=min(2000, len(pool)), random_state=42)
    sv = shap.TreeExplainer(art["model"]).shap_values(X)
    sv = sv[1] if isinstance(sv, list) else sv
    imp = pd.Series(np.abs(sv).mean(0), index=cols).sort_values(ascending=False)
    grp = imp.groupby(imp.index.map(group_of)).sum().sort_values(ascending=False)
    imp.to_csv("reports/shap_feature_importance.csv", header=["mean_abs_shap"])
    grp.to_csv("reports/shap_group_importance.csv", header=["mean_abs_shap"])
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.barh(grp.index[::-1], grp.values[::-1], color=RED)
    ax.set_xlabel("sum of mean |SHAP| (log-odds)"); ax.set_title("What the Guard relies on, by feature group", fontsize=10)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    plt.tight_layout(); plt.savefig("reports/plot_shap_groups.png", dpi=160); plt.close()
    plt.figure()
    shap.summary_plot(sv, X, max_display=15, show=False)
    plt.title("Top 15 features (beeswarm): red = high value, right = pushes toward AI", fontsize=9)
    plt.tight_layout(); plt.savefig("reports/plot_shap_beeswarm.png", dpi=160); plt.close()
    print("SHAP groups:\n" + grp.round(3).to_string())
    print("top 10 features:\n" + imp.head(10).round(3).to_string())


def errors():
    P = pd.read_parquet("reports/guard_oof_logo_stylo_lm_essay_fixed.parquet")
    full = pd.read_parquet("artifacts/corpus_full.parquet",
                           columns=["uid", "n_words", "tone", "persona", "evasion", "length", "gen_kind", "prompt_type"])
    P = P.merge(full, on="uid", how="left")
    P["wrong"] = (P.p_guard >= 0.5).astype(int) != P.label
    P["len_bin"] = pd.cut(P.n_words, [0, 200, 300, 400, 600, 2000])
    out = {"overall_error": float(P.wrong.mean()), "n": int(len(P))}
    for col in ["fold", "label", "len_bin"]:
        out[f"by_{col}"] = P.groupby(col, observed=True).wrong.agg(["mean", "size"]).round(4).reset_index().astype(str).values.tolist()
    ai = P[P.label == 1]
    for col in ["tone", "persona", "evasion", "gen_kind", "prompt_type"]:
        out[f"ai_by_{col}"] = ai.groupby(col, dropna=False).wrong.agg(["mean", "size"]).round(4).reset_index().astype(str).values.tolist()
    hum = P[P.label == 0]
    out["human_fp_by_topic"] = hum.groupby("topic").wrong.agg(["mean", "size"]).round(4).sort_values("mean", ascending=False).reset_index().astype(str).values.tolist()
    P["conf"] = (P.p_guard - 0.5).abs()
    worst = P[P.wrong].sort_values("conf", ascending=False).head(12)
    out["most_confident_mistakes"] = worst[["uid", "fold", "label", "p_guard", "n_words", "evasion", "persona", "gen_kind"]].astype(str).values.tolist()
    json.dump(out, open("reports/error_analysis.json", "w"), indent=1)
    print(f"overall error {out['overall_error']:.3%} (n={out['n']})")
    for k in ("by_label", "by_len_bin", "ai_by_evasion", "ai_by_persona", "ai_by_gen_kind"):
        print(k, out[k])
    print("human FP by topic (top 5):", out["human_fp_by_topic"][:5])
    print("most confident mistakes:", *out["most_confident_mistakes"], sep="\n  ")


if __name__ == "__main__":
    errors()
    shap_global()
