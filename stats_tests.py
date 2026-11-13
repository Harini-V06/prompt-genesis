"""
Hypothesis tests on stylometric features (all p-values Benjamini-Hochberg corrected
within each family of tests).

 T1  Human vs AI essays (length-matched pool): Mann-Whitney U, rank-biserial r, Cohen's d
 T2  HC3 paired by question: Wilcoxon signed-rank on per-question human vs AI means,
     matched-pairs rank-biserial
 T3  Prompt settings (gpt-oss factorial essays only; the Claude template would confound
     setting with template): Kruskal-Wallis + epsilon^2 for tone & persona (3 levels),
     Mann-Whitney for evasion & length (2 levels); post-hoc pairwise Mann-Whitney (BH)
     for the significant 3-level tests
 A   Assumption checks: D'Agostino-Pearson normality and Levene variance homogeneity,
     justifying non-parametric tests
Outputs reports/stats_*.csv and a printed summary.
"""
import itertools
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

ALPHA = 0.05


def bh(p):
    return multipletests(p, method="fdr_bh")[1]


def cohens_d(a, b):
    s = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
    return (b.mean() - a.mean()) / s if s else 0.0


def main():
    full = pd.read_parquet("artifacts/corpus_full.parquet")
    pool = pd.read_parquet("artifacts/corpus_matched.parquet")
    F = pd.read_parquet("artifacts/features_stylo.parquet")
    feats = [c for c in F.columns if c != "uid"]
    res = {}

    # T1 + assumption checks
    ess = pool[pool.domain == "essay"].merge(F, on="uid")
    rows, norm_fail, lev_fail = [], 0, 0
    for f in feats:
        h, a = ess.loc[ess.label == 0, f].values, ess.loc[ess.label == 1, f].values
        if np.std(np.r_[h, a]) == 0:
            continue
        u, p = stats.mannwhitneyu(h, a, alternative="two-sided")
        r = 1 - 2 * u / (len(h) * len(a))          # rank-biserial (+ => higher in AI)
        norm_fail += stats.normaltest(h).pvalue < ALPHA or stats.normaltest(a).pvalue < ALPHA
        lev_fail += stats.levene(h, a).pvalue < ALPHA
        rows.append((f, h.mean(), a.mean(), p, r, cohens_d(h, a)))
    t1 = pd.DataFrame(rows, columns=["feature", "human_mean", "ai_mean", "p", "rank_biserial", "cohens_d"])
    t1["p_bh"] = bh(t1.p)
    t1 = t1.reindex(t1.rank_biserial.abs().sort_values(ascending=False).index)
    t1.to_csv("reports/stats_T1_human_vs_ai_essays.csv", index=False)
    res["T1"] = (len(t1), int((t1.p_bh < ALPHA).sum()))
    res["assumptions"] = (len(t1), int(norm_fail), int(lev_fail))

    # T2 paired HC3
    qa = pool[pool.domain == "qa"].merge(F, on="uid")
    m = qa.groupby(["topic", "label"])[feats].mean()
    both = m.reset_index().groupby("topic").label.nunique()
    qs = both[both == 2].index
    H = m.xs(0, level="label").loc[qs]; A = m.xs(1, level="label").loc[qs]
    rows = []
    for f in feats:
        d = (A[f] - H[f]).values
        if np.all(d == 0):
            continue
        w, p = stats.wilcoxon(A[f].values, H[f].values, zero_method="wilcox")
        nz = d[d != 0]; rk = stats.rankdata(np.abs(nz))
        r = (rk[nz > 0].sum() - rk[nz < 0].sum()) / rk.sum()
        rows.append((f, H[f].mean(), A[f].mean(), p, r))
    t2 = pd.DataFrame(rows, columns=["feature", "human_mean", "ai_mean", "p", "matched_rank_biserial"])
    t2["p_bh"] = bh(t2.p)
    t2 = t2.reindex(t2.matched_rank_biserial.abs().sort_values(ascending=False).index)
    t2.to_csv("reports/stats_T2_hc3_paired_wilcoxon.csv", index=False)
    res["T2"] = (len(qs), len(t2), int((t2.p_bh < ALPHA).sum()))

    # T3 prompt settings (gpt-oss only)
    fac = full[(full.prompt_type == "factorial") & (full.generator == "gpt-oss-120b")].merge(F, on="uid")
    rows, post = [], []
    for setting in ["tone", "persona", "evasion", "length"]:
        levels = sorted(fac[setting].unique())
        for f in feats:
            groups = [fac.loc[fac[setting] == l, f].values for l in levels]
            if np.std(np.concatenate(groups)) == 0:
                continue
            if len(levels) > 2:
                H_, p = stats.kruskal(*groups)
                n = sum(map(len, groups))
                eff = (H_ - len(levels) + 1) / (n - len(levels))   # epsilon^2
                kind = "kruskal_eps2"
            else:
                u, p = stats.mannwhitneyu(*groups)
                eff = 1 - 2 * u / (len(groups[0]) * len(groups[1]))
                kind = "mwu_rank_biserial"
            rows.append((setting, f, kind, p, eff))
    t3 = pd.DataFrame(rows, columns=["setting", "feature", "test", "p", "effect"])
    t3["p_bh"] = t3.groupby("setting").p.transform(bh)
    t3.to_csv("reports/stats_T3_prompt_settings.csv", index=False)
    for setting in ["tone", "persona"]:
        top = t3[(t3.setting == setting) & (t3.p_bh < ALPHA)].nlargest(5, "effect")
        for f in top.feature:
            for l1, l2 in itertools.combinations(sorted(fac[setting].unique()), 2):
                p = stats.mannwhitneyu(fac.loc[fac[setting] == l1, f], fac.loc[fac[setting] == l2, f]).pvalue
                post.append((setting, f, l1, l2, p))
    pt = pd.DataFrame(post, columns=["setting", "feature", "level_a", "level_b", "p"])
    pt["p_bh"] = bh(pt.p)
    pt.to_csv("reports/stats_T3_posthoc.csv", index=False)

    n, nf, lf = res["assumptions"]
    print(f"Assumptions: non-normal in {nf}/{n} features, unequal variances in {lf}/{n} -> non-parametric tests")
    print(f"T1 human vs AI essays: {res['T1'][1]}/{res['T1'][0]} features significant after BH")
    print(t1.head(8)[["feature", "human_mean", "ai_mean", "rank_biserial", "cohens_d", "p_bh"]].round(3).to_string(index=False))
    print(f"\nT2 HC3 paired ({res['T2'][0]} questions): {res['T2'][2]}/{res['T2'][1]} significant after BH")
    print(t2.head(5)[["feature", "human_mean", "ai_mean", "matched_rank_biserial", "p_bh"]].round(3).to_string(index=False))
    print("\nT3 prompt settings (gpt-oss, n=%d): significant features per setting / strongest" % len(fac))
    for s, g in t3.groupby("setting"):
        top = g.loc[g.effect.abs().idxmax()]
        print(f"  {s:8s} {int((g.p_bh < ALPHA).sum())}/{len(g)}   strongest: {top.feature} ({top.test}={top.effect:.3f})")
    print(f"Post-hoc pairs significant: {int((pt.p_bh < ALPHA).sum())}/{len(pt)}")


if __name__ == "__main__":
    main()
