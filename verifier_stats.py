"""Statistics for the Verifier validity test from data/verifier_validity.jsonl (the PC run writes it;
this script only reads it). Same tests as verifier_validity.py: one-sided paired Wilcoxon
(true-prompt match > wrong-prompt match) per component, BH-FDR, rank-biserial, bootstrap CI."""
import json, sys
import numpy as np
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests

path = sys.argv[1] if len(sys.argv) > 1 else "data/verifier_validity.jsonl"
recs = list({json.loads(l)["job_key"]: json.loads(l) for l in open(path, encoding="utf-8")}.values())
print(f"n essays = {len(recs)} | by family:", {f: sum(r['family'] == f for r in recs) for f in sorted({r['family'] for r in recs})})
res = {}
for comp in ("match", "words", "structure", "ideas"):
    t = np.array([np.mean([r["m_true_1"][comp], r["m_true_2"][comp]]) for r in recs if r["m_wrong"].get(comp) is not None])
    w = np.array([r["m_wrong"][comp] for r in recs if r["m_wrong"].get(comp) is not None])
    d = t - w
    p = wilcoxon(t, w, alternative="greater").pvalue
    nz = d[d != 0]; rk = np.argsort(np.argsort(np.abs(nz))) + 1
    rb = (rk[nz > 0].sum() - rk[nz < 0].sum()) / rk.sum()
    bs = [np.mean(np.random.default_rng(i).choice(d, len(d))) for i in range(2000)]
    res[comp] = dict(n=len(d), mean_true=t.mean(), mean_wrong=w.mean(), mean_diff=d.mean(),
                     diff_ci=list(np.percentile(bs, [2.5, 97.5])), p=p, rank_biserial=rb, share_true_higher=float((d > 0).mean()))
for (c, r), pb in zip(res.items(), multipletests([r["p"] for r in res.values()], method="fdr_bh")[1]):
    r["p_bh"] = pb
    print(f"{c:9s} true {r['mean_true']:.3f} vs wrong {r['mean_wrong']:.3f} | diff {r['mean_diff']:.3f} "
          f"[{r['diff_ci'][0]:.3f},{r['diff_ci'][1]:.3f}] | p_BH {pb:.1e} | r_rb {r['rank_biserial']:.2f} | true>wrong {r['share_true_higher']:.0%}")
ceil = np.array([r["m_ceiling"]["match"] for r in recs]); tru = np.array([np.mean([r["m_true_1"]["match"], r["m_true_2"]["match"]]) for r in recs])
wr = np.array([r["m_wrong"]["match"] for r in recs])
pos = np.clip((tru - wr) / np.maximum(ceil - wr, 1e-9), 0, 1)
print(f"ceiling (LLM vs itself) {ceil.mean():.3f} | true-prompt position on floor->ceiling scale: median {np.median(pos):.2f}")
res["ceiling_mean"] = float(ceil.mean()); res["position_median"] = float(np.median(pos))
json.dump(res, open("reports/verifier_validity_summary.json", "w"), indent=2, default=float)
