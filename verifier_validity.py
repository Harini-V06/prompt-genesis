"""
VERIFIER VALIDITY TEST: run on the LOCAL machine (needs the API keys in .env).

  python src/verifier_validity.py            # ~120 LLM calls, resumable

For N essays whose TRUE prompt is known (generated essays), using the SAME model that
wrote the essay (or a same-family fallback):
  true_1, true_2 = regenerate twice with the true prompt
  wrong          = regenerate once with a prompt from a different essay (different topic)
  match_true     = mean(Match(orig, true_1), Match(orig, true_2))
  match_wrong    = Match(orig, wrong)                 -> the FLOOR
  ceiling        = Match(true_1, true_2)              -> the CEILING (LLM vs itself)
Hypothesis: match_true > match_wrong (one-sided paired Wilcoxon), also per component,
Benjamini-Hochberg across components; rank-biserial effect size; bootstrap CI of the mean
difference. Every call is cached to data/verifier_validity.jsonl so reruns resume.
Cost: free-tier APIs; token totals are printed.
"""
import json
import os
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import features_lm  # noqa: E402,F401  (sets the same model cache as the LM features)
import generate_groq as G  # noqa: E402
from build_corpus import clean_generated  # noqa: E402  (strip <thought> traces / truncation, as in the corpus)
import verifier as V  # noqa: E402

N = 40
SEED = 11
OUT = ROOT / "data" / "verifier_validity.jsonl"


def main():
    try:
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env")
    except ImportError:
        pass
    from openai import OpenAI
    model_client, ranked_ids = {}, []
    for env, url in G.PROVIDERS.items():
        if os.environ.get(env):
            c = OpenAI(base_url=url, api_key=os.environ[env], timeout=240, max_retries=0)
            try:
                ids = [m.id.replace("models/", "") for m in c.models.list().data]
            except Exception as e:
                print(f"[{env}] list failed: {e}")
                continue
            ranked_ids += ids
            for m in ids:
                model_client[m] = c
    ranked = G.rank_models(ranked_ids)

    essays = [json.loads(l) for l in open(ROOT / "data" / "generated" / "groq_essays.jsonl", encoding="utf-8")]
    for e in essays:                                   # same cleaning as the training corpus
        e["text"] = clean_generated(e["text"], e.get("finish_reason"))
    essays = [e for e in essays if e["kind"] in ("plain", "factorial", "newtopic") and len(e["text"].split()) >= 100]
    rnd = random.Random(SEED)
    by_fam = {}
    for e in essays:
        by_fam.setdefault(e["family"], []).append(e)
    sample = []
    for fam in sorted(by_fam):                                   # stratified by family
        sample += rnd.sample(by_fam[fam], min(N // len(by_fam), len(by_fam[fam])))
    sample = sample[:N]

    done = {}
    if OUT.exists():
        for l in open(OUT, encoding="utf-8"):
            r = json.loads(l); done[r["job_key"]] = r
    stats = {"n": 0, "in": 0, "out": 0}
    log = lambda m: print(m, flush=True)

    for k, e in enumerate(sample):
        if e["job_key"] in done:
            continue
        others = [x for x in essays if x["topic"] != e["topic"]]
        wrong_prompt = rnd.choice(others)["prompt"]
        models = [e["generator"]] + [m for m in ranked.get(e["family"], []) if m != e["generator"]]
        models = [m for m in models if m in model_client]
        if not models:
            log(f"skip {e['job_key']}: no model for family {e['family']}"); continue
        outs = {}
        for name, prompt in (("true_1", e["prompt"]), ("true_2", e["prompt"]), ("wrong", wrong_prompt)):
            for m in models:
                text, fin = G.call(model_client[m], m, prompt, 1500, stats, log)
                text = clean_generated(text, fin) if text else text
                if text and len(text.split()) >= 50:
                    outs[name] = text; outs[name + "_model"] = m; break
        if len(outs) < 6:
            log(f"[{k+1}/{len(sample)}] incomplete (rate limits), will retry on next run"); continue
        rec = {"job_key": e["job_key"], "family": e["family"], "topic": e["topic"], "true_prompt": e["prompt"],
               "wrong_prompt": wrong_prompt, **outs,
               "m_true_1": V.match_score(e["text"], outs["true_1"]), "m_true_2": V.match_score(e["text"], outs["true_2"]),
               "m_wrong": V.match_score(e["text"], outs["wrong"]), "m_ceiling": V.match_score(outs["true_1"], outs["true_2"])}
        with open(OUT, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        done[e["job_key"]] = rec
        log(f"[{k+1}/{len(sample)}] {e['family']:14s} true {np.mean([rec['m_true_1']['match'], rec['m_true_2']['match']]):.3f}"
            f"  wrong {rec['m_wrong']['match']:.3f}  ceiling {rec['m_ceiling']['match']:.3f}")

    recs = list(done.values())
    print(f"\ncompleted {len(recs)} essays | tokens in={stats['in']:,} out={stats['out']:,}")
    if len(recs) >= 10:
        from scipy.stats import wilcoxon
        from statsmodels.stats.multitest import multipletests
        res = {}
        for comp in ("match", "words", "structure", "ideas"):
            t = np.array([np.mean([r["m_true_1"][comp], r["m_true_2"][comp]]) for r in recs if r["m_wrong"][comp] is not None])
            w = np.array([r["m_wrong"][comp] for r in recs if r["m_wrong"][comp] is not None])
            if len(t) < 10:
                continue
            d = t - w
            p = wilcoxon(t, w, alternative="greater").pvalue
            nz = d[d != 0]; rk = np.argsort(np.argsort(np.abs(nz))) + 1
            rb = (rk[nz > 0].sum() - rk[nz < 0].sum()) / rk.sum()
            bs = [np.mean(np.random.default_rng(i).choice(d, len(d))) for i in range(2000)]
            res[comp] = dict(n=len(d), mean_true=t.mean(), mean_wrong=w.mean(), mean_diff=d.mean(),
                             diff_ci=list(np.percentile(bs, [2.5, 97.5])), p=p, rank_biserial=rb,
                             share_true_higher=float((d > 0).mean()))
        ps = multipletests([r["p"] for r in res.values()], method="fdr_bh")[1]
        for (c, r), pb in zip(res.items(), ps):
            r["p_bh"] = pb
            print(f"{c:9s} true {r['mean_true']:.3f} vs wrong {r['mean_wrong']:.3f} | diff {r['mean_diff']:.3f} "
                  f"[{r['diff_ci'][0]:.3f},{r['diff_ci'][1]:.3f}] | p_BH {pb:.2e} | r_rb {r['rank_biserial']:.2f} "
                  f"| true>wrong in {r['share_true_higher']:.0%}")
        ceil = np.array([r["m_ceiling"]["match"] for r in recs])
        print(f"ceiling (LLM vs itself) mean {ceil.mean():.3f}")
        json.dump(res, open(ROOT / "data" / "verifier_validity_summary.json", "w"), indent=2, default=float)


if __name__ == "__main__":
    main()
