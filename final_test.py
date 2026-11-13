"""
FINAL ONE-SHOT TESTS with the deployed models. Run ONCE, after every model choice is frozen.
Nothing here feeds back into training, tuning or thresholds.

Test sets (none used for training, tuning or threshold choice):
  ood_generated  essays from the generator families on NEW topics/genres (kind = newtopic)
  raw_llm        AI essays from the original project's separate collection (narrative etc.)
  realchat_*     essays pasted by hand from the ChatGPT and Claude apps (data/chatgpt_test/), new topics
  teacher_sim    teacher_sim/{ai,human}_essays.csv if present (column `text`)

Reports per set: n, accuracy at p >= 0.5 (all-AI sets: = recall), answered share and accuracy outside the
"insufficient evidence" band, mean P(AI), Wilson CIs; Detective settings for the real-chat essays vs the
true prompt. Writes reports/final_test.json and prints a table.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fairness import wilson  # noqa: E402


def score(texts, M):
    """Same path as the app: normalize -> stylo + LM features -> calibrated Guard."""
    import pipeline
    out = []
    for t in texts:
        a = pipeline.analyze(t, M, topic_fn=None)
        out.append(a)
    return out


def summarize(name, labels, res, band):
    p = np.array([r.get("p_ai", np.nan) for r in res]); y = np.array(labels)
    ok = ~np.isnan(p); p, y = p[ok], y[ok]
    pred = (p >= .5).astype(int); k = int((pred == y).sum()); n = len(y)
    conf = np.abs(p - .5) >= band
    kc, nc = int((pred[conf] == y[conf]).sum()), int(conf.sum())
    r = {"n": n, "acc": k / n if n else None, "acc_ci": wilson(k, n), "answered": nc / n if n else None,
         "acc_answered": kc / nc if nc else None, "acc_answered_ci": wilson(kc, nc) if nc else None,
         "mean_p_ai": float(p.mean()) if n else None, "invalid_inputs": int((~ok).sum())}
    print(f"{name:22s} n={n:4d}  acc {r['acc']:.3f} [{r['acc_ci'][0]:.3f},{r['acc_ci'][1]:.3f}]  "
          f"answered {r['answered']:.0%} at {r['acc_answered'] if nc else float('nan'):.3f}  mean P(AI) {r['mean_p_ai']:.2f}")
    return r


def cached():
    """OOD sets whose features were already extracted (same normalize + extractors as the app):
    scored with the deployed Guard directly from the feature tables. Runs anywhere."""
    from portable import load as pload
    g = pload(ROOT / "artifacts" / "guard_final.joblib")
    full = pd.read_parquet(ROOT / "artifacts" / "corpus_full.parquet")
    F = pd.read_parquet(ROOT / "artifacts" / "features_stylo.parquet").merge(
        pd.read_parquet(ROOT / "artifacts" / "features_lm.parquet"), on="uid")
    out = {}
    for name, q in [("ood_generated (new topics/genres)", full.prompt_type.eq("ood_generated")),
                    ("raw_llm (separate AI collection)", full.dataset.eq("raw_llm"))]:
        d = full[q & full.n_words.ge(100)].merge(F, on="uid", how="left")
        miss = int(d[g["features"]].isna().any(axis=1).sum())
        d = d.dropna(subset=g["features"])
        p = g["iso"].predict(g["model"].predict_proba(d[g["features"]])[:, 1])
        res = [{"p_ai": float(x)} for x in p]
        out[name] = summarize(name, d.label.tolist(), res, g["band"])
        out[name]["missing_features_dropped"] = miss
        out[name]["by_generator"] = {gg: float(np.mean(p[(d.generator == gg).values] >= .5)) for gg in sorted(d.generator.unique())}
        print("   by generator:", {k: round(v, 3) for k, v in out[name]["by_generator"].items()})
    json.dump(out, open(ROOT / "reports" / "final_test_cached.json", "w"), indent=1, default=float)


def live():
    """Hand-collected real-chatbot essays (+ teacher_sim if present) through the FULL app pipeline
    (needs the frozen LMs: run on the PC)."""
    import pipeline
    from load_chatgpt_test import load as load_realchat
    M = pipeline.Models()
    band = M.band
    sets = {}
    rc = load_realchat()
    for src, d in rc.groupby("source"):
        sets[f"realchat_{src}"] = (d.text_raw.tolist(), d.label.tolist(), [src] * len(d))
    ts = ROOT / "teacher_sim"
    if (ts / "ai_essays.csv").exists() and (ts / "human_essays.csv").exists():
        a, h = pd.read_csv(ts / "ai_essays.csv"), pd.read_csv(ts / "human_essays.csv")
        sets["teacher_sim"] = (a.text.tolist() + h.text.tolist(), [1] * len(a) + [0] * len(h), ["ai"] * len(a) + ["human"] * len(h))
    out = {}
    for name, (texts, labels, gens) in sets.items():
        res = score(texts, M)
        out[name] = summarize(name, labels, res, band)
        out[name]["by_generator"] = {g: float(np.mean([(r.get("p_ai", 0) >= .5) == y for r, y, gg in zip(res, labels, gens) if gg == g]))
                                     for g in sorted(set(gens))}
        if name.startswith("realchat"):
            d = rc[rc.source == name.split("_", 1)[1]]
            out[name]["essays"] = [{"prompt_id": int(pid), "p_ai": r.get("p_ai"), "verdict": r.get("verdict"),
                                    "settings": {k: v["value"] for k, v in r.get("settings", {}).items()},
                                    "prompt": r.get("prompt")} for pid, r in zip(d.prompt_id, res)]
    json.dump(out, open(ROOT / "reports" / "final_test_live.json", "w"), indent=1, default=float)
    print("saved reports/final_test_live.json")


if __name__ == "__main__":
    {"cached": cached, "live": live}[sys.argv[1] if len(sys.argv) > 1 else "live"]()
