"""
Build artifacts/demo_cache.json: pre-run examples for the app's DEMO MODE (the live-demo
backup: works with no models and no API). Run on the LOCAL machine with the FINAL models
(needs the LM features' models + an API key in .env for topic + regeneration).

  python src/make_demo_cache.py

Examples (chosen to show contrast, picked deterministically, never from teacher_sim):
  1. AI essay, Gemini, plain "Write an essay about X" prompt   (what a teacher would type)
  2. AI essay, styled prompt with known settings                (Detective has ground truth)
  3. CONTROL: human student essay (PERSUADE)                    (must NOT be flagged)
  4. Out-of-distribution AI narrative (new topic, never trained)
  5. Hard negative: AI essay lightly edited to sound human
Each entry stores the text, ground truth, analysis and (for AI verdicts) verification.
"""
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    try:
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env")
    except ImportError:
        pass
    import pipeline
    from llm import LLM, wrong_prompt_for
    # examples were picked deterministically OUTSIDE the training pool (artifacts/demo_examples.json,
    # built from corpus_full: first uid per category, 180-450 words, never trained on)
    ex = json.load(open(ROOT / "artifacts" / "demo_examples.json", encoding="utf-8"))
    fast = "--fast" in sys.argv          # no API calls at all: keyword topic, no regeneration
    models = pipeline.Models()
    llm = None if fast else LLM()
    llm = llm if (llm and llm.available) else None
    print("models loaded |", "LLM: " + ("off (--fast)" if fast else ("on" if llm else "no API key found")), flush=True)

    def topic_fn(t):
        try:
            return llm.topic(t)
        except Exception as e:
            print(f"   topic via API failed ({type(e).__name__}), using keywords", flush=True)
            return None
    cache = {}
    for i, (name, r) in enumerate(ex.items(), 1):
        truth, text = r["truth"], r["text"]
        print(f"[{i}/{len(ex)}] {name}: analyzing...", flush=True)
        a = pipeline.analyze(text, models, topic_fn=topic_fn if llm else None)
        entry = {"text": text, "truth": truth, "analysis": a}
        if a.get("verdict") == "AI" and llm:
            print("   verifying: 3 regenerations + 1 wrong-prompt control (1-3 min)...", flush=True)
            try:
                entry["verify"] = pipeline.verify(text, a["prompt"], llm.chat, wrong_prompt_for(a["topic"]), n=3)
            except Exception as e:
                print("   verify failed:", type(e).__name__, "(rate limit?) - entry kept without verification")
        cache[name] = entry
        print(f"{name:55s} -> {a.get('verdict')} (P(AI)={a.get('p_ai', float('nan')):.2f})  truth: {truth[:40]}")
    json.dump(cache, open(ROOT / "artifacts" / "demo_cache.json", "w", encoding="utf-8"), indent=1, default=float)
    print("saved artifacts/demo_cache.json")


if __name__ == "__main__":
    main()
