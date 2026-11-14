"""
ONE inference pipeline shared by the Streamlit app, the demo-mode cache and the final
teacher_sim evaluation, so the reported numbers come from exactly the code that runs live.

  analyze(text)  -> verdict, calibrated probability, abstention, SHAP reasons,
                    Detective settings, reconstructed prompt (topic via LLM if a key exists)
  verify(text, prompt, n=3) -> regenerations, Match Scores, floor, ceiling, position
"""
import json
import re
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
_HERE = Path(__file__).resolve().parent
# repository layout: <root>/src/*.py + <root>/artifacts/*; a flat upload (everything in one folder) also works
ART = ROOT / "artifacts" if (ROOT / "artifacts").exists() else _HERE
from normalize import normalize  # noqa: E402

MIN_WORDS = 100
_STOP = set("the of and to a in is that it for on with as was be by this are or not but at from have an they which you were".split())

# Plain-English explanations of features (used to turn SHAP values into reasons)
EXPLAIN = {
    "mtld": ("vocabulary variety", "uses a wider, more varied vocabulary", "repeats a smaller set of words"),
    "mattr50": ("vocabulary variety", "uses a wider, more varied vocabulary", "repeats a smaller set of words"),
    "word_len_mean": ("word length", "uses longer words on average", "uses shorter, everyday words"),
    "long_word_share": ("long words", "uses many long words", "uses few long words"),
    "sent_len_mean": ("sentence length", "writes longer sentences", "writes shorter sentences"),
    "sent_len_std": ("sentence-length variation", "varies sentence length a lot", "keeps sentences a similar length"),
    "sent_burstiness": ("rhythm", "has an uneven, bursty rhythm (typical of people)", "has an even, regular rhythm (typical of AI)"),
    "sent_short_share": ("short sentences", "has many very short sentences", "has few very short sentences"),
    "p_comma": ("commas", "uses many commas", "uses few commas"),
    "p_question": ("questions", "asks questions", "asks few questions"),
    "p_exclaim": ("exclamations", "uses exclamation marks", "uses no exclamation marks"),
    "p_ellipsis": ("ellipses", "uses '...' ", "avoids '...'"),
    "p_semicolon": ("semicolons", "uses semicolons", "avoids semicolons"),
    "contraction_rate": ("contractions", "uses contractions (don't, it's)", "avoids contractions"),
    "fp_sing_rate": ("first person", "talks about 'I' and 'me' a lot", "rarely says 'I'"),
    "second_person_rate": ("addressing the reader", "talks to the reader ('you')", "rarely addresses the reader"),
    "hedge_rate": ("hedging", "hedges a lot (maybe, perhaps, I think)", "rarely hedges"),
    "discourse_rate": ("signposting words", "uses many signposting words (moreover, furthermore, overall)", "uses few signposting words"),
    "booster_rate": ("emphatic words", "uses emphatic words (crucial, significant)", "uses few emphatic words"),
    "bigram_repeat_rate": ("repetition", "repeats the same word pairs", "rarely repeats word pairs"),
    "opener_diversity": ("sentence openings", "starts sentences in many different ways", "starts sentences the same way"),
    "flesch_ease": ("readability", "is easy to read", "is harder to read"),
    "fk_grade": ("reading level", "reads at a higher grade level", "reads at a lower grade level"),
    "gunning_fog": ("complexity", "is linguistically complex", "is linguistically simple"),
    "passive_rate": ("passive voice", "uses the passive voice often", "rarely uses the passive voice"),
    "subord_per_sent": ("clause complexity", "packs many clauses into each sentence", "uses simple clause structure"),
    "dep_depth_mean": ("sentence nesting", "builds deeply nested sentences", "builds flat sentences"),
    "lm_obs_ppl_log": ("predictability", "is less predictable to a language model (typical of people)", "is very predictable to a language model (typical of AI)"),
    "lm_perf_ppl_log": ("predictability", "is less predictable to a language model (typical of people)", "is very predictable to a language model (typical of AI)"),
    "lm_binoculars": ("Binoculars score", "has a high Binoculars ratio (typical of people)", "has a low Binoculars ratio (typical of AI)"),
    "lm_obs_rank_1": ("top-choice words", "often uses the single most likely next word (typical of AI)", "rarely uses the single most likely next word"),
    "lm_obs_rank_gt1000": ("surprising words", "uses many surprising, low-probability words (typical of people)", "uses few surprising words"),
    "lm_obs_logrank_mean": ("word surprise", "picks less obvious words", "picks very obvious words"),
    "lm_obs_entropy_mean": ("uncertainty", "leaves the language model uncertain what comes next", "makes the next word easy to guess"),
    "lm_obs_lp_std": ("predictability swings", "mixes predictable and surprising passages", "is uniformly predictable"),
    "emb_coh_adj_mean": ("sentence-to-sentence flow", "flows very smoothly between sentences", "jumps between ideas"),
}


def explain_feature(name, value, shap_value, median=None):
    """Plain-English reason. Direction of the sentence = is this text ABOVE or BELOW the
    training median for the feature; 'toward' = sign of its SHAP contribution."""
    toward = "AI" if shap_value > 0 else "human"
    high = median is None or value >= median
    if name in EXPLAIN:
        label, hi, lo = EXPLAIN[name]
        sent = f"The text {hi if high else lo}."
    elif name.startswith("fw_"):
        label = f"use of '{name[3:]}'"
        sent = f"It uses the word '{name[3:]}' {'more' if high else 'less'} often than typical."
    elif name.startswith("pos_"):
        pos_names = {"noun": "nouns", "verb": "verbs", "adj": "adjectives", "adv": "adverbs", "pron": "pronouns",
                     "det": "articles and determiners (the, a, this)", "adp": "prepositions (in, of, on)",
                     "cconj": "joining words (and, but, or)", "sconj": "linking words (because, although, if)",
                     "aux": "helping verbs (is, was, have, can)", "propn": "names (proper nouns)", "num": "numbers",
                     "part": "particles (to, not)", "intj": "interjections (oh, well)"}
        nm = pos_names.get(name[4:], name[4:])
        label = nm.split(" (")[0]
        sent = f"It uses {'more' if high else 'fewer'} {nm} than typical."
    elif name.startswith("p_"):
        label = name[2:].replace("_", " ")
        sent = f"Its {label} rate is {'higher' if high else 'lower'} than typical."
    else:
        label = name.replace("_", " ")
        sent = f"Its {label} is {'higher' if high else 'lower'} than typical."
    return {"feature": name, "label": label, "toward": toward, "sentence": sent,
            "value": float(value), "median": None if median is None else float(median), "shap": float(shap_value)}


def validate(text):
    words = re.findall(r"[A-Za-z']+", text)
    n = len(text.split())
    if n < MIN_WORDS:
        return f"Please paste at least {MIN_WORDS} words (you have {n}). Shorter texts do not carry enough evidence."
    low = [w.lower() for w in words]
    if not low or sum(w in _STOP for w in low) / len(low) < 0.15:
        return "This does not look like English text. Prompt Genesis only supports English."
    return None


class Models:
    def __init__(self, art=ART):
        from portable import load as pload    # XGBoost stored as portable JSON (see portable.py)
        g = pload(art / "guard_final.joblib")
        self.guard, self.iso, self.band, self.cols = g["model"], g["iso"], g["band"], g["features"]
        self.uses_lm = any(c.startswith(("lm_", "emb_")) for c in self.cols)
        self.det = {}
        for lab in ("tone", "persona", "evasion", "length"):
            p = art / f"detective_{lab}.joblib"
            if p.exists():
                self.det[lab] = pload(p)
        self.explainer = None
        med = art / "guard_feature_medians.json"
        self.medians = json.load(open(med)) if med.exists() else {}
        sg = art / "scope_gate.json"          # "is this a student essay at all?" (src/scope_gate.py)
        self.gate = json.load(open(sg)) if sg.exists() else None

    def p_out_of_scope(self, F):
        g = self.gate
        x = np.array([F.iloc[0].get(c, np.nan) for c in g["features"]], dtype=float)
        x = np.where(np.isnan(x), [g["medians"][c] for c in g["features"]], x)
        z = (x - np.array(g["mean"])) / np.array(g["scale"])
        return float(1 / (1 + np.exp(-(z @ np.array(g["coef"]) + g["intercept"]))))

    def shap_reasons(self, X, k=4):
        import shap
        if self.explainer is None:
            self.explainer = shap.TreeExplainer(self.guard)
        sv = self.explainer.shap_values(X)
        sv = sv[1] if isinstance(sv, list) else sv
        sv = np.asarray(sv)[0]
        out, seen = [], set()
        for i in np.argsort(-np.abs(sv)):          # strongest first, one reason per plain-English label
            r = explain_feature(self.cols[i], X.iloc[0, i], sv[i], self.medians.get(self.cols[i]))
            if r["label"] in seen:
                continue
            seen.add(r["label"]); out.append(r)
            if len(out) == k:
                break
        return out


def featurize(texts, uses_lm=True):
    from features_stylo import extract_stylo
    F = extract_stylo(texts)
    if uses_lm:
        from features_lm import lm_features
        F = pd.concat([F, lm_features(texts)], axis=1)
    return F


def analyze(raw_text, models, topic_fn=None, force=False):
    """force=True: give the Guard's verdict even when the scope gate says the text is not a student essay."""
    err = validate(raw_text)
    if err:
        return {"error": err}
    text = normalize(raw_text)
    F = featurize([text], models.uses_lm)
    X = F[models.cols]
    p_raw = float(models.guard.predict_proba(X)[0, 1])
    p = float(models.iso.predict([p_raw])[0])
    if abs(p - 0.5) < models.band:
        verdict = "insufficient evidence"
    else:
        verdict = "AI" if p >= 0.5 else "human"
    out = {"verdict": verdict, "p_ai": p, "p_ai_uncalibrated": p_raw, "band": models.band,
           "confidence": max(p, 1 - p), "n_words": len(text.split()), "reasons": models.shap_reasons(X)}
    if models.gate is not None:
        po = models.p_out_of_scope(F)
        out["scope"] = {"p_out": po, "threshold": models.gate["threshold"], "out": po > models.gate["threshold"]}
        if out["scope"]["out"] and not force:
            # not a student essay: the Guard is unreliable here (adult prose: 19% confident false alarms),
            # so no verdict and no reconstruction. The Guard's number is kept only for the "show anyway" view.
            out["guard_verdict"] = verdict
            out["verdict"] = "out of scope"
            return out
    if verdict == "AI" and models.det:
        settings = {}
        for lab, d in models.det.items():
            Xd = F.reindex(columns=d["features"]).copy()
            if "n_words" in d["features"]:
                Xd["n_words"] = len(text.split())
            pr = d["model"].predict_proba(Xd)[0]
            i = int(np.argmax(pr))
            settings[lab] = {"value": str(d["encoder"].classes_[i]), "prob": float(pr[i]),
                             "confident": bool(pr[i] >= d.get("conf_threshold", 0.0)),
                             "threshold": float(d.get("conf_threshold", 0.0)),
                             "all": {str(c): float(q) for c, q in zip(d["encoder"].classes_, pr)}}
        out["settings"] = settings
        topic = topic_fn(text) if topic_fn else None
        if not topic:
            from style_profile import noun_phrase_topic
            topic = noun_phrase_topic(text)
            out["topic_source"] = "keywords" if topic else "none"
        else:
            out["topic_source"] = "llm"
        out["topic"] = topic or keyword_topic(text)
        from verifier import build_prompt, build_prompt_v2
        from style_profile import measure, prompt_clauses
        # only settings the Detective is confident about go into the reconstructed prompt
        val = lambda k: settings[k]["value"] if k in settings and settings[k]["confident"] else None
        out["style"] = measure(raw_text)                 # COUNTED (raw text keeps paragraphs/headings)
        out["style_clauses"] = prompt_clauses(out["style"])
        out["prompt_v1"] = build_prompt(out["topic"], val("tone"), val("persona"), val("evasion"), val("length"))
        # The rebuilt prompt uses ONLY measured information (topic + counted style). The Detective's tone/persona
        # guesses are shown but kept OUT of the prompt: on real chatbot-app essays they were mostly wrong
        # (RESULTS 13.1: tone 6/14, persona 5/14 for ChatGPT; 0/6 for Claude), and a wrong instruction pulls
        # the rebuilt essay away from the original. Length is counted exactly instead of classified.
        out["prompt"] = build_prompt_v2(out["topic"], None, None, out["style_clauses"])
        out["unstyled"] = val("tone") in (None, "unstyled") and val("persona") in (None, "unstyled")
    return out


def keyword_topic(text, k=4):
    """Offline fallback topic: top TF-IDF-weighted content words."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    sents = [s for s in re.split(r"(?<=[.!?])\s+", text) if len(s.split()) > 3] or [text]
    v = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=500)
    try:
        X = v.fit_transform(sents)
    except ValueError:
        return "the topic of the text"
    s = np.asarray(X.sum(0)).ravel()
    terms = [t for t in v.get_feature_names_out()[np.argsort(-s)] if not t.isdigit()][:k]
    return ", ".join(terms)


def verify(original, prompt, generate_fn, wrong_prompt, n=3):
    """generate_fn(prompt) -> text. Returns regenerations, per-run Match Scores, floor, ceiling, position."""
    from verifier import match_score, place
    from concurrent.futures import ThreadPoolExecutor
    # the n rewrites + the wrong-prompt control are independent: request them in parallel
    with ThreadPoolExecutor(max_workers=n + 1) as ex:
        futs = [ex.submit(generate_fn, prompt) for _ in range(n)] + [ex.submit(generate_fn, wrong_prompt)]
        outs = [f.result() for f in futs]
    regens, wrong = [r for r in outs[:n] if r], outs[n]
    if len(regens) < 2:
        return {"error": "The LLM did not return enough regenerations (rate limit?). Try again in a minute."}
    scores = [match_score(original, r) for r in regens]
    floor = match_score(original, wrong)["match"] if wrong else None
    ceiling = match_score(regens[0], regens[1])["match"]
    agg = {k: float(np.mean([s[k] for s in scores if s.get(k) is not None]))
           for k in ("match", "words", "structure", "ideas") if any(s.get(k) is not None for s in scores)}
    return {"regenerations": regens, "scores": scores, "summary": agg, "floor": floor, "ceiling": ceiling,
            "position": place(agg["match"], floor, ceiling) if floor is not None else None,
            "wrong_prompt": wrong_prompt}
