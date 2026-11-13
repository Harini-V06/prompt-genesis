"""
VERIFIER: "rebuild the crime". Send the reconstructed prompt to an LLM, regenerate the
essay, and measure how close the regeneration is to the original.

Match Score (0-1) = mean of three components:
  words      0.5 * TF-IDF cosine (word 1-2 grams) + 0.5 * ROUGE-L F1
  structure  cosine similarity of the two texts' stylometric vectors, z-scored with the
             training-set mean/std (artifacts/stylo_scaler.json), mapped to 0-1
  ideas      sentence-level point matching with MiniLM embeddings: every sentence is matched
             to its most similar sentence in the other text; F1 of the two mean best-match
             similarities (if MiniLM is unavailable, ideas = None and the score uses words+structure)

Placement on a scale:
  floor    = Match(original, regeneration from a deliberately WRONG prompt)
  ceiling  = Match(true-prompt regeneration #1, true-prompt regeneration #2)
             i.e. how similar the LLM is to ITSELF under the same prompt; an original can
             rarely beat this
  position = (match - floor) / (ceiling - floor), clipped to [0, 1]

All text is passed through normalize() first, exactly like the Guard.
"""
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from normalize import normalize  # noqa: E402

_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")
_EMB = None
_SCALER = None


def _embedder():
    global _EMB
    if _EMB is None:
        try:
            from sentence_transformers import SentenceTransformer
            _EMB = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        except Exception:
            _EMB = False
    return _EMB or None


def _scaler():
    global _SCALER
    if _SCALER is None:
        p = ROOT / "artifacts" / "stylo_scaler.json"
        p = p if p.exists() else Path(__file__).resolve().parent / "stylo_scaler.json"   # flat layout
        _SCALER = json.load(open(p))
    return _SCALER


def words_sim(a, b):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    from rouge_score import rouge_scorer
    X = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True).fit_transform([a, b])
    tf = float(cosine_similarity(X[0], X[1])[0, 0])
    rl = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True).score(a, b)["rougeL"].fmeasure
    return 0.5 * tf + 0.5 * rl, {"tfidf_cosine": tf, "rougeL_f1": rl}


def structure_sim(a, b, feats=None):
    from features_stylo import extract_stylo
    sc = _scaler()
    F = feats if feats is not None else extract_stylo([a, b])
    cols = sc["columns"]
    Z = (F[cols].values - np.array(sc["mean"])) / np.array(sc["std"])
    Z = np.clip(Z, -5, 5)
    cos = float(Z[0] @ Z[1] / (np.linalg.norm(Z[0]) * np.linalg.norm(Z[1]) + 1e-9))
    return (cos + 1) / 2, {"style_cosine": cos}


def ideas_sim(a, b):
    m = _embedder()
    if m is None:
        return None, {}
    sa = [s for s in _SENT.split(a) if len(s.split()) >= 3] or [a]
    sb = [s for s in _SENT.split(b) if len(s.split()) >= 3] or [b]
    A = m.encode(sa, normalize_embeddings=True, show_progress_bar=False)
    B = m.encode(sb, normalize_embeddings=True, show_progress_bar=False)
    S = A @ B.T
    p, r = float(S.max(1).mean()), float(S.max(0).mean())     # original->regen, regen->original
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return f1, {"ideas_precision": p, "ideas_recall": r}


def match_score(original, regenerated):
    a, b = normalize(original), normalize(regenerated)
    w, wd = words_sim(a, b)
    s, sd = structure_sim(a, b)
    i, idd = ideas_sim(a, b)
    parts = [w, s] + ([i] if i is not None else [])
    return {"match": float(np.mean(parts)), "words": w, "structure": s, "ideas": i, **wd, **sd, **idd}


def place(match, floor, ceiling):
    if ceiling <= floor:
        return None
    return float(np.clip((match - floor) / (ceiling - floor), 0, 1))


def build_prompt(topic, tone=None, persona=None, evasion=None, length=None):
    """Readable reconstructed prompt; None / 'unstyled' settings are simply omitted."""
    parts = [f"Write an essay about {topic}."]
    if tone in ("formal", "casual", "academic"):
        parts.append(f"Use a {tone} tone.")
    if persona == "student":
        parts.append("Write it as a student.")
    elif persona == "non-native speaker":
        parts.append("Write as a non-native English speaker would.")
    elif persona == "expert":
        parts.append("Write as an expert on the subject.")
    if evasion == "sound_human":
        parts.append("Make it sound like a human wrote it, not an AI.")
    if length == "concise":
        parts.append("Keep it to about 200 words.")
    elif length == "elaborate":
        parts.append("Aim for about 500 words.")
    return " ".join(parts)


def save_scaler():
    """Training-set mean/std of stylometric features (z-scoring for the structure score)."""
    import pandas as pd
    pool = pd.read_parquet(ROOT / "artifacts" / "corpus_matched.parquet")
    F = pd.read_parquet(ROOT / "artifacts" / "features_stylo.parquet")
    F = F[F.uid.isin(pool.uid)]
    cols = [c for c in F.columns if c != "uid"]
    sd = F[cols].std().replace(0, 1.0)
    json.dump({"columns": cols, "mean": F[cols].mean().tolist(), "std": sd.tolist()},
              open(ROOT / "artifacts" / "stylo_scaler.json", "w"))
    print("saved stylo_scaler.json", len(cols), "features")


if __name__ == "__main__":
    save_scaler()
