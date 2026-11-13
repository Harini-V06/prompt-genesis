"""
Length matching (removes the length confound: raw PERSUADE ~382 words median vs
AI essays ~160-330). Within each domain and each word-count bin, keep equally
many human and AI texts (random, seeded). Texts under MIN_WORDS are dropped
(the app also refuses them); 100-250 word texts are explicitly kept so short
inputs are represented. Raw length is NEVER used as a model feature.

Output: artifacts/corpus_matched.parquet (the training/CV pool)
OOD (raw_llm) and fairness (toefl) rows are NOT in this pool.
"""
import numpy as np
import pandas as pd

SEED = 42
MIN_WORDS, MAX_WORDS = 100, 1200
BINS = [100, 150, 200, 250, 300, 350, 400, 500, 600, 800, 1201]
QA_CAP_PER_BIN = 500   # HC3 is huge and is the domain least like essays; cap keeps LM (CPU) extraction tractable


def match(df, cap=None, ratio=1, seed=SEED, prefer=frozenset()):
    """Per length bin keep n humans and ratio*n AI texts. With ratio>1 the class COUNTS
    differ but the length DISTRIBUTIONS are identical, so length still carries no label
    information (count imbalance is handled by sample weights).
    Within a bin, rows in `prefer` (already LM-featurized, themselves a random sample) are
    taken first, so adding data does not discard expensive features; order is otherwise
    a seeded random permutation."""
    rng = np.random.default_rng(seed)

    def order(ix):
        ix = np.array(ix)
        perm = ix[rng.permutation(len(ix))]
        pref = np.array([df.at[i, "uid"] in prefer for i in perm], bool)
        return list(perm[pref]) + list(perm[~pref])

    keep = []
    for b, g in df.groupby("len_bin", observed=True):
        h, a = g[g.label == 0], g[g.label == 1]
        n = min(len(h), len(a) // ratio)
        if cap:
            n = min(n, cap)
        if n == 0:
            continue
        keep += order(h.index)[:n] + order(a.index)[:n * ratio]
    return df.loc[sorted(keep)]


def main():
    df = pd.read_parquet("artifacts/corpus_full.parquet")
    pool = df[df.prompt_type.isin(["qa", "human_essay", "factorial", "raid", "generated", "hard_negative"])]
    pool = pool[pool.n_words.between(MIN_WORDS, MAX_WORDS)].copy()
    pool["len_bin"] = pd.cut(pool.n_words, BINS, right=False)

    from pathlib import Path
    lm_done = set()
    if Path("artifacts/texts_for_lm.parquet").exists():   # rows queued/scored in the first LM batch
        lm_done = set(pd.read_parquet("artifacts/texts_for_lm.parquet").uid)
    essay = match(pool[pool.domain == "essay"], prefer=lm_done)
    qa = match(pool[pool.domain == "qa"], cap=QA_CAP_PER_BIN, prefer=lm_done)
    # RAID: 900 human vs 9,900 AI -> up to 4 AI per human in every length bin
    prose = match(pool[pool.domain == "prose"], ratio=4, prefer=lm_done)
    out = pd.concat([essay, qa, prose]).drop(columns="len_bin")
    out.to_parquet("artifacts/corpus_matched.parquet", index=False)

    print("matched pool:", len(out))
    print(out.groupby(["domain", "generator"]).agg(n=("uid", "size"), med_words=("n_words", "median"),
          clusters=("cluster_id", "nunique")).to_string())
    print("\nmedian words by domain/label:")
    print(out.groupby(["domain", "label"]).n_words.median().unstack().to_string())


if __name__ == "__main__":
    main()
