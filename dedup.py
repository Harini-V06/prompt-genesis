"""
Near-duplicate clustering. Any two AI essays with TF-IDF cosine > THRESH on the
same canonical topic AND same generator are joined (connected components). The resulting
cluster_id is used as the GROUP in every cross-validation split, so a
near-copy of a training essay can never sit in the test fold.

TF-IDF (word 1-2 grams) is used instead of MiniLM here because the cloud dev
sandbox cannot download sentence-transformers; MiniLM agreement is checked
later on the local machine (see reports/dedup_minilm_check.txt).

Also reports EFFECTIVE sample size = number of unique clusters.
"""
import numpy as np
import pandas as pd
from scipy.sparse.csgraph import connected_components
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

THRESH = 0.85


def cluster_block(texts):
    if len(texts) == 1:
        return np.array([0]), 0.0
    X = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True).fit_transform(texts)
    S = cosine_similarity(X)
    np.fill_diagonal(S, 0)
    n_comp, lab = connected_components(S > THRESH, directed=False)
    return lab, float((S.max(axis=1) > THRESH).mean())


def main():
    df = pd.read_parquet("artifacts/corpus_full.parquet")
    df["cluster_id"] = df.uid  # default: every row its own cluster
    report = []
    ai_mask = df.label.eq(1) & (df.dataset.isin(["factorial_api", "factorial_claude", "raw_llm"])
                                | (df.dataset.eq("generated") & ~df.prompt_type.eq("hard_negative")))
    for (ds, topic), idx in df[ai_mask].groupby(["generator", "topic"]).groups.items():
        lab, frac = cluster_block(df.loc[idx, "text"].tolist())
        df.loc[idx, "cluster_id"] = [f"{ds}|{topic}|c{l}" for l in lab]
        report.append((ds, topic, len(idx), len(set(lab)), frac))
    # Hard negatives (paraphrase / light edit of a gpt-oss factorial essay) join the
    # SOURCE essay's cluster so source and paraphrase never straddle train/test.
    if "source_ref" in df:
        api = df[df.dataset.eq("factorial_api")].reset_index()
        row_to_cluster = {f"factorial_api_row{i}": c for i, c in enumerate(api.cluster_id)}
        hn = df.prompt_type.eq("hard_negative")
        df.loc[hn, "cluster_id"] = df.loc[hn, "source_ref"].map(row_to_cluster).fillna(df.loc[hn, "uid"])
    # RAID: human source + its AI continuations share source_id (topic) -> same group
    rd = df.dataset.eq("raid")
    df.loc[rd, "cluster_id"] = df.loc[rd, "topic"]
    # HC3: human + AI answers to the same question are paired -> same group
    qa = df.dataset.eq("hc3")
    df.loc[qa, "cluster_id"] = "hc3q|" + df.loc[qa, "topic"]

    rep = pd.DataFrame(report, columns=["generator", "topic", "rows", "clusters", "frac_with_neardup"])
    summ = rep.groupby("generator").agg(rows=("rows", "sum"), clusters=("clusters", "sum"),
                                      mean_frac_neardup=("frac_with_neardup", "mean"))
    print(summ.to_string())
    rep.to_csv("reports/dedup_by_topic.csv", index=False)
    df.to_parquet("artifacts/corpus_full.parquet", index=False)


if __name__ == "__main__":
    main()
