"""Compute stylometric features for every corpus row that a model may need and that is
not yet featurized: the matched pool, factorial essays (Detective), OOD and fairness sets,
and generated essays. Appends to artifacts/features_stylo.parquet."""
import sys; sys.path.insert(0, "src")
import pandas as pd
from features_stylo import extract_stylo
full = pd.read_parquet("artifacts/corpus_full.parquet")
pool = pd.read_parquet("artifacts/corpus_matched.parquet")
F = pd.read_parquet("artifacts/features_stylo.parquet")
want = set(pool.uid) | set(full[full.prompt_type.isin(["factorial", "ood", "fairness"]) | full.dataset.eq("generated")].uid)
need = full[full.uid.isin(want) & ~full.uid.isin(F.uid)]
print("missing rows:", len(need), need.groupby("dataset").size().to_dict(), flush=True)
if len(need):
    G = extract_stylo(need.text); G.insert(0, "uid", need.uid.values)
    pd.concat([F, G], ignore_index=True).to_parquet("artifacts/features_stylo.parquet", index=False)
    print("features_stylo now", len(F) + len(G))
