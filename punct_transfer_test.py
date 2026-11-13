"""
Is the punctuation-n-gram signal genuine AI style or a source artifact?
Train the punctuation probe on one slice, test on a DIFFERENT source/generator.
Artifact => collapses to ~50% (or flips). Genuine style => transfers.
"""
import sys; sys.path.insert(0, "src")
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from shortcut_probe import punct_stream

df = pd.read_parquet("artifacts/corpus_matched.parquet")
df["p"] = df.text.map(punct_stream)
H_es = df[(df.domain == "essay") & (df.label == 0)]
H_qa = df[(df.domain == "qa") & (df.label == 0)]
sl = {
    "essay(all AI)": df[df.domain == "essay"],
    "qa": df[df.domain == "qa"],
    "gpt-oss vs human-essay": pd.concat([df[df.generator == "gpt-oss-120b"], H_es.sample(1793, random_state=0)]),
    "claude-tmpl vs human-essay": pd.concat([df[df.generator == "claude-template"], H_es.sample(1582, random_state=1)]),
}
pairs = [("essay(all AI)", "qa"), ("qa", "essay(all AI)"),
         ("gpt-oss vs human-essay", "claude-tmpl vs human-essay"),
         ("claude-tmpl vs human-essay", "gpt-oss vs human-essay")]
rows = []
for a, b in pairs:
    m = make_pipeline(TfidfVectorizer(analyzer="char", ngram_range=(1, 3), min_df=5),
                      LogisticRegression(max_iter=2000)).fit(sl[a].p, sl[a].label)
    pr = m.predict_proba(sl[b].p)[:, 1]
    rows.append((a, b, accuracy_score(sl[b].label, pr > .5), roc_auc_score(sl[b].label, pr)))
out = pd.DataFrame(rows, columns=["train_on", "test_on", "acc", "auc"])
out.to_csv("reports/punct_transfer_test.csv", index=False)
print(out.round(3).to_string(index=False))
