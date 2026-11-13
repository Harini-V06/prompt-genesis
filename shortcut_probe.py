"""
SHORTCUT PROBE: can human vs AI be predicted from formatting/encoding alone?
If yes, the dataset leaks its source through format, and any detector score is
inflated. Run on RAW text (before) and NORMALIZED text (after), per domain,
on the length-matched pool (classes balanced -> chance = 50%).

Probe A (format-only): non-ASCII, newlines, tokenization artifacts, spacing,
  quote style, URL tokens, markdown. Pure encoding/layout, no writing style.
Probe B (punctuation char n-grams): letters removed, digits -> 0, char 1-3
  grams of the remaining punctuation/space stream. This mixes real style
  (comma habits) with formatting, so it is reported but interpreted with care.

CV: 5-fold GroupKFold, groups = topic (essays) / HC3 question (qa).
Outputs reports/shortcut_probe.csv and prints a summary.
"""
import re
import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.pipeline import make_pipeline

sys.path.insert(0, "src")
from normalize import normalize

FORMAT_PATTERNS = {
    "non_ascii": r"[^\x00-\x7f]",
    "newline": r"\n",
    "double_space": r"  ",
    "space_before_punct": r" [,.;:!?]",
    "spaced_nt": r" n't",
    "spaced_clitic": r" '(s|re|ve|ll|d|m)\b",
    "missing_space_after_period": r"[a-z][.!?][A-Z]",
    "url_token": r"URL_\d|https?://",
    "curly_quote": r"[‘’“”]",
    "unicode_dash": r"[‐-―]",
    "markdown": r"\*\*|^#|^\s*[-*] ",
    "straight_dquote": r'"',
    "ellipsis": r"\.\.\.",
    "spaced_hyphen": r" - ",
    # NOTE: apostrophe rate is deliberately NOT here. It is ~entirely
    # contractions/possessives, a genuine stylometric trait the spec lists as a
    # legitimate feature, not a formatting artifact.
}


def format_feats(texts):
    rows = []
    for t in texts:
        n = max(len(t), 1)
        rows.append([len(re.findall(p, t, flags=re.M)) / n * 1000 for p in FORMAT_PATTERNS.values()])
    return np.array(rows)


def punct_stream(t):
    t = re.sub(r"[A-Za-z]+", "", t)
    t = re.sub(r"\d", "0", t)
    return re.sub(r"\s+", " ", t)


def run(df, text_col):
    res = {}
    for domain, g in df.groupby("domain"):
        y, groups = g.label.values, g.topic.values
        cv = GroupKFold(n_splits=5)
        Xa = format_feats(g[text_col].tolist())
        res[(domain, "A_format")] = cross_val_score(
            HistGradientBoostingClassifier(random_state=0), Xa, y, groups=groups, cv=cv).mean()
        pb = make_pipeline(TfidfVectorizer(analyzer="char", ngram_range=(1, 3), min_df=5),
                           LogisticRegression(max_iter=2000, C=1.0))
        res[(domain, "B_punct_ngram")] = cross_val_score(
            pb, g[text_col].map(punct_stream), y, groups=groups, cv=cv).mean()
    return res


def main():
    df = pd.read_parquet("artifacts/corpus_matched.parquet")
    before = run(df, "text_raw")
    after = run(df, "text")
    out = pd.DataFrame({"raw_text": pd.Series(before), "normalized": pd.Series(after)})
    out.index.names = ["domain", "probe"]
    out.to_csv("reports/shortcut_probe.csv")
    print(out.round(3).to_string())


if __name__ == "__main__":
    main()
