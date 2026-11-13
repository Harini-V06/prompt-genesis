"""Fit + save the deployable Detective heads WITHOUT re-running the outer cross-validation (already run and
reported by `detective.py --features stylo+lm --tune`, reports/detective_stylo_lm_tuned.json).
Same procedure as `--save`: config chosen by inner GroupKFold(3) by topic on all data, then fit on all data;
the confidence cut-off is the one chosen on the outer out-of-fold predictions in that report."""
import json, sys
import joblib
from sklearn.preprocessing import LabelEncoder
sys.path.insert(0, "src")
from detective import LABELS, load, select, make
df_all, cols = load("stylo+lm")
rep = json.load(open("reports/detective_stylo_lm_tuned.json"))
for lab in LABELS:
    df = df_all[df_all[lab].notna()].reset_index(drop=True)
    use = cols + (["n_words"] if lab == "length" else [])
    le = LabelEncoder().fit(df[lab]); y = le.transform(df[lab])
    assert list(le.classes_) == rep[lab]["classes"]
    kind, p = select(df[use], y, df.topic.values)
    clf = make(kind, p).fit(df[use], y)
    joblib.dump({"model": clf, "encoder": le, "features": use, "conf_threshold": rep[lab]["conf_threshold"],
                 "kind": kind, "params": p}, f"artifacts/detective_{lab}.joblib")
    from portable import to_portable; to_portable(f"artifacts/detective_{lab}.joblib")
    print(f"saved detective_{lab}: {kind} | cut-off {rep[lab]['conf_threshold']}", flush=True)
