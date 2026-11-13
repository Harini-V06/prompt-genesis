"""Portable model storage. XGBoost's pickled booster can fail to load on another OS
("input stream corrupted" seen moving Linux -> Windows), so XGBoost models are stored in
XGBoost's own JSON model format inside the artifact and rebuilt on load. Other models
(LightGBM, isotonic, label encoders) stay as ordinary pickles."""
import joblib


def to_portable(path):
    art = joblib.load(path)
    m = art.get("model")
    if m is not None and type(m).__name__ in ("XGBClassifier",):
        art["model"] = {"_xgb_json": bytes(m.get_booster().save_raw("json")),
                        "_xgb_params": m.get_params(), "_n_classes": int(getattr(m, "n_classes_", 2))}
        joblib.dump(art, path)
        return True
    return False


def load(path):
    art = joblib.load(path)
    m = art.get("model")
    if isinstance(m, dict) and "_xgb_json" in m:
        from xgboost import XGBClassifier
        clf = XGBClassifier(**{k: v for k, v in m["_xgb_params"].items() if v is not None})
        clf.load_model(bytearray(m["_xgb_json"]))
        art["model"] = clf
    return art


if __name__ == "__main__":
    import sys
    for p in sys.argv[1:]:
        print(p, "converted" if to_portable(p) else "unchanged (not XGBoost)")
