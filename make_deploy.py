"""Assemble deploy/prompt-genesis: exactly what goes into the GitHub repository for Streamlit Cloud.
No data/ (licences, size), no .env (keys), no large feature tables. Prints a file list to check."""
import shutil
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "deploy" / "prompt-genesis"
shutil.rmtree(OUT, ignore_errors=True); OUT.mkdir(parents=True)
copy = ["app.py", "requirements.txt", "runtime.txt", "README.md", "RESULTS.md", "VIVA.md", ".gitignore",
        ".streamlit/config.toml", "docs/DATASET_CARD_generated.md"]
copy += [str(p.relative_to(ROOT)) for p in sorted((ROOT / "src").glob("*.py"))]
copy += ["artifacts/" + f for f in ["guard_final.joblib", "guard_feature_medians.json", "stylo_scaler.json",
                                    "model_card.json", "demo_cache.json", "detective_tone.joblib",
                                    "detective_persona.joblib", "detective_evasion.joblib", "detective_length.joblib"]]
copy += [str(p.relative_to(ROOT)) for p in sorted((ROOT / "reports").glob("*")) if p.suffix in (".json", ".csv", ".png")]
missing = []
for rel in copy:
    src = ROOT / rel
    if not src.exists():
        missing.append(rel); continue
    (OUT / rel).parent.mkdir(parents=True, exist_ok=True); shutil.copy2(src, OUT / rel)
files = [p for p in OUT.rglob("*") if p.is_file()]
print(f"{len(files)} files, {sum(p.stat().st_size for p in files)/1e6:.1f} MB ->", OUT)
print("MISSING:", missing or "none")
shutil.make_archive(str(ROOT / "deploy" / "prompt-genesis"), "zip", OUT.parent, OUT.name)
