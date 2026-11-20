# Prompt Genesis

**🔗 Live demo: https://promptgenesis.streamlit.app** · [Results](RESULTS.md) · [Dataset card](docs/DATASET_CARD_generated.md)

 **Scope:** student essays. Verdicts are probabilistic, not proof. Non-native writers are still flagged more often than native writers (see RESULTS.md §10 and §12.8).
 
Detects whether an essay was written by AI, reconstructs the prompt that most likely produced it,
and verifies the reconstruction by regenerating the essay and measuring how close it comes.
Classical ML + statistics + interpretability only: gradient-boosted trees and logistic regression
on hand-built stylometric features, with frozen pretrained language models used purely as feature
extractors (gpt2 / distilgpt2 for predictability, all-MiniLM-L6-v2 for coherence). No model is trained
from scratch.

| Part | Question it answers | Method |
|---|---|---|
| **Guard** | Is this essay AI-written? | XGBoost / LightGBM chosen by nested cross-validation, isotonic calibration, "insufficient evidence" band, SHAP reasons |
| **Detective** | Which prompt settings produced it (tone, persona, "sound human", length)? | one classifier per setting, confidence cut-off chosen on cross-validation |
| **Verifier** | Is the reconstructed prompt any good? | regenerate the essay, Match Score (words + structure + ideas), placed between a wrong-prompt floor and a same-prompt ceiling |

**Live app: https://promptgenesis.streamlit.app** (demo mode in the sidebar works without keys)

Results, with confidence intervals and every number traceable to a script: **[RESULTS.md](RESULTS.md)**.
Oral-exam preparation (why each design choice was made): **[VIVA.md](VIVA.md)**.
Generated data documentation: **[docs/DATASET_CARD_generated.md](docs/DATASET_CARD_generated.md)**.

## Run the app locally

    pip install -r requirements.txt
    streamlit run app.py

- Works with no API key: detection, prompt settings and SHAP reasons all run locally.
- An API key (Groq or Google Gemini, entered in the sidebar or set as `GROQ_API_KEY` / `GEMINI_API_KEY`)
  adds an LLM-detected topic and enables the Verifier page (regeneration).
- **Demo mode** (sidebar toggle) shows pre-computed examples from `artifacts/demo_cache.json` and needs
  neither models nor network: the backup for a live demo.

## Deploy (Streamlit Community Cloud, free)

1. Push this folder to a GitHub repository. `data/` and `.env` are git-ignored; the repository must
   contain `app.py`, `src/`, `requirements.txt`, `runtime.txt`, `.streamlit/config.toml` and these
   files from `artifacts/`: `guard_final.joblib`, `guard_feature_medians.json`, `detective_*.joblib`,
   `stylo_scaler.json`, `demo_cache.json`.
2. On share.streamlit.io: New app -> pick the repository, branch `main`, file `app.py`.
3. App settings -> Secrets: `GROQ_API_KEY = "..."` (optional `GEMINI_API_KEY`). Never commit keys.
4. First boot downloads the frozen gpt2 / distilgpt2 / MiniLM weights (~1 GB) and takes a few minutes.
   If the instance runs out of memory, the app still answers in demo mode.

## Reproduce the results

All scripts run from this folder. Steps marked PC need the API keys in `.env` or the frozen LMs.

| step | command | output |
|---|---|---|
| 1. sources | `data/`: HC3, PERSUADE, factorial essays (original project), raw_llm, Liang et al. TOEFL; `python src/sample_raid.py` (PC) | `data/*` |
| 2. generate essays (PC) | `python src/generate_groq.py` | `data/generated/groq_essays.jsonl` |
| 3. corpus + cleaning | `python src/build_corpus.py` | `artifacts/corpus_full.parquet` |
| 4. shortcut probe | `python src/shortcut_probe.py`, `python src/punct_transfer_test.py` | `reports/shortcut_probe.csv` |
| 5. near-duplicates | `python src/dedup.py` | cluster ids |
| 6. length matching | `python src/length_match.py` | `artifacts/corpus_matched.parquet` |
| 7. style features | `python src/features_stylo.py`, `python src/add_stylo_missing.py` | `artifacts/features_stylo.parquet` |
| 8. LM features (PC) | `python src/features_lm.py --input artifacts/<texts>.parquet` | `artifacts/features_lm*.parquet` |
| 9. feasibility gate | `python src/gate.py --features stylo+lm --domain essay` | `reports/gate_*.json` |
| 10. Guard | `python src/guard.py --protocol lopo|logo --features stylo+lm --domain essay` (+ `--inner gkf` ablation, `--final` to save) | `reports/guard_*.json`, `artifacts/guard_final.joblib` |
| 11. plots | `python src/plots.py reports/guard_oof_logo_stylo_lm_essay.parquet` | calibration + abstention PNG |
| 12. Detective | `python src/detective.py --features stylo+lm --tune --save` | `reports/detective_*.json`, `artifacts/detective_*.joblib` |
| 13. statistics | `python src/stats_tests.py`, `python src/evasion_test.py` | `reports/stats_*.csv`, `reports/evasion_effect_by_generator.csv` |
| 14. fairness | `python src/fairness.py` | `reports/fairness.json` |
| 15. SHAP + errors | `python src/explain.py` | `reports/shap_*`, `reports/error_analysis.json` |
| 16. Verifier validity (PC) | `python src/verifier_validity.py` | `data/verifier_validity_summary.json` |
| 17. demo cache (PC) | `python src/make_demo_cache.py` | `artifacts/demo_cache.json` |
| 18. final tests, once | real-chatbot essays (`src/load_chatgpt_test.py`) and `teacher_sim/` | RESULTS.md section "final" |

Rules followed throughout: no number is reported unless a script produced it; nothing is tuned,
selected or thresholded on a test set (tuning uses inner folds of the training split only);
`teacher_sim/` and the real-chatbot essays are scored exactly once, at the end.

## Layout

    app.py                Streamlit app (Detect & Reconstruct, Verify, About)
    src/normalize.py      the single text cleaner, used identically in training and in the app
    src/pipeline.py       loads models, featurizes, explains, reconstructs, verifies
    src/*.py              one script per step (table above)
    artifacts/            models and feature tables (large ones git-ignored)
    reports/              every metric, log and plot
    docs/                 dataset card
