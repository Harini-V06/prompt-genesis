# Final deliverables checklist

Status checked on 14 Nov 2026 against the spec's list. "Where" points to the file and section.

| # | Deliverable | Status | Where |
|---|---|---|---|
| 1 | `normalize()` + shortcut-probe result | done: format-only probe 0.936 -> 0.552 (essay), 0.975 -> 0.566 (QA) after normalize; punctuation transfer test shows real style signal | `src/normalize.py`; RESULTS s.1 |
| 2 | Dataset card for the new generated data | done (sources, prompt kinds, cleaning with row counts, ids, limitations) | `docs/DATASET_CARD_generated.md` |
| 3 | Results table (a)(b)(c)(d) with CIs | done | RESULTS summary table + s.12-13 |
| 4 | Calibration + abstention plots | done | `reports/plot_calib_abstain_logo_stylo_lm_essay_fixed.png` (+ LOPO); RESULTS s.12.1, 12.8 |
| 5 | Ablation | done: data/tuning ablation (s.12.5), tuned vs fixed (s.12.7), feature groups stylo vs stylo+LM (s.12.9), training domain (s.12.6) | RESULTS |
| 6 | Hypothesis tests | done: T1 64/68, T2 62/68, T3 per setting, assumption checks, BH-FDR, effect sizes; evasion per generator | RESULTS s.5, s.12.2; `reports/stats_*.csv`, `reports/evasion_effect_by_generator.csv` |
| 7 | Detective per-label results | done (accuracy, macro-F1, chance, cluster-bootstrap CI, per generator, confidence cut-offs, confusion matrices in JSON) + real-chatbot transfer test | RESULTS s.12.2, s.13.1; `reports/detective_stylo_lm_tuned.json` |
| 8 | Verifier validity test | done: n = 30, true > wrong in 30/30, p = 1.2e-9 per component (BH), ceiling 0.58 | RESULTS s.12.3; `reports/verifier_validity_summary.json` |
| 9 | Fairness audit | done: TOEFL vs native (Fisher p = 3e-10), mitigation experiments, adult-prose out-of-scope check, deployed-model numbers | RESULTS s.10, 12.6, 12.8 |
| 10 | Error analysis | done (by generator, length, persona, evasion, topic; most confident mistakes read) + global SHAP | RESULTS s.14; `reports/error_analysis.json`, `reports/plot_shap_*.png` |
| 11 | Deployed Streamlit URL | done | https://promptgenesis.streamlit.app |
| 12 | Demo-mode fallback | done: 5 pre-run examples never seen in training (4 AI caught at 0.98-1.00, human control 0.00) | `artifacts/demo_cache.json`; sidebar toggle |
| 13 | RESULTS.md | done | `RESULTS.md` |
| 14 | VIVA.md | done | `VIVA.md` |
| 15 | README with run instructions | done (local run, deploy, 18-step reproduction table) | `README.md` |

## Rules
- No fabricated numbers: every number in RESULTS.md was printed by a script in `src/` (logs/JSON in `reports/`).
- No tuning on test sets: model choice, calibration and thresholds used inner folds or leave-one-generator-out
  validation only. The abstention band and the Detective cut-offs were chosen on cross-validated predictions with
  targets fixed in advance (10% abstention; 80% accuracy).
- Final test sets scored once with frozen models (section 13). Disclosed exception: `raw_llm` had been used as an
  out-of-distribution check during design.

## Open limitations (stated in RESULTS and in the app)
- Scope is student essays; adult professional writing is confidently flagged 19.0% of the time.
- Non-native student essays are confidently flagged 8.8% vs 0.0% for native ones: reduced, not solved.
- The Detective does not transfer to consumer chatbot apps.
- LM features add no accuracy on the final data (s.12.9); a lighter style-only model is the main future-work item.
- teacher_sim was not provided, so it was not evaluated.
