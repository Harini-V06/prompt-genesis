# RESULTS — real numbers only

Every figure below comes from a script that ran; source file in brackets.
95% CIs are bootstrap (1,000 resamples; cluster bootstrap where near-duplicates exist) or Wilson for proportions.

## Summary (deployed models; details and sources in sections 12-14)
| | test | result |
|---|---|---|
| (a) | AI vs human, essay topic never seen (leave-one-prompt-out, 15 topics) | **98.7%** [98.4, 99.0], AUC 1.000 |
| (b) | AI vs human, AI model never seen (leave-one-generator-out, 6 models) | **97.3%** [97.0, 97.6], AUC 0.998, ECE 0.012 |
| (c) | out of distribution: 4 new families on new topics/genres (n=311) | **99.7%** [98.2, 99.9] |
| (c) | out of distribution: separate free-form Claude collection (n=58) | 82.8% [71.1, 90.4] (weakest case) |
| (d) | real chatbot apps, new topics, scored once: ChatGPT 15 + Claude 8 | **23/23 caught** (Wilson lower bounds 0.80 / 0.68) |
| (d) | teacher_sim | not provided, not evaluated |
| | abstention band P(AI) 0.04-0.96 | answers 90% of essays at 99.6% |
| | Detective (API essays, unseen topics): tone / persona / evasion / length | 85.4% / 86.2% / 65.2% (80.4% when confident, 44%) / 98.8% |
| | Detective on real chatbot apps | does NOT transfer (tone 6/14, persona 5/14 ChatGPT; 0/6 Claude) |
| | Verifier: true prompt beats wrong prompt | 30/30 essays, p = 1.2e-9 |
| | false alarms: native students / non-native (TOEFL) / adult prose | confidently flagged 0.0% / 8.8% / 19.0% |
Live app: https://promptgenesis.streamlit.app

## 1. Leakage before vs after normalize()  [src/shortcut_probe.py, reports/shortcut_probe.csv]

Raw-data artifact rates:

| group | Reddit tokenization | unicode dash/quote | missing space after "." | newline |
|---|---|---|---|---|
| HC3 human | 0.877 | 0.030 | 0.015 | 0.016 |
| HC3 AI (ChatGPT) | 0.001 | 0.005 | 0.197 | 0.308 |
| PERSUADE (human) | 0.122 | 0.349 | 0.002 | 0.962 |
| gpt-oss-120b | 0.000 | 0.972 | 0.000 | 0.675 |
| Claude template | 0.000 | 0.208 | 0.000 | 1.000 |

Shortcut probe accuracy (5-fold GroupKFold by topic, balanced, chance = 0.50):

| domain | probe | raw text | normalized |
|---|---|---|---|
| essay | format-only | 0.936 | **0.552** |
| qa | format-only | 0.975 | **0.566** |
| essay | punctuation n-grams | 0.977 | 0.962 |
| qa | punctuation n-grams | 0.911 | 0.874 |

Punctuation transfer test [src/punct_transfer_test.py]:

| train on | test on | acc | AUC |
|---|---|---|---|
| essay (all AI) | qa | 0.640 | 0.729 |
| qa | essay (all AI) | 0.804 | 0.888 |
| gpt-oss vs human essay | Claude-template vs human essay | 0.968 | 0.994 |
| Claude-template vs human essay | gpt-oss vs human essay | 0.922 | 0.978 |

## 2. Dataset card (so far)

- Corpus: 165k rows (HC3 134k, PERSUADE 15.6k, factorial 4.3k, RAID sample 10.8k, OOD 58, TOEFL 91)
- Near-duplicates: Claude template 2,280 rows -> ~655 clusters; gpt-oss 2,040 -> 2,040
- Training pool after length matching: 16,243 rows
  - essay: 3,372 human / 3,372 AI (gpt-oss 1,784, Claude template 1,588)
  - prose (RAID): 783 human / 3,132 AI across 11 generators
  - qa (HC3): 2,792 / 2,792
- Generated (in progress): Qwen, ALLaM, Gemini, Gemma; 386 jobs per family

## 3. Feasibility gate — stylometric features (68), untuned  [src/gate.py, reports/gate_stylo.json]

| evaluation | XGBoost | TF-IDF + LogReg |
|---|---|---|
| (a) leave-one-prompt-out | 0.951 [0.946, 0.956] | 0.953 |
| (b) LOGO ChatGPT (cross-domain) | 0.478 [0.470, 0.486] | 0.519 |
| (b) LOGO gpt-oss-120b | 0.534 [0.519, 0.550] | 0.608 |
| (b) LOGO Claude template | 0.922 [0.913, 0.932] | 0.734 |
| (c) OOD raw_llm detection rate | 0.345 (narrative 0.24, debate 0.41, explainer 0.44) | |

Tuned (nested), stylometric, LOGO pooled [src/guard.py, reports/guard_logo_stylo.json]:
acc 0.542 [0.535, 0.548], AUC 0.553, PR-AUC 0.566, ECE 0.381; abstention band chosen in-distribution = 0 (coverage 1.00).
(Pre-RAID pool, 3 generators.)

## 4. Detective — stylometric  [src/detective.py, reports/detective_stylo.json]

4,320 factorial essays, 2,695 effective clusters, GroupKFold by topic, cluster-bootstrap CIs.

| label | acc | 95% CI | macro-F1 | chance | gpt-oss | Claude template |
|---|---|---|---|---|---|---|
| tone | 0.889 | [0.878, 0.899] | 0.889 | 0.333 | 0.784 | 0.982 |
| persona | 0.899 | [0.887, 0.911] | 0.898 | 0.333 | 0.841 | 0.950 |
| evasion | 0.642 | [0.631, 0.653] | 0.641 | 0.500 | 0.781 | 0.517 |
| length | 0.998 | [0.995, 1.000] | 0.998 | 0.500 | 0.999 | 0.997 |

## 5. Hypothesis tests  [src/stats_tests.py, reports/stats_*.csv]

- Assumptions: 68/68 features non-normal (D'Agostino-Pearson), 63/68 unequal variances (Levene) -> non-parametric.
- T1 human vs AI essays (Mann-Whitney, BH): 64/68 significant. Largest: MTLD r=0.974 (d=2.55), MATTR 0.957, comma rate 0.925, word length 0.886, Flesch ease -0.845.
- T2 HC3 paired by question (Wilcoxon, 1,486 questions, BH): 62/68 significant. Largest: semicolons -0.953, ellipsis -0.951, bigram repetition +0.925, burstiness -0.765.
- T3 prompt settings, gpt-oss only (n=2,040; Kruskal-Wallis / Mann-Whitney, BH): tone 67/68 (strongest second-person rate, eps2=0.666), persona 61/68 (MTLD, eps2=0.311), evasion 56/68, length 57/68. Post-hoc: 29/30 pairs significant.

## 6. Fairness (prior model, to be redone)
Old app model: 24/91 TOEFL non-native essays falsely flagged AI (26.4%) vs 3.9% for native human essays.

(Sections 1-11 record the project's history in order; the final numbers are in sections 12-14 and the summary at the top.)

## 7. Gate rerun on the corrected pool (16,243 rows, all featurized)  [src/gate.py; reports/gate_*.log]

### Essay-domain subset with LM features (first 54 LM chunks; 5,451 essays with both feature sets)
| test | stylo | stylo + LM |
|---|---|---|
| LOPO | 0.986 [0.983,0.989] | 0.992 [0.990,0.994] |
| LOGO Claude template | 0.964 | 0.971 |
| LOGO gpt-oss | 0.872 | 0.867 |
| OOD raw_llm detected | 0.810 | 0.828 (narrative 0.88, explainer 0.875, debate 0.706) |
| TOEFL false-positive rate (n=91, Wilson 95% CI) | 0.066 [0.031,0.136] | 0.099 [0.053,0.177] |
(old app model TOEFL FPR: 0.264)

### Training-domain experiment, stylometric only, untuned (full pool)
| | essay-only training | all domains |
|---|---|---|
| LOPO essays | 0.989 [0.986,0.991] | 0.959 [0.954,0.964] |
| LOGO gpt-oss | 0.801 [0.787,0.813] | 0.858 [0.848,0.870] |
| LOGO Claude template | 0.954 | 0.956 |
| OOD raw_llm detected | 0.759 | 0.672 |
| TOEFL false-positive rate | 0.066 | 0.275 |
All-domain LOGO on RAID generators (stylo only): chatgpt 0.661 (cross-domain), cohere 0.675, cohere-chat 0.711,
gpt-2 0.660, gpt-3 0.728, gpt-4 0.786, llama-chat 0.797, mistral 0.695, mistral-chat 0.751, mpt 0.693, mpt-chat 0.759.
Interpretation: mixing Q&A/news prose into training hurts essay accuracy and fairness (TOEFL FPR x4); more generators
help on an unseen generator. Design decision to be made by nested validation: essay-focused model with other domains
as down-weighted auxiliary data.
Note: run-to-run variation of a few points between data subsets; OOD n=58 gives ~+/-11 pt CI.

## 8. Tuned Guard, essay domain, stylo + LM  [src/guard.py; reports/guard_{lopo,logo}_stylo_lm_essay.json]
Nested CV (12 configs XGBoost/LightGBM, inner 3-fold GroupKFold by topic), isotonic calibration, abstention.
- LOPO (15 prompts): acc **0.994** [0.992, 0.996], AUC 0.999, PR-AUC 0.999, ECE 0.005; LightGBM chosen in every fold.
- LOGO essays: Claude template 0.861 (AUC 0.963), gpt-oss 0.731 (AUC 0.943); pooled **0.792** [0.782, 0.802],
  AUC 0.953, ECE 0.182; abstention band chosen in-distribution = 0.
- Finding: ranking survives an unseen generator (AUC 0.95) but the 0.5 threshold does not. Tuning and calibrating on
  known-generator validation folds made the model overconfident; the untuned gate scored higher (0.971 / 0.867).
- Fix implemented: inner leave-one-GENERATOR-out validation for model selection, calibration and abstention when a
  training fold has >= 3 AI generators (needs the newly generated essays: 6 essay generators).

## 9. Detective v2: factorial + generated essays with known prompts (stylometric)  [reports/detective_stylo.json]
4,964 essays (3,339 effective clusters); plain "Write an essay about X" prompts labelled "unstyled".
GroupKFold by topic (clusters nested in topics), cluster-bootstrap 95% CI.

| label | acc | 95% CI | macro-F1 | chance (majority) | gpt-oss | Claude tmpl | Gemini | Gemma | Qwen | ALLaM |
|---|---|---|---|---|---|---|---|---|---|---|
| tone (4 classes) | 0.855 | [0.844,0.866] | 0.803 | 0.325 | 0.782 | 0.991 | 0.611 | 0.711 | 0.488 | 0.539 |
| persona (4) | 0.865 | [0.854,0.875] | 0.815 | 0.320 | 0.832 | 0.974 | 0.571 | 0.784 | 0.523 | 0.425 |
| evasion (2) | 0.653 | [0.643,0.664] | 0.651 | 0.517 | 0.795 | 0.519 | 0.674 | 0.684 | 0.767 | 0.632 |
| length (2) | 0.993 | [0.991,0.995] | 0.993 | 0.506 | 0.999 | 1.000 | 0.978 | 0.969 | 0.825 | 0.888 |

Finding: prompt-setting inference transfers across generators far worse than human-vs-AI detection.
Above chance for every family, but tone/persona on the new families is 43-78%. Each model realises
"casual" or "expert" in its own way. The app shows per-setting confidence for this reason.
Per-family n is small for Qwen (~86 essays): wide uncertainty.

## 10. Fairness audit: non-native writers  [src/fairness.py; reports/fairness.json]
Untuned XGB (same as gate), essay pool with LM features (n=6,374). TOEFL = 91 non-native essays
(Liang et al. 2023), median 104 words. Native comparison = PERSUADE human essays <= 200 words (n=1,316;
shortest the pool has, min 139), scored out-of-fold by topic. Threshold 0.5. Wilson 95% CIs; Fisher exact.

| Model | TOEFL FPR | Native FPR (<=200w, OOF) | Fisher p | Essay AI recall |
|---|---|---|---|---|
| M0 stylo + LM | 8/91 = 8.8% [4.5, 16.4] | 3/1316 = 0.2% [0.1, 0.7] | 3.2e-08 | 0.986 |
| M1 drop LM features | 5/91 = 5.5% [2.4, 12.2] | 7/1316 = 0.5% [0.3, 1.1] | 5.6e-04 | 0.981 |
| M2 + non-native data (5-fold CV over TOEFL) | 4/91 = 4.4% [1.7, 10.8] | 2/1316 = 0.2% [0.0, 0.6] | 2.2e-04 | 0.984 |

Old app model: 26.4%. The bias is reduced ~3x but NOT removed: non-native writers are still flagged
significantly more often than native writers (all p < 0.001). Perplexity features add some of the bias
(8.8% vs 5.5%; CIs overlap, so not conclusive at n=91). Adding a small amount of non-native human data is the
most effective mitigation at almost no recall cost. Limitations: n=91; TOEFL essays are still shorter than
the shortest native comparison essays; one prompt set. Deployment consequence: the app never presents a verdict
as proof and shows an "insufficient evidence" band.

## 11. Guard on the 6-generator data, BEFORE generated-text cleaning (superseded by section 12)
> Found afterwards during error analysis: all Gemma essays contained a `<thought>` planning block and 329
> generated essays were cut off at the token limit. Kept here as a record; re-run on cleaned data in section 12.

### (was) FINAL-DATA Guard, essay domain, stylo + LM  [src/guard.py; reports/guard_{logo,lopo}_stylo_lm_essay.json]
Data: 7,949 essays with full stylometric + LM features; 6 AI generators (gpt-oss-120b, Claude template, Gemini,
Gemma, Qwen, ALLaM) + 3,989 PERSUADE human essays. Nested tuning (12 XGB/LGBM configs), isotonic calibration.
Hard negatives (paraphrases of another model's essay) are removed from training whenever their SOURCE essay's
generator is held out (93 rows for gpt-oss; before this fix gpt-oss was 0.977, after 0.978: no inflation).

### (b) Unseen generator: leave-one-generator-out (inner leave-one-generator-out tuning)
| held-out generator | acc | AUC | TF-IDF LR baseline |
|---|---|---|---|
| ALLaM | 0.971 | 0.998 | 0.985 |
| Claude template | 0.980 | 0.998 | 0.967 |
| Gemini | 0.988 | 1.000 | 0.976 |
| Gemma | 0.967 | 0.996 | 0.958 |
| gpt-oss-120b | 0.978 | 0.998 | 0.812 |
| Qwen | 0.997 | 1.000 | 0.997 |
| **pooled** | **0.979 [0.976, 0.982]** | 0.998 | 0.903 |
ECE 0.016. Accuracy-coverage: 99.7% when the 10% least-confident essays are abstained on; 99.9% at 80%.
(Pooled n counts some human test essays in several folds: each fold samples its own held-out human topics.)

### (a) Unseen prompt/topic: leave-one-prompt-out over 15 PERSUADE prompts
Pooled acc **0.988 [0.986, 0.990]**, AUC 0.999, ECE 0.010; per prompt 0.957 (Car-free cities) to 0.996.
TF-IDF LR 0.990: on KNOWN generators a bag-of-words model is just as good. The style + LM features earn their
place on UNSEEN generators (0.979 vs 0.903), which is the case that matters in deployment.

### Ablation: what closed the gap from 0.792 to 0.979?
| setting | pooled LOGO acc | ECE |
|---|---|---|
| old data (2 essay generators, partial LM), inner GroupKFold tuning (section 8) | 0.792 | 0.182 |
| new data (6 generators, full LM), inner GroupKFold tuning | 0.963 [0.960, 0.967] | 0.040 |
| new data, inner leave-one-generator-out tuning (final) | **0.979** [0.976, 0.982] | **0.016** |
Most of the gain (+17 pts) is generator diversity in training; tuning the way we test adds +1.6 pts and cuts
calibration error by 60%, with the largest per-generator gains on the hardest cases (Gemma 0.939 -> 0.967,
Claude template 0.956 -> 0.980, gpt-oss 0.963 -> 0.978).

## 12. FINAL (cleaned generated text), essay domain, stylo + LM
7,994 length-matched essays (3,997 PERSUADE human, 3,997 AI: gpt-oss-120b 1,784, Claude template 1,588, Gemini 192, ALLaM 169, Qwen 145, Gemma 119), every row with all 91 features (68 stylometric + 23 LM). Generated text cleaned as in the dataset card.

### 12.1 Guard, (b) unseen generator (leave-one-generator-out, inner leave-one-generator-out tuning)
| held-out generator | acc | AUC | TF-IDF LR |
|---|---|---|---|
| ALLaM | 0.980 | 0.993 | 0.986 |
| Claude template | 0.951 | 0.998 | 0.967 |
| Gemini | 0.985 | 1.000 | 0.977 |
| Gemma | 0.991 | 1.000 | 0.976 |
| gpt-oss-120b | 0.985 | 0.999 | 0.834 |
| Qwen | 0.985 | 1.000 | 0.978 |
| **pooled** | **0.972 [0.969, 0.976]** | 0.997 | 0.914 |
ECE 0.022. Abstention (most-confident first): 97.2% at 100% answered, 99.6% at 90%, 99.9% at 80%.
Effect of cleaning vs section 11: pooled 0.979 -> 0.972; Gemma rose (0.967 -> 0.991), so the `<thought>`
blocks were not an easy shortcut; Claude template fell (0.980 -> 0.951) with an unchanged AUC (0.998):
a threshold shift on the new human sample, not a loss of separability.
Plot: reports/plot_calib_abstain_logo_stylo_lm_essay.png

### 12.2 Detective (tuned: nested random search, inner GroupKFold(3) by topic; outer GroupKFold(5) by topic)
5,022 AI essays with known prompt settings; cluster-bootstrap 95% CIs.
| setting | classes | acc [CI] | macro-F1 | chance (majority) | confident cut-off (chosen on OOF) |
|---|---|---|---|---|---|
| tone | 4 (incl. unstyled) | 0.854 [0.842, 0.864] | 0.784 | 0.321 | p >= 0.34: 100% answered at 85.4% |
| persona | 4 (incl. unstyled) | 0.862 [0.849, 0.872] | 0.788 | 0.321 | p >= 0.34: 100% at 86.2% |
| evasion ("sound human") | 2 | 0.652 [0.642, 0.663] | 0.651 | 0.518 | **p >= 0.66: 44% answered at 80.4%** |
| length | 2 | 0.988 [0.985, 0.991] | 0.988 | 0.507 | 100% at 98.8% |
Evasion per generator: gpt-oss 0.788, Qwen 0.735, Gemma 0.720, Gemini 0.680, ALLaM 0.648, Claude template 0.517.
Tuning and LM features did not raise evasion overall (stylo-only untuned 0.653). Why (reports/evasion_effect_by_generator.csv,
Mann-Whitney per feature, sound_human vs none, BH-FDR within generator): gpt-oss changes 69/91 features, Gemma 9;
Claude template 1/91 with max |r| = 0.04 at n = 2,172, i.e. Claude's essays are statistically unchanged by the
instruction, so no classifier can recover it there. The app shows evasion only above the cut-off.

### 12.3 Verifier validity  [src/verifier_validity.py (PC), src/verifier_stats.py]
For essays with a KNOWN prompt, regenerate twice with the true prompt and once with a wrong prompt (other topic),
same model family. One-sided paired Wilcoxon (true > wrong), BH-FDR over 4 components.
(n = 30 essays: ALLaM 10, Gemini 9, Gemma 6, Qwen 5; the other 10 of the planned 40 failed on free-tier rate limits / server errors)
| component | true | wrong | diff [95% CI] | p (BH) | rank-biserial | true > wrong |
|---|---|---|---|---|---|---|
| Match Score | 0.581 | 0.373 | 0.208 [0.183, 0.235] | 1.2e-09 | 1.00 | 30/30 |
| words | 0.244 | 0.117 | 0.128 [0.110, 0.147] | 1.2e-09 | 1.00 | 30/30 |
| structure | 0.895 | 0.719 | 0.176 [0.126, 0.232] | 9.3e-09 | 0.98 | 28/30 |
| ideas | 0.604 | 0.284 | 0.320 [0.274, 0.362] | 1.2e-09 | 1.00 | 30/30 |
Ceiling (same model, same prompt, twice): 0.583. The true prompt sits at the ceiling (median position 0.98 on the
floor->ceiling scale). Scope: this validates the MEASURING STICK with true prompts; it does not show that the
app's reconstructed prompts are correct.

### 12.4 Guard, (a) unseen prompt/topic (leave-one-prompt-out, 15 PERSUADE prompts)
Pooled **0.988 [0.985, 0.990]**, AUC 1.000, ECE 0.010; per prompt 0.954 ("Car-free cities") to 0.997 ("Community service").
TF-IDF LR also 0.988: on KNOWN generators bag-of-words is as good; the style + LM features matter on UNSEEN
generators (0.972 vs 0.914).

### 12.5 Ablation on clean data: what closed the gap from 0.792?
| setting | pooled LOGO acc | ECE |
|---|---|---|
| 2 essay generators, partial LM, inner GroupKFold tuning (section 8) | 0.792 | 0.182 |
| 6 generators, full LM, clean text, inner GroupKFold tuning | 0.970 [0.967, 0.974] | 0.028 |
| 6 generators, full LM, clean text, inner leave-one-generator-out tuning (final) | **0.972** [0.969, 0.976] | **0.022** |
Honest reading: on the clean data almost all of the gain (+17.8 pts) comes from generator diversity + full LM
features. Tuning by inner leave-one-generator-out adds only +0.2 pts (CIs overlap: not significant) and a small
calibration gain (ECE 0.028 -> 0.022). Per generator it helps Claude template (0.944 -> 0.951) and Qwen
(0.974 -> 0.985) and slightly hurts Gemini (0.998 -> 0.985) and Gemma (0.996 -> 0.991). (Before cleaning the
same comparison showed +1.6 pts; that larger gain did not survive the cleaning, so we do not claim it.)
We keep inner leave-one-generator-out because it is the principled choice (validation mirrors the test), not
because it scored higher.

### 12.6 Threat: does the essay Guard treat polished ADULT human writing as AI?  [src/domain_mix.py; reports/domain_mix.json]
Found from SHAP: the top feature is lexical diversity (MTLD), and our human essays are grade 6-12 students.
Direct test on adult human prose the essay model never trained on (RAID human news/wiki/books/reviews etc.):
- deployed final Guard, RAID human texts 150-1,200 words with features (n = 659): **42.3% flagged AI**
  (19.7% outside the "insufficient evidence" band).
Mitigation experiment, same untuned model for both rows so only the training data differs:
| training data | essay LOGO acc | adult human prose FPR (OOF) | TOEFL non-native FPR |
|---|---|---|---|
| E: essays only | 0.974 [0.971, 0.977] | 0.267 [0.237, 0.299] | 0.110 [0.061, 0.191] |
| E+P: essays + RAID prose (domain x label balanced) | 0.962 [0.957, 0.965] | **0.120** [0.099, 0.145] | **0.264** [0.184, 0.363] |
Adding adult prose halves the adult false-positive rate but more than doubles the false-positive rate for
non-native student writers, and costs 1.2 pts on essays. There is no free lunch: the model learns "what humans
in the training data write like". For a classroom essay tool (many students in our setting write in English as a
second language) wrongly accusing non-native students is the worse error, so we DEPLOY E (essays only) and
**restrict the stated scope to student essays**: the app says so, and adult/professional writing is listed as
out of scope in the limitations. This is a deliberate, evidence-based trade-off, reported with both numbers.

### 12.7 Model selection: tuned vs fixed (decided on leave-one-generator-out validation only)
| Guard | pooled LOGO acc | AUC | ECE |
|---|---|---|---|
| tuned (12-config nested search, inner leave-one-generator-out) | 0.972 [0.969, 0.976] | 0.997 | 0.022 |
| **fixed** (one preset XGBoost config, same folds + isotonic calibration) | **0.973 [0.970, 0.976]** | 0.998 | **0.012** |
Per generator (fixed): ALLaM 0.971, Claude template 0.957, Gemini 0.979, Gemma 0.983, gpt-oss 0.984, Qwen 0.985.
Rule: keep the simpler model unless tuning wins on validation. It does not (equal accuracy, worse calibration),
so the DEPLOYED Guard is the fixed model (artifacts/guard_final.joblib; the tuned one is kept as
guard_final_tuned.joblib). Fairness/out-of-scope numbers below were looked at for both models but did NOT
drive the choice.

### 12.8 Deployed Guard on human text it never trained on  [src/deployed_checks.py; reports/deployed_checks.json]
"Insufficient evidence" band: P(AI) in [0.04, 0.96] (target fixed in advance: abstain on the least-confident
~10% of LOGO OOF predictions); outside it, LOGO accuracy 99.6% on 90.2% of essays (76.2% inside it).
| human group | n | flagged AI (p >= 0.5) | flagged CONFIDENTLY (outside band) |
|---|---|---|---|
| native student essays <= 200 words (OOF) | 1,270 | 1.1% [0.7, 1.8] | 0.0% [0.0, 0.3] |
| TOEFL non-native essays (Liang et al.) | 91 | 26.4% [18.4, 36.3] | 8.8% [4.5, 16.4] |
| adult professional prose (RAID, OUT OF SCOPE) | 659 | 39.8% [36.1, 43.5] | 19.0% [16.2, 22.1] |
Non-native writers are confidently flagged far more often than comparable native students
(Fisher exact p = 3e-10). The bias is reduced relative to the original project's model (26.4% at a hard
threshold with no abstention, section 10) only through the abstention band; it is NOT solved. The app
therefore never presents a verdict as proof, shows "insufficient evidence" in the band, and states its scope
(student essays). The essays + adult-prose model (12.6) trades adult false alarms for even more non-native
false alarms, so it was not deployed.

## 13. FINAL ONE-SHOT TESTS (deployed models frozen; each set scored once)  [src/final_test.py]
| test set | what it is | n | accuracy [95% CI] | answered outside band | accuracy when answered |
|---|---|---|---|---|---|
| ood_generated | our 4 new generator families on NEW topics/genres (UBI, narratives, explainers...) | 311 | 0.997 [0.982, 0.999] | 95% | 1.000 |
| raw_llm | separate free-form Claude collection (narrative etc.), original project | 58 | 0.828 [0.711, 0.904] | 67% | 0.974 |
| **realchat_chatgpt** | essays copied from the ChatGPT app, 15 new-topic prompts, new chat each | 15 | **1.000 [0.796, 1.000]** | 93% | 1.000 |
| **realchat_claude** | essays copied from the Claude app, 8 prompts (web search on for one) | 8 | **1.000 [0.676, 1.000]** | 75% | 1.000 |
All four sets are AI-only, so accuracy here = recall (share of AI essays caught); the human-side error rates are in 12.8.
Note on independence: raw_llm was used as an out-of-distribution CHECK during design (section 4 domain decision),
so it is not fully untouched; ood_generated and the real-chat essays never influenced any choice.
The real-chat sets are small: the Wilson lower bounds (0.80 and 0.68) are the honest guarantee, not the 100%.
Claude-style free-form writing (raw_llm 0.83) is the weakest case found.
teacher_sim: not provided; no teacher-simulated set was evaluated.
Deployed (fixed) Guard, (a) unseen topic (leave-one-prompt-out): 0.987 [0.984, 0.990], AUC 1.000, ECE 0.011
(tuned: 0.988; TF-IDF 0.988).

## 14. Interpretability and error analysis of the deployed Guard  [src/explain.py]
**Global SHAP** (mean |SHAP|, 2,000 pool essays; reports/plot_shap_groups.png, plot_shap_beeswarm.png):
vocabulary 3.95 > punctuation 1.98 > function words 1.77 > parts of speech 1.60 > LM predictability 1.43 >
sentence rhythm 0.61 > readability 0.59 > LM coherence 0.44 > pronouns 0.38 > hedges 0.06 > syntax 0.01.
Top features: MTLD lexical diversity 2.50, comma rate 1.34, auxiliary-verb rate 0.64, Binoculars ratio 0.61,
mean word length 0.58, "a" 0.54, "to" 0.51, MATTR 0.48.
Reading: the model relies most on vocabulary richness and punctuation habits. MTLD being first is exactly why
adult professional prose and non-native writing are the risk groups (12.6, 12.8): it partly learns "writes like
a school student" vs "writes like a polished model".
**Errors** (leave-one-generator-out out-of-fold, n = 9,232 predictions, error 2.7%):
- AI essays missed 4.6% vs humans wrongly flagged 1.2%: the model errs towards "human" (the safer direction
  for a classroom tool).
- Length: <= 200 words 5.0% error vs 1.2-1.6% for 200-600 words: short texts carry less evidence.
- Prompt persona: AI essays told to write as a "non-native speaker" are missed most (8.1%, vs student 2.3%);
  the evasion instruction barely matters (4.5% vs 5.0%).
- By held-out generator: Claude template 4.4%, ALLaM 2.9%, Gemini 2.1%, Gemma 1.7%, gpt-oss 1.6%, Qwen 1.5%.
- Read examples: the most confident misses are short, plain Claude-template "expert" essays (~165 words) and
  simple ALLaM essays; the most confident false alarms are fluent, mostly long student essays (262-988 words; 3 of 5 over 900).

### 13.1 Detective on the real-chatbot essays (true settings known from the prompts we typed)  [reports/final_test_detective_realchat.json]
| app | tone correct | persona correct | evasion correct |
|---|---|---|---|
| ChatGPT (14 essays with a confident AI verdict) | 6/14 | 5/14 | 13/14 |
| Claude (6) | 0/6 | 0/6 | 1/6 |
Most prompts were plain ("Write an essay about X"), so the truth is mostly "unstyled". The Detective labelled most
ChatGPT essays "non-native speaker" (simple sentences, British spelling from the app) and every Claude essay
"casual / sound human" (Claude's chatty default voice). ChatGPT evasion 13/14 mostly reflects predicting the majority
class "none". **The Detective does not transfer to consumer chatbot apps**: it learned prompt-setting signatures from
API generations with our own prompt templates, while the apps add hidden system prompts and personalisation that
change the default voice. The Guard transfers (23/23 caught); the Detective's numbers in 12.2 hold only for
API-generated essays. Future work: label essays collected from real chatbot apps.

### 12.9 Feature ablation on the final setup (fixed config, clean data, leave-one-generator-out)  [reports/guard_logo_stylo_essay_fixed.log]
| features | pooled LOGO acc | ECE | ALLaM | Claude tmpl | Gemini | Gemma | gpt-oss | Qwen |
|---|---|---|---|---|---|---|---|---|
| stylometric only (68) | 0.976 [0.973, 0.979] | 0.011 | 0.957 | 0.979 | 0.990 | 0.985 | 0.973 | 0.980 |
| stylometric + LM (91, deployed) | 0.973 [0.970, 0.976] | 0.012 | 0.971 | 0.957 | 0.979 | 0.983 | 0.984 | 0.985 |
| TF-IDF LR baseline | 0.914 | | 0.986 | 0.967 | 0.977 | 0.976 | 0.834 | 0.978 |
On the clean 6-generator data the LM features add NOTHING overall (CIs overlap); they help ALLaM and gpt-oss and
hurt the Claude template. Their earlier gain (section 7, +OOD) was measured on the smaller uncleaned data. By the
"simpler model wins" rule the stylometric-only model would be preferred, and it would make the app far lighter
(no PyTorch). We did NOT switch: the final one-shot tests (section 13) were run on the frozen stylo+LM model, and
re-testing a new model on them would use the test sets twice. Recorded as the main future-work item.
