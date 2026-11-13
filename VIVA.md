# VIVA — likely questions from a strict examiner, with answers

Every number quoted here comes from a script in `src/` and is logged in RESULTS.md.
Sections grow as the project progresses.

---

## 1. Data and leakage

**Q: Your old model scored 96.9%. Why should I believe the new, lower numbers instead?**
Because the old number measured the wrong thing. We ran a *shortcut probe*: a classifier that sees only formatting (non-ASCII characters, newlines, tokenization spacing, URL tokens, quote style) and no words at all. On the raw data it reached 93.6% (essays) and 97.5% (Q&A). So the data itself leaked its source: HC3 human answers carry Reddit tokenization (" n't", " , ") in 87.7% of cases vs 0.1% of AI answers; gpt-oss essays contain unicode dashes/quotes 97.2% of the time vs 34.9% for PERSUADE. A model can score highly by learning "Reddit formatting = human" without learning anything about AI writing. After `normalize()` the same probe drops to 55.2% / 56.6% (chance = 50%), so formatting no longer gives the answer away.

**Q: What does normalize() do, and why is it applied at inference too?**
NFKC unicode normalization; all dash/quote/space variants mapped to ASCII; URL tokens, markdown, titles and section headings removed; Reddit/PTB detokenization; missing spaces after periods fixed; all whitespace collapsed. It is one function (`src/normalize.py`) imported by both training and the app, so the model only ever sees text cleaned the same way. Cleaning differently at inference would create a train/test mismatch.

**Q: Why did you turn dashes into commas and delete double quotes? Isn't that destroying signal?**
Each removal is backed by evidence, not taste:
- *Spaced dashes*: their rate predicts AI in essays (gpt-oss em-dash habit) but predicts HUMAN in Q&A (Reddit users). A feature whose direction flips with the data source is a source artifact, not an AI trait.
- *Double quotes*: PERSUADE students were given source articles to quote; the AI generators never saw those articles. Quote rate reflects the task setup, not authorship.
Apostrophes were deliberately kept: they are contractions, a genuine style trait.

**Q: Your punctuation n-gram probe is still at 96%. Isn't that leakage?**
We tested that directly. If punctuation were a source artifact it would not transfer across sources. A punctuation-only model trained on Q&A (ChatGPT vs Reddit humans) scores 80.4% accuracy / 0.888 AUC on essays (different generators, different humans, different domain). Artifacts do not survive that; genuine style does. The top punctuation patterns are commas (AI writes longer, comma-heavy sentences) and question marks / short sentences (humans), which are writing style. (`src/punct_transfer_test.py`)

**Q: Humans wrote longer essays than the AI. Did length leak?**
It would have: raw PERSUADE essays have a median of ~382 words vs 160-330 for AI. We (1) never use raw word count or sentence count as a Guard feature, only rates and ratios, and replaced raw type-token ratio with MATTR because raw TTR falls as text gets longer; (2) length-matched the classes inside 50-word bins per domain. Medians after matching: essays 222 vs 220, prose 197 vs 196, Q&A 239 vs 233. In RAID we keep up to 4 AI texts per human per bin, so counts differ but length distributions are identical; the count imbalance is handled by sample weights.

**Q: The Claude essays are templated. How did you stop near-duplicates inflating scores?**
Every essay in the Claude template file has a near-duplicate (TF-IDF cosine > 0.85), and every "sound human" essay ended with the same appended sentence. We stripped that sentence, clustered near-duplicates (connected components, per generator and topic), and use the cluster as the cross-validation group, so near-copies can never be split across train and test. Effective sample size: 2,280 Claude-template rows are really ~655 independent essays. We report that, not the row count. (TF-IDF was used because the dev sandbox could not download MiniLM; MiniLM agreement is checked later.)

**Q: Your human essays are school students. Isn't that a confound?**
Yes, and we found it ourselves. The strongest human-vs-AI difference is vocabulary diversity (MTLD 67 vs 171, rank-biserial 0.97), but PERSUADE is grade 6-12 writing. A model could learn "polished = AI" and flag an adult's writing. Mitigation: we added adult human prose from RAID (news, books, abstracts, Wikipedia, reviews, Reddit; MIT license), verified from the file itself to contain 13,371 human rows. Remaining risk is reported in threats to validity.

**Q: Why these external datasets? Did you check licenses?**
RAID (Dugan et al., ACL 2024) is MIT-licensed per its repository and dataset card. We stream-sample its no-attack file (802 MB) instead of downloading it, keep up to 150 texts per (generator, domain) with reservoir sampling, and only prose domains (code, poetry, recipes excluded). ArguGPT was rejected: its license is listed only as "cc" with no variant, and its humans are TOEFL/GRE learners, which would contaminate the fairness audit.

---

## 2. Generators and evaluation design

**Q: What is leave-one-generator-out and why does it matter more than normal accuracy?**
We train without any text from one AI model and test on that model. This simulates the real situation: the teacher will paste text from ChatGPT/Gemini/Claude web versions the model may never have seen. Random-split accuracy only tells you how well the model recognises generators it already knows.

**Q: Your first gate said 95%. Why not just report that?**
95.1% is leave-one-prompt-out: unseen *topics* but *seen generators*. Leave-one-generator-out with the same features gave 47.8% (ChatGPT, cross-domain), 53.4% (gpt-oss), 92.2% (Claude template); fully tuned XGBoost/LightGBM pooled at 54.2%. Stylometry recognises known generators, not AI in general. Reporting only 95% would hide the thing the teacher's test will actually measure.

**Q: How did you expand the data?**
- 4 new generator families via free APIs: Qwen (Groq), ALLaM (Groq), Gemini and Gemma (Google AI Studio, OpenAI-compatible endpoint). gpt-oss was excluded because it was already in the data.
- Per family, identical job specs: plain prompts ("Write an essay about X", what a teacher types), a reduced tone x persona x evasion grid, 37 topics outside the original 15, and hard negatives (paraphrased / lightly edited AI essays).
- Temperature 0.9-1.0, varied "angle" per call, no fixed appended text (avoiding the Claude-template mistake).
- 11 more generators from RAID (GPT-4, ChatGPT, GPT-3, GPT-2, Llama-2-chat, Mistral (2), Cohere (2), MPT (2)).
Every row keeps generator, prompt type and topic, so evaluation can hold out any of them.

**Q: Why did you turn off the Qwen/Gemini "thinking" mode?**
Groq's model page documents `reasoning_effort="none"` as instruct mode. Thinking spends hidden tokens: it was exhausting the free-tier rate limit, and in the earlier app it returned empty essays. Essays from instruct mode are what a normal chat request returns.

---

## 3. Features

**Q: Why only rates and ratios?**
So text length cannot re-enter through the back door (see length confound above).

**Q: Why use GPT-2 at all? Isn't that a neural network?**
The course allows frozen pretrained models as feature extractors. GPT-2 and DistilGPT-2 are never trained or fine-tuned; we only read their token probabilities. Rationale: AI text tends to consist of tokens a language model finds highly probable (low perplexity, many rank-1 tokens). That property is shared across generators, unlike one model's stylistic habits, which is exactly what failed in leave-one-generator-out. GPT-2 and DistilGPT-2 share a tokenizer, which the Binoculars-style cross-perplexity ratio requires.

**Q: Why 128 tokens?**
The app's minimum input is 100 words, about 128 GPT-2 tokens. Scoring every text on the same window means a long essay and a short paragraph are judged on equal terms; per-token averages over a fixed window also keep length out of the LM features. (It also halved CPU time, since there was no GPU.)

**Q: Why not feed sentence embeddings directly into the classifier?**
384-dimensional MiniLM embeddings mostly encode *topic*. Feeding them in would reintroduce the topic confound we removed. We only use topic-free statistics derived from them (coherence between adjacent sentences, spread around the document centroid). Raw embeddings are used where topic IS the point: the Verifier's "Ideas" score.

---

## 4. Modelling

**Q: How did you avoid tuning on the test set?**
Nested cross-validation. For every outer fold (one held-out prompt or generator), hyperparameter search, model choice (XGBoost vs LightGBM), isotonic calibration and the abstention threshold are all fitted using inner 3-fold GroupKFold *on the outer-training data only*. The held-out fold is scored exactly once. teacher_sim is untouched until the very end and scored once.

**Q: Why GroupKFold and what are the groups?**
Groups are the topic/prompt (essays), the question (HC3, so a human and AI answer to the same question stay together), and the source document (RAID). Near-duplicate clusters always sit inside one topic, so they can never straddle train and test.

**Q: What does "insufficient evidence" mean technically?**
After isotonic calibration, predictions inside a band [0.5-d, 0.5+d] are withheld. d is the smallest half-width whose inner-validation accuracy on answered cases reaches 90%. We always report accuracy *at* a stated coverage plus overall accuracy.
**Honest finding:** with stylometric features the inner data is so easy (seen generators) that d came out ~0 and the model never abstained on unseen generators. Abstention calibrated in-distribution does not protect against a new generator. This is reported as a limitation.

---

## 5. Detective

**Q: How accurate is the prompt reconstruction?**
(stylometric features, 5-fold GroupKFold by topic, cluster-bootstrap 95% CIs)
tone 88.9% [87.8, 89.9] (chance 33%), persona 89.9% [88.7, 91.1] (33%), evasion 64.2% [63.1, 65.3] (50%), length 99.8% (50%).
The Claude-template rows inflate tone/persona (98%/95%) because templates use fixed phrases. gpt-oss (real LLM text) is the honest estimate: 78% tone, 84% persona, 78% evasion.

**Q: Why is Claude evasion at chance (52%)?**
Because we removed the only thing that distinguished it: the appended "(I know this is a debated topic...)" sentence. Leaving it in would have given a fake high score. This is the correct result.

**Q: You said length is never a feature, yet the Detective uses word count.**
Only the Detective's *length* head uses it, because "concise vs elaborate" IS a length instruction; word count is the direct evidence for that target, not a confound. The Guard never sees length.

---

## 6. Statistics

**Q: Why non-parametric tests?**
Assumption checks: all 68 features fail D'Agostino-Pearson normality and 63/68 fail Levene's equal-variance test, so t-tests/ANOVA would be invalid. We use Mann-Whitney U, Wilcoxon signed-rank (HC3, paired by question, 1,486 questions) and Kruskal-Wallis, with rank-biserial / epsilon-squared effect sizes and Benjamini-Hochberg FDR correction.

**Q: With n in the thousands, everything is significant. So what?**
Correct, which is why we rank by effect size, not p-value. Large effects: MTLD (r = 0.97), comma rate (0.93), word length (0.89); HC3 pairs: AI repeats bigrams more (0.93) and has lower sentence-length burstiness (-0.77), matching the "low burstiness" signature reported in the detection literature.

---

## 7. Engineering decisions

**Q: Why are some steps run on your laptop and others elsewhere?**
The development environment could not reach Hugging Face or the APIs, so steps needing internet (model downloads, essay generation, RAID streaming) run on the local machine and their cached outputs are brought back. All other processing, evaluation and documentation happen in one reproducible pipeline under git.

**Q: Bugs you found and fixed?**
- spaCy dependency-depth loop compared Token objects with `is`, which is never true because spaCy builds new objects on each access, causing an infinite loop. Fixed by comparing indices.
- The heading remover first deleted hard-wrapped sentence fragments and ChatGPT list items; fixed by requiring the next line to start a new sentence and protecting list markers. Verified by diffing every changed text.


## Q: Your generated data came from APIs. How do you know it is clean?
I did not assume it. Error analysis on the out-of-fold mistakes showed a Gemma essay that started with
"<thought> * Topic: ... * Tone: ..." - the model's planning notes. Checking all generated text found this in
every Gemma output (321), plus 329 essays that stopped mid-sentence at the token limit (finish_reason =
"length"). Both are artefacts, not essay content, and both could act as shortcuts (planning bullets look
nothing like a human essay; a cut-off ending is unusual). `clean_generated()` removes everything up to
`</thought>` and trims truncated essays to their last full sentence; 59 outputs with < 20 words left are
dropped. Every Guard/Detective number that included Gemma was re-run on the cleaned text, and the old
numbers are kept in RESULTS.md, marked as superseded. Generated rows got permanent ids (line number in the
jsonl) so cleaning one row can never shift another row's stored features.

## Q: Does the prompt reconstruction work on real ChatGPT / Claude essays?
Honestly, not reliably. On 23 essays we copied from the ChatGPT and Claude apps (prompts known), the Guard caught
all 23, but the Detective got tone right 6/14 and persona 5/14 for ChatGPT and 0/6 for Claude: it called plain
ChatGPT essays "non-native speaker" and every Claude essay "casual, sound human". It was trained on API generations
with our own templates; the apps add hidden instructions and personalisation that shift the default voice. So the
Detective's 85-99% holds for API-generated essays only. The Verifier itself is valid (true prompt beats a wrong one
in 30/30), so a wrong reconstruction shows up as a LOW match score: the system can tell you when its own
reconstruction is poor.

## Q: Why did you deploy the untuned model?
Tuning (12-config nested search) did not beat one preset configuration on leave-one-generator-out validation
(97.2% vs 97.3%, CIs overlap) and was worse calibrated (ECE 0.022 vs 0.012). Rule: keep the simpler model unless
tuning wins on validation. Fairness numbers were looked at for both but did not decide it.

## Q: Is the detector fair to non-native writers?
Not fully. Native student essays are confidently flagged 0.0% of the time, TOEFL non-native essays 8.8% (p = 3e-10).
Adding adult prose to training halves adult false alarms but more than doubles non-native ones, so we kept essays
only, show "insufficient evidence" in an uncertainty band, restrict the scope to student essays, and never present
a verdict as proof.
