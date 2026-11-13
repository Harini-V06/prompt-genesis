# Dataset card: Prompt Genesis generated essays

File: `data/generated/groq_essays.jsonl` (1,255 raw generations; 1,196 kept after cleaning)
Built by: `src/generate_groq.py` (generation), `src/build_corpus.py` (cleaning, labels, ids)
Collected: 13 November 2026, 05:28 to 14:04 UTC (Asia/Dubai 09:28 to 18:04), from one laptop

## Why it exists
The original project had AI essays from only two sources (gpt-oss-120b and a Claude template),
so a detector could not be tested on an AI model it had never seen. This set adds four more
model families, so every essay generator can be held out in turn (leave-one-generator-out).

## Sources (APIs, all called with temperature defaults, reasoning/thinking switched off where the API allows it)
| family | model id(s) | provider | rows |
|---|---|---|---|
| ALLaM | allam-2-7b | Groq | 386 |
| Gemini | gemini-3.1-flash-lite (334), 3.6-flash (18), 3.5-flash-lite (7), 3.8-flash (6), 3.5-flash (1) | Google, OpenAI-compatible endpoint | 366 |
| Gemma | gemma-4-26b-a4b-it | Google, OpenAI-compatible endpoint | 321 |
| Qwen | qwen3.8-27b | Groq | 182 |
When a model hit rate limits the generator fell back to another model **of the same family** (never across
families), so the family label, which is what leave-one-generator-out holds out, is always correct.

## Prompt kinds (`kind`)
| kind | what the model was asked | used as |
|---|---|---|
| plain (206) | "Write an essay about X." (no style instructions) | training/eval, Detective label "unstyled" |
| factorial (520) | topic + tone x persona x evasion x length instructions, same design as the original factorial set | training/eval, Detective ground truth |
| newtopic (326) | essays on topics/genres NOT in training (narrative, explainer, debate...) | out-of-distribution TEST only |
| paraphrase (103) | rewrite an existing gpt-oss / Claude-template essay in new words | hard negatives (still AI) |
| light_edit (100) | lightly edit an existing AI essay to sound more human | hard negatives (still AI) |
Topics for plain/factorial are the 15 PERSUADE prompts, so AI and human essays cover the same subjects
(the topic cannot be used as a shortcut).

## Cleaning (src/build_corpus.py: `clean_generated`), documented row counts
1. **Reasoning traces removed.** All 321 Gemma outputs start with a `<thought>...</thought>` planning
   block (task restated as bullets, word-count targets). Everything up to the closing tag is removed;
   every affected row had a closing tag. Found during error analysis on 13 Nov; all Guard/Detective results
   before that date that include Gemma were re-run on the cleaned text.
2. **Token-limit truncation trimmed.** 329 outputs stopped at the token limit (finish_reason = "length":
   ALLaM 188, Gemma 131 after cleaning, Qwen 8, Gemini 2) and end mid-sentence. They are trimmed back to the
   last complete sentence so "ends mid-sentence" cannot become a shortcut.
3. Outputs with fewer than 20 words are dropped: 12 ALLaM outputs that were near-empty as returned, and
   47 Gemma outputs whose thought block used the whole token budget (59 in total; 1,255 - 59 = 1,196 kept). Then the same `normalize()` as every other source (headings, markdown, quotes, dashes).
No other edits. Essays were not filtered by how "AI-like" they look.

## Identifiers
Generated rows have permanent ids `g<line index in the jsonl>`, so cleaning or dropping one row can never
shift another row's id (features are stored by id). `artifacts/uid_migration_generated.csv` maps the
earlier sequential ids.

## How it enters the experiments
- Length matching: like every source, essays enter the training pool only via 50-word bins matched to
  human essays (`src/length_match.py`), so length cannot separate the classes.
- Near-duplicates: clustered per (generator, topic); paraphrases/edits join their SOURCE essay's cluster,
  and are removed from training whenever their source's generator is held out (`src/guard.py`).
- `newtopic` essays never enter training or tuning: they are an out-of-distribution test set.

## Known limitations
- One day of collection, default sampling settings, API versions as of 13 Nov 2026: future model
  versions may write differently.
- Qwen has few paraphrase/edit rows and only 182 essays (rate limits), so its per-model numbers are noisier.
- ALLaM essays are long (median ~700 words) and ALLaM truncated often; trimming keeps complete sentences
  but the essays may lack a conclusion.
- Generated text is used for non-commercial academic research under each provider's terms and is not
  redistributed (data/ is excluded from the repository).
