"""
Build ONE unified, normalized corpus from all raw sources.

Output: artifacts/corpus_full.parquet (every row, before length matching)

Columns
  uid          unique id
  text_raw     original text
  text         normalize(text_raw)   <- the only text any model ever sees
  label        1 = AI, 0 = human
  domain       'essay' | 'qa'
  dataset      hc3 | persuade | factorial_api | factorial_claude | raw_llm | toefl
  generator    human | chatgpt (HC3 AI) | gpt-oss-120b | claude-template | claude-freeform
  prompt_type  qa | factorial | ood | fairness
  topic        canonical PERSUADE prompt (essays) / HC3 question (qa) / free topic
  hc3_source   reddit_eli5/finance/... (qa only)
  tone persona evasion length   factorial labels (AI essays only)
  n_words      words after normalization
"""
import json
from pathlib import Path
import re
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, "src")
from normalize import normalize

D = "data/"

# --- map factorial topic names onto PERSUADE's 15 canonical prompts --------
CANON = {
    "A cowboy in a storm": '"A Cowboy Who Rode the Waves"',
    "Cowboy who rode the storm out": '"A Cowboy Who Rode the Waves"',
    "A Cowboy Who Rode the Waves": '"A Cowboy Who Rode the Waves"',
    "Car-free cities": "Car-free cities",
    "Cell phones in school": "Cell phones at school",
    "Cell phones at school": "Cell phones at school",
    "Community service": "Community service",
    "Distance learning": "Distance learning",
    "Driverless cars": "Driverless cars",
    "Electoral college": "Does the electoral college work?",
    "Does the electoral college work?": "Does the electoral college work?",
    "Exploring Venus": "Exploring Venus",
    "Face on Mars": "The Face on Mars",
    "The Face on Mars": "The Face on Mars",
    "Facial action coding system": "Facial action coding system",
    "Grades for extracurricular activities": "Grades for extracurricular activities",
    "Mandatory extracurricular activities": "Mandatory extracurricular activities",
    "Multiple perspectives": "Seeking multiple opinions",
    "Seeking multiple opinions": "Seeking multiple opinions",
    "Phones and driving": "Phones and driving",
    "Summer projects": "Summer projects",
}

# the one appended evasion boilerplate in the Claude template file
CLAUDE_BOILERPLATE = re.compile(r"\s*\(I know this is a debated topic.*?thinking it over\.\)\s*$", re.S)


_THOUGHT_END = re.compile(r"</\s*(thought|think|thinking)\s*>", re.I)
_SENT_END = re.compile(r"[.!?][\"'\u201d\u2019)\]*_]*(?=\s|$)")


def clean_generated(text, finish_reason=None):
    """Remove generation artefacts that are NOT part of the essay:
    1. reasoning traces: Gemma models emit a <thought>...</thought> planning block before the
       essay; everything up to the closing tag is removed (all such rows have a closing tag).
    2. truncation: when the API stopped at the token limit (finish_reason == "length") the last
       sentence is cut mid-way; the text is trimmed back to its last complete sentence.
    Nothing else is changed."""
    parts = _THOUGHT_END.split(text)
    t = parts[-1].strip() if len(parts) > 1 else text.strip()
    if finish_reason == "length":
        ends = list(_SENT_END.finditer(t))
        if ends:
            t = t[:ends[-1].end()].strip()
    return t


def frame(df, **const):
    out = df.copy()
    for k, v in const.items():
        out[k] = v
    return out


def main():
    parts = []

    h = pd.read_csv(D + "hc3_filtered.csv", low_memory=False)
    h = pd.DataFrame({
        "text_raw": h.text, "label": (h.label == "AI").astype(int),
        "topic": h.question.astype(str), "hc3_source": h.source.astype(str),
    })
    h["generator"] = np.where(h.label == 1, "chatgpt", "human")
    parts.append(frame(h, domain="qa", dataset="hc3", prompt_type="qa"))

    p = pd.read_csv(D + "persuade_clean.csv")
    p = pd.DataFrame({"text_raw": p.text, "label": 0, "topic": p.question})
    parts.append(frame(p, domain="essay", dataset="persuade", prompt_type="human_essay", generator="human"))

    a = pd.read_csv(D + "factorial_essays_real_api_final.csv")
    a = pd.DataFrame({
        "text_raw": a.text, "label": 1, "topic": a.topic.map(CANON),
        "tone": a.tone, "persona": a.persona, "evasion": a.evasion, "length": a.length,
        "generator": np.where(a.model_used.str.contains("gpt-oss"), "gpt-oss-120b", "claude-template"),
    })
    parts.append(frame(a, domain="essay", dataset="factorial_api", prompt_type="factorial"))

    c = pd.read_csv(D + "factorial_essays_claude.csv")
    c = pd.DataFrame({
        "text_raw": c.text.str.replace(CLAUDE_BOILERPLATE, "", regex=True),
        "label": 1, "topic": c.topic.map(CANON),
        "tone": c.tone, "persona": c.persona, "evasion": c.evasion, "length": c.length,
    })
    parts.append(frame(c, domain="essay", dataset="factorial_claude", prompt_type="factorial",
                       generator="claude-template"))

    r = pd.DataFrame(json.load(open(D + "raw_llm_essays.json")))
    r = pd.DataFrame({"text_raw": r.text, "label": 1, "topic": r.topic, "genre": r.genre})
    parts.append(frame(r, domain="essay", dataset="raw_llm", prompt_type="ood", generator="claude-freeform"))

    t = pd.read_csv(D + "liang_toefl_nonnative.csv")
    t = pd.DataFrame({"text_raw": t.text, "label": 0, "topic": "TOEFL"})
    parts.append(frame(t, domain="essay", dataset="toefl", prompt_type="fairness", generator="human"))

    # RAID sample (Dugan et al. 2024, MIT): adult human prose + 11 generators.
    # Appended LAST so earlier uids stay stable. domain = "prose".
    rp = Path(D + "raid_sample.jsonl")
    if rp.exists():
        rd = pd.read_json(rp, lines=True)
        gen_map = {"gpt4": "gpt-4", "gpt3": "gpt-3", "gpt2": "gpt-2"}
        rd = pd.DataFrame({
            "text_raw": rd.text, "label": (rd.model != "human").astype(int),
            "topic": "raid|" + rd.source_id.astype(str), "raid_domain": rd.domain,
            "generator": rd.model.map(lambda m: "human" if m == "human" else gen_map.get(m, m)),
        })
        parts.append(frame(rd, domain="prose", dataset="raid", prompt_type="raid"))

    # Generated essays (Groq / Gemini families). Appended LAST. Their uids are PERMANENT:
    # "g<line index in groq_essays.jsonl>", so cleaning or dropping one row never shifts another
    # (artifacts/uid_migration_generated.csv maps the earlier sequential ids).
    gp = Path(D + "generated/groq_essays.jsonl")
    if gp.exists():
        g = pd.read_json(gp, lines=True)
        g["gen_idx"] = np.arange(len(g))
        g = g[g.text.str.split().str.len() >= 20]
        g["text"] = [clean_generated(t, fr) for t, fr in zip(g.text, g.finish_reason)]
        g = g[g.text.str.split().str.len() >= 20]
        api = pd.read_csv(D + "factorial_essays_real_api_final.csv")
        src_topic = {f"factorial_api_row{i}": CANON.get(t, t) for i, t in enumerate(api.topic)}
        styled = g.kind.isin(["plain", "factorial", "newtopic"])
        topic = np.where(g.kind.isin(["paraphrase", "light_edit"]), g.source_ref.map(src_topic), g.topic)
        gd = pd.DataFrame({
            "text_raw": g.text, "label": 1, "topic": topic,
            "generator": g.family.map(lambda f: {"google-gemini": "gemini", "google-gemma": "gemma"}.get(f, f)),
            "gen_model": g.generator, "gen_kind": g.kind, "gen_prompt": g.prompt,
            "source_ref": g.get("source_ref"),
            # plain prompts carry no style settings -> the Detective's "unstyled" outcome
            "tone": np.where(g.kind == "plain", "unstyled", g.get("tone")),
            "persona": np.where(g.kind == "plain", "unstyled", g.get("persona")),
            "evasion": np.where(g.kind == "plain", "none", g.get("evasion")),
            "length": g.get("length"),
            "genre": g.get("genre"),
            "gen_uid": [f"g{i:05d}" for i in g.gen_idx],
            "truncated": g.finish_reason.eq("length").values,
        })
        gd["prompt_type"] = np.where(g.kind == "newtopic", "ood_generated",
                                     np.where(styled, "generated", "hard_negative"))
        parts.append(frame(gd, domain="essay", dataset="generated"))

    df = pd.concat(parts, ignore_index=True)
    assert df.topic.notna().all(), df[df.topic.isna()].dataset.value_counts()
    df["text"] = df.text_raw.map(normalize)
    df["n_words"] = df.text.str.split().str.len()
    df = df[df.n_words > 0].reset_index(drop=True)
    is_gen = df.dataset.eq("generated").values
    assert not is_gen[:(~is_gen).sum()].any(), "generated block must be last"
    uid = [f"u{i:06d}" for i in range(int((~is_gen).sum()))] + list(df.loc[is_gen, "gen_uid"])
    df.insert(0, "uid", uid)
    df = df.drop(columns=["gen_uid"])
    df.to_parquet("artifacts/corpus_full.parquet", index=False)

    print("rows:", len(df))
    print(df.groupby(["dataset", "generator", "label"]).agg(n=("uid", "size"),
          words=("n_words", "median")).to_string())


if __name__ == "__main__":
    main()
