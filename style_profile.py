"""
COUNTED STYLE for prompt reconstruction.

The Detective GUESSES prompt settings with a classifier (tone, persona...). Some aspects of an essay
do not need guessing: they can be COUNTED in the text itself and written straight into the
reconstructed prompt. A count cannot be "wrong" and is never limited to a fixed set of classes.

  measure(raw_text)       -> dict of counts/rates (paragraphs, words, sentence length, reading grade,
                             first/second person, questions, numbers, contractions, headings, lists)
  prompt_clauses(profile) -> list of plain-English instructions for the prompt
  noun_phrase_topic(text) -> offline topic: the most central noun phrase (fallback when no LLM)

Thresholds below are fixed rules of thumb (not fitted to any test data); the benchmark
(src/prompt_benchmark.py) measures whether the clauses actually raise the Match Score.
"""
import re
from collections import Counter

_FIRST = re.compile(r"\b(i|me|my|mine|myself)\b", re.I)
_FIRST_PL = re.compile(r"\b(we|us|our|ours|ourselves)\b", re.I)
_SECOND = re.compile(r"\b(you|your|yours|yourself)\b", re.I)
# 's only for pronoun/it forms (it's, that's...) so possessives (people's) are not counted as contractions
_CONTR = re.compile(r"\b(\w+'(t|re|ve|ll|d|m)|(it|that|there|here|what|he|she|who|let)'s)\b", re.I)
_NUM = re.compile(r"\b\d[\d,.]*%?")
_HEADING = re.compile(r"^\s{0,3}(#{1,6}\s+\S|[A-Z][^.!?\n]{2,60}:?\s*$)", re.M)
_LIST = re.compile(r"^\s*([-*•]|\d+[.)])\s+\S", re.M)
_SENT = re.compile(r"(?<=[.!?])\s+")


def measure(raw_text):
    raw = raw_text.replace("\r\n", "\n").strip()
    lines = raw.split("\n")
    # a title = first non-empty line that is short and has no sentence ending (or a markdown '#')
    first = next((l.strip() for l in lines if l.strip()), "")
    title = bool(first) and (first.startswith("#") or (len(first.split()) <= 12 and not re.search(r"[.!?]$", first)))
    body = "\n".join(lines[[l.strip() for l in lines].index(first) + 1:]) if title else raw
    is_head = lambda b: b.strip().startswith("#") or (len(b.split()) <= 12 and not re.search(r"[.!?:]\s*$", b.strip()) and "\n" not in b.strip())
    paras = [p for p in re.split(r"\n\s*\n", body) if len(p.split()) >= 8 and not is_head(p)]
    if len(paras) <= 1:                                   # single newlines used as paragraph breaks
        paras = [p for p in body.split("\n") if len(p.split()) >= 8 and not is_head(p)] or [raw]
    flat = " ".join(raw.split())
    words = flat.split()
    n = max(len(words), 1)
    sents = [s for s in _SENT.split(flat) if len(s.split()) >= 3] or [flat]
    try:
        import textstat
        grade = float(textstat.flesch_kincaid_grade(flat))
    except Exception:
        grade = None
    # short lines right after a line ending in ':' are an unbulleted LIST, not headings
    headings, unbulleted, after_colon = [], 0, False
    for ln in (raw.split("\n") if "\n" in raw else []):
        t = ln.strip()
        if not t:
            after_colon = False; continue
        if _HEADING.match(ln) and not _LIST.match(ln):
            if after_colon and not t.startswith("#"):
                unbulleted += 1
            else:
                headings.append(t)
        else:
            after_colon = t.endswith(":")
    return {
        "n_words": len(words), "paragraphs": len(paras),
        "sent_len": sum(len(s.split()) for s in sents) / len(sents),
        "grade": grade,
        "title": title,
        "first_person": len(_FIRST.findall(flat)) / n, "first_person_count": len(_FIRST.findall(flat)), "first_plural": len(_FIRST_PL.findall(flat)) / n,
        "second_person": len(_SECOND.findall(flat)) / n,
        "questions": flat.count("?"), "numbers": len(_NUM.findall(flat)),
        "contractions": len(_CONTR.findall(flat.replace("’", "'"))) / n,
        "headings": len(headings), "list_items": len(_LIST.findall(raw)) + unbulleted,
        "em_dashes": raw.count("\u2014") + raw.count(" -- "),
    }


def _round_words(n):
    return int(round(n / 50.0) * 50) if n >= 150 else int(round(n / 25.0) * 25)


def prompt_clauses(p):
    """Turn counts into instructions. Each rule is simple and stated, so it can be explained."""
    c = [f"Write about {_round_words(p['n_words'])} words in {p['paragraphs']} paragraphs."]
    if p.get("title"):
        c.append("Give the essay a title.")
    if p["headings"] >= 2 + int(bool(p.get("title"))):
        c.append("Use section headings.")
    if p["list_items"] >= 2:
        c.append("Include a bulleted or numbered list.")
    c.append(f"Keep sentences around {round(p['sent_len'])} words on average.")
    if p["grade"] is not None:
        g = max(5, min(16, round(p["grade"])))
        c.append(f"Write at about a grade {g} reading level." if g <= 12 else "Write at a university reading level.")
    if p["first_person"] >= 0.01:
        c.append("Write in the first person (I, my), sharing your own view or experience.")
    elif p.get("first_person_count", 0) >= 2:      # a few 'I think / in my opinion' in an otherwise general essay
        c.append("Mostly write in general terms, but give your own opinion in the first person (I think, in my opinion) in a few places.")
    if p["first_plural"] >= 0.01 and p["first_person"] < 0.01:
        c.append("Use 'we' and 'our' to address society as a group.")
    if p["first_person"] < 0.01 and p.get("first_person_count", 0) < 2 and p["first_plural"] < 0.01:
        c.append("Do not use the first person.")
    if p["second_person"] >= 0.01:
        c.append("Address the reader directly as 'you'.")
    if p["questions"] >= 2:
        c.append("Include a few rhetorical questions.")
    if p["numbers"] >= 3:
        c.append("Include specific numbers, dates or statistics.")
    if p.get("em_dashes", 0) >= 2:
        c.append("Use a few em dashes (\u2014) for asides.")
    if p["contractions"] >= 0.01:
        c.append("Use contractions (it's, don't).")
    elif p["contractions"] == 0:
        c.append("Do not use contractions.")
    return c


def noun_phrase_topic(text):
    """Offline topic: the noun phrase that is most frequent and appears early (title-like)."""
    from features_stylo import nlp
    doc = nlp()(text[:4000])
    cnt, first = Counter(), {}
    for nc in doc.noun_chunks:
        toks = [t for t in nc if t.pos_ not in ("DET", "PRON") and not t.is_stop]
        if not toks or len(toks) > 6:
            continue
        key = " ".join(t.text for t in toks)
        if len(key) < 3:
            continue
        cnt[key.lower()] += 1
        first.setdefault(key.lower(), (nc.start, key))
    if not cnt:
        return None
    n_tok = max(len(doc), 1)
    score = {k: v * (1.5 - first[k][0] / n_tok) * (1 + 0.3 * (len(k.split()) - 1)) for k, v in cnt.items()}
    top = sorted(score, key=score.get, reverse=True)[:3]
    top = sorted(top, key=lambda k: first[k][0])            # keep the order they appear in the essay
    return ", ".join(first[k][1] for k in top)
