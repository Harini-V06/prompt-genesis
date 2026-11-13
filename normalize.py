"""
normalize(text): the ONE text cleaner used identically in training and at
inference (app). Its job is to erase formatting/tokenization artifacts that
reveal the *source dataset* rather than the *author* (human vs AI), so the
models cannot learn shortcuts such as "Reddit tokenization => human" or
"curly quotes => gpt-oss".

Steps (order matters):
 1. Unicode NFKC + map every dash/quote/space variant to plain ASCII
 2. Remove URL tokens (HC3 "URL_0", raw http links) and markdown (headings,
    bold/italics, bullets, numbered lists)
 3. Drop a leading title line (short line with no sentence punctuation)
 4. Detokenize Reddit/PTB-style spacing (" , " " n't" " 's" "( x )")
 5. Fix missing space after sentence punctuation ("end.Next" -> "end. Next")
 6. Collapse ALL whitespace incl. newlines to single spaces (paragraph layout
    differs by source: 96% of PERSUADE has newlines vs 2% of HC3-human)
"""
import re
import unicodedata

_CHAR_MAP = {
    # dashes / hyphens
    "‐": "-", "‑": "-", "‒": "-", "–": "-", "—": " - ",
    "―": " - ", "−": "-", "⸺": " - ", "⸻": " - ",
    # quotes
    "‘": "'", "’": "'", "‚": "'", "‛": "'", "′": "'",
    "“": '"', "”": '"', "„": '"', "‟": '"', "″": '"',
    "«": '"', "»": '"',
    # spaces / invisibles
    " ": " ", " ": " ", " ": " ", " ": " ", " ": " ",
    "​": "", "‌": "", "‍": "", "﻿": "",
    # ellipsis, bullets
    "…": "...", "•": " ", "·": " ",
}
_TRANS = str.maketrans(_CHAR_MAP)

_URL_RE = re.compile(r"(https?://\S+|www\.\S+|\bURL_\d+\b)", re.I)
_MD_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s*", re.M)
_MD_EMPH_RE = re.compile(r"(\*\*|__|\*|_)(?=\S)(.+?)(?<=\S)\1")
_MD_BULLET_RE = re.compile(r"^\s*(?:[-*+]|\d{1,2}[.)])\s+", re.M)

# PTB / Reddit-style detokenization
_DETOK = [
    (re.compile(r"\s+n't\b"), "n't"),
    (re.compile(r"\s+'(s|re|ve|ll|d|m)\b", re.I), r"'\1"),
    (re.compile(r"\s+([,.;:!?%)\]}])"), r"\1"),
    (re.compile(r"([(\[{$])\s+"), r"\1"),
    (re.compile(r"``\s*|\s*''"), '"'),
    (re.compile(r'"\s+([^"]*?)\s+"'), r'"\1"'),
    (re.compile(r"\s+-\s+-\s+"), " - "),
]
_MISSING_SPACE_RE = re.compile(r"([a-z0-9\)\"'][.!?])([A-Z][a-z])")
_WS_RE = re.compile(r"\s+")


def _strip_title(text: str) -> str:
    lines = text.strip().split("\n")
    if len(lines) > 1:
        first = lines[0].strip()
        nxt = next((l.strip() for l in lines[1:] if l.strip()), "")
        if (first and len(first.split()) <= 12 and not re.search(r"[.!?,;:]$", first)
                and re.match(r"[A-Z0-9\"'*#]", nxt or "A")):
            return "\n".join(lines[1:])
    return text


_LIST_ITEM = re.compile(r"^\s*(?:[-*+\u2022]\s+|\d{1,2}[.)]\s+|\(\d{1,2}\)\s+)")
_LABEL_PREFIX = re.compile(r"^\s*(title|essay|introduction|conclusion|body)\s*:\s*", re.I)


def _strip_headings(text: str) -> str:
    """Drop standalone section-heading lines ("Introduction", "Conclusion",
    "The Case for Uniforms") anywhere in the text: short lines (<= 8 words) with
    no sentence-ending punctuation and no comma (so "Dear Principal," survives).
    Also drops "Title: ..." label lines. Generated essays often carry headings,
    human essays rarely do, so leaving them in would be a formatting shortcut."""
    lines = text.split("\n")
    out = []
    for i, line in enumerate(lines):
        s = line.strip()
        if s and _LABEL_PREFIX.match(s) and not re.search(r"[.!?]$", s):
            continue
        if _LIST_ITEM.match(s):
            out.append(line)
            continue
        if s and len(s.split()) <= 8 and not re.search(r"[.!?,;:\"')]$", s) and re.match(r"[A-Z*#]", s):
            nxt = next((l.strip() for l in lines[i + 1:] if l.strip()), None)
            # a heading is followed by a NEW sentence (capital) or ends the text;
            # a hard-wrapped sentence fragment is followed by a lowercase continuation
            if nxt is None or re.match(r"[A-Z0-9\"'*]", nxt):
                continue
        out.append(line)
    return "\n".join(out)


def normalize(text) -> str:
    if text is None:
        return ""
    t = unicodedata.normalize("NFKC", str(text))
    t = t.translate(_TRANS)
    t = t.replace("\r\n", "\n").replace("\r", "\n")
    t = _URL_RE.sub(" ", t)
    t = _MD_EMPH_RE.sub(r"\2", t)
    t = _strip_title(t)
    t = _strip_headings(t)          # before bullet removal, so list items stay protected
    t = _MD_HEADING_RE.sub("", t)
    t = _MD_BULLET_RE.sub("", t)
    t = _WS_RE.sub(" ", t)          # collapse before detok so patterns are uniform
    for pat, rep in _DETOK:
        t = pat.sub(rep, t)
    t = _MISSING_SPACE_RE.sub(r"\1 \2", t)
    # Dash neutralization: spaced dashes / em-dashes / "--" used as punctuation
    # become commas. Shortcut-probe evidence: spaced-dash rate predicts AI in
    # essays (gpt-oss em-dash habit) but predicts HUMAN in Q&A (Reddit users),
    # i.e. its direction flips with the source => dataset artifact, not an AI
    # trait. Hyphenated words ("well-known") are untouched.
    t = re.sub(r"\s*-{2,}\s*", ", ", t)
    t = re.sub(r"\s+-\s+", ", ", t)
    t = re.sub(r"\s*,(\s*,)+", ",", t)
    t = re.sub(r",\s*([.!?;:])", r"\1", t)
    # Quote removal: PERSUADE students were given source articles to quote,
    # the AI generators were not => quote rate reflects the task setup, not
    # authorship (shortcut-probe AUC ~0.58-0.60). Apostrophes are kept
    # (contractions are genuine style).
    t = t.replace('"', "")
    t = re.sub(r" {2,}", " ", t)
    # any remaining non-ASCII letters (accents etc.) are kept by NFKC; strip
    # other non-ASCII symbols (emoji, arrows) that are source-specific
    t = "".join(ch for ch in t if ord(ch) < 128 or ch.isalpha())
    return t.strip()


if __name__ == "__main__":
    tests = [
        "I do n't think so , it 's fine . URL_0 Next",
        "AI loves em—dashes and “curly quotes”‑here.Then no space.",
        "# My Title\n\n**Bold** point.\n\n- bullet one\n- bullet two",
    ]
    for s in tests:
        print(repr(normalize(s)))
