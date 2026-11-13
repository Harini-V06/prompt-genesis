"""
Stylometric features. Shared by training and the app (import extract_stylo).

Design rule: every feature is a RATE or RATIO, never a raw count or raw length,
so text length (a confound: humans ~421w vs AI ~283w in the raw data) cannot
re-enter through the back door. Raw TTR is replaced by MATTR (moving-average
TTR, window 50), which does not shrink as texts get longer.

Groups (used for ablation):
  sent     sentence-length mean/std/burstiness, short/long sentence share
  lex      word length, MATTR, MTLD, long-word share, sentence-opener diversity, self-repetition
  pos      spaCy universal POS ratios
  syntax   passive voice, subordination, dependency depth (spaCy parser)
  pron     first/second person pronoun rates, contractions
  disc     hedges, boosters, formal discourse markers
  punct    punctuation rates per 100 words
  read     Flesch reading ease, Flesch-Kincaid grade, Gunning fog
  func     relative frequencies of 20 common function words
"""
import re
import sys
import numpy as np
import pandas as pd
import spacy
import textstat

POS = ["NOUN", "VERB", "ADJ", "ADV", "PRON", "DET", "ADP", "CCONJ", "SCONJ", "AUX", "PROPN", "NUM", "PART", "INTJ"]
FUNC = ["the", "of", "and", "to", "a", "in", "that", "is", "it", "for", "as", "with", "be", "on", "not",
        "this", "but", "by", "or", "so"]
HEDGES = ["maybe", "perhaps", "might", "possibly", "probably", "likely", "seem", "seems", "seemed", "appear",
          "appears", "somewhat", "generally", "often", "arguably", "i think", "i believe", "sort of", "kind of",
          "in my opinion", "i guess"]
BOOSTERS = ["clearly", "certainly", "definitely", "undoubtedly", "indeed", "always", "never", "absolutely",
            "obviously", "truly", "crucial", "essential", "significant", "significantly"]
DISCOURSE = ["however", "moreover", "furthermore", "additionally", "therefore", "thus", "consequently",
             "overall", "in conclusion", "ultimately", "firstly", "secondly", "finally", "in addition",
             "on the other hand", "for example", "for instance", "in summary", "notably", "importantly"]
FP_SING = {"i", "me", "my", "mine", "myself"}
FP_PLUR = {"we", "us", "our", "ours", "ourselves"}
SECOND = {"you", "your", "yours", "yourself", "yourselves"}
SUBORD_DEPS = {"mark", "advcl", "ccomp", "acl", "relcl", "xcomp", "csubj"}

_WORD = re.compile(r"[A-Za-z']+")
_NLP = None


def nlp():
    global _NLP
    if _NLP is None:
        _NLP = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer"])
    return _NLP


def _phrase_rate(low, phrases, n_words):
    return sum(len(re.findall(r"\b" + re.escape(p) + r"\b", low)) for p in phrases) / n_words * 100


def mattr(words, w=50):
    if len(words) < w:
        return len(set(words)) / max(len(words), 1)
    counts, uniq, vals = {}, 0, []
    for i, t in enumerate(words):
        counts[t] = counts.get(t, 0) + 1
        uniq += counts[t] == 1
        if i >= w:
            old = words[i - w]
            counts[old] -= 1
            uniq -= counts[old] == 0
        if i >= w - 1:
            vals.append(uniq / w)
    return float(np.mean(vals))


def mtld(words, thr=0.72):
    def one(seq):
        f, types, n = 0.0, set(), 0
        for t in seq:
            types.add(t); n += 1
            if len(types) / n <= thr:
                f += 1; types = set(); n = 0
        if n:
            ttr = len(types) / n
            f += (1 - ttr) / (1 - thr) if ttr < 1 else 0
        return len(seq) / f if f else len(seq)
    return (one(words) + one(words[::-1])) / 2 if len(words) >= 10 else 0.0


def _depth(tok):
    d = 0
    while tok.head.i != tok.i and d < 100:   # spaCy builds new Token objects; compare indices
        tok = tok.head; d += 1
    return d


def features_from_doc(doc, text):
    low = text.lower()
    words = [w.lower() for w in _WORD.findall(text)]
    nw = max(len(words), 1)
    sents = [s for s in doc.sents if any(t.is_alpha for t in s)]
    sl = np.array([sum(1 for t in s if not t.is_punct and not t.is_space) for s in sents] or [0])
    ns = max(len(sents), 1)
    toks = [t for t in doc if not t.is_space]
    nt = max(len(toks), 1)
    f = {}
    # sent
    f["sent_len_mean"] = sl.mean()
    f["sent_len_std"] = sl.std()
    f["sent_burstiness"] = sl.std() / sl.mean() if sl.mean() else 0
    f["sent_short_share"] = (sl < 8).mean()
    f["sent_long_share"] = (sl > 30).mean()
    # lex
    wl = np.array([len(w) for w in words] or [0])
    f["word_len_mean"] = wl.mean()
    f["word_len_std"] = wl.std()
    f["long_word_share"] = (wl >= 7).mean()
    f["mattr50"] = mattr(words)
    f["mtld"] = mtld(words)
    firsts = [next((t.lower_ for t in s if t.is_alpha), "") for s in sents]
    f["opener_diversity"] = len(set(firsts)) / ns
    bigr = list(zip(words, words[1:]))
    f["bigram_repeat_rate"] = 1 - len(set(bigr)) / max(len(bigr), 1)
    # pos
    pc = {p: 0 for p in POS}
    for t in toks:
        if t.pos_ in pc:
            pc[t.pos_] += 1
    for p in POS:
        f[f"pos_{p.lower()}"] = pc[p] / nt
    # syntax
    f["passive_rate"] = sum(1 for s in sents if any(t.dep_ in ("nsubjpass", "auxpass") for t in s)) / ns
    f["subord_per_sent"] = sum(1 for t in toks if t.dep_ in SUBORD_DEPS) / ns
    f["dep_depth_mean"] = float(np.mean([max((_depth(t) for t in s), default=0) for s in sents])) if sents else 0
    # pron
    f["fp_sing_rate"] = sum(w in FP_SING for w in words) / nw * 100
    f["fp_plur_rate"] = sum(w in FP_PLUR for w in words) / nw * 100
    f["second_person_rate"] = sum(w in SECOND for w in words) / nw * 100
    f["contraction_rate"] = len(re.findall(r"\b\w+'(t|s|re|ve|ll|d|m)\b", low)) / nw * 100
    # disc
    f["hedge_rate"] = _phrase_rate(low, HEDGES, nw)
    f["booster_rate"] = _phrase_rate(low, BOOSTERS, nw)
    f["discourse_rate"] = _phrase_rate(low, DISCOURSE, nw)
    # punct (per 100 words)
    for name, ch in [("comma", ","), ("semicolon", ";"), ("colon", ":"), ("question", "?"), ("exclaim", "!"),
                     ("paren", "("), ("ellipsis", "...")]:
        f[f"p_{name}"] = text.count(ch) / nw * 100
    f["p_period_per_sent"] = text.count(".") / ns
    f["p_intraword_hyphen"] = len(re.findall(r"\w-\w", text)) / nw * 100
    # read
    try:
        f["flesch_ease"] = textstat.flesch_reading_ease(text)
        f["fk_grade"] = textstat.flesch_kincaid_grade(text)
        f["gunning_fog"] = textstat.gunning_fog(text)
    except Exception:
        f["flesch_ease"] = f["fk_grade"] = f["gunning_fog"] = np.nan
    # func
    for w in FUNC:
        f[f"fw_{w}"] = words.count(w) / nw * 100
    return f


def extract_stylo(texts, n_process=1, batch_size=64):
    texts = list(texts)
    docs = nlp().pipe(texts, n_process=n_process, batch_size=batch_size)
    return pd.DataFrame([features_from_doc(d, t) for d, t in zip(docs, texts)])


GROUPS = {
    "sent": lambda c: c.startswith("sent_"),
    "lex": lambda c: c in ("word_len_mean", "word_len_std", "long_word_share", "mattr50", "mtld",
                           "opener_diversity", "bigram_repeat_rate"),
    "pos": lambda c: c.startswith("pos_"),
    "syntax": lambda c: c in ("passive_rate", "subord_per_sent", "dep_depth_mean"),
    "pron": lambda c: c in ("fp_sing_rate", "fp_plur_rate", "second_person_rate", "contraction_rate"),
    "disc": lambda c: c in ("hedge_rate", "booster_rate", "discourse_rate"),
    "punct": lambda c: c.startswith("p_"),
    "read": lambda c: c in ("flesch_ease", "fk_grade", "gunning_fog"),
    "func": lambda c: c.startswith("fw_"),
}


def group_of(col):
    for g, fn in GROUPS.items():
        if fn(col):
            return g
    return "other"


if __name__ == "__main__":
    import time
    full = pd.read_parquet("artifacts/corpus_full.parquet")
    matched = pd.read_parquet("artifacts/corpus_matched.parquet")
    extra = full[full.dataset.isin(["raw_llm", "toefl"])]
    rows = pd.concat([matched, extra], ignore_index=True).drop_duplicates("uid")
    t0 = time.time()
    F = extract_stylo(rows.text, n_process=1)
    F.insert(0, "uid", rows.uid.values)
    F.to_parquet("artifacts/features_stylo.parquet", index=False)
    print(f"{len(F)} rows x {F.shape[1]-1} features in {time.time()-t0:.0f}s")
    print("groups:", pd.Series([group_of(c) for c in F.columns[1:]]).value_counts().to_dict())
    print("NaN cells:", int(F.isna().sum().sum()))
