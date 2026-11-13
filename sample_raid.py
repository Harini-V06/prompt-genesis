"""
Stream-sample RAID (Dugan et al., ACL 2024; MIT license) WITHOUT saving the full file.
Source: https://dataset.raid-bench.xyz/train_none.csv  (802 MB, no adversarial attacks)

Why: (1) 11 extra AI generators (GPT-4, ChatGPT, Llama-2-chat, Mistral, Cohere, MPT, ...)
make leave-one-generator-out meaningful; (2) ADULT human prose (news, books, abstracts,
reviews, wiki) counters the confound that our essay humans are grade 6-12 students.

Keeps a seeded reservoir sample of up to PER_CELL rows per (model, domain), only for
prose domains (code / poetry / recipes / non-English excluded), only texts of >= 100 words.
Prints every distinct model label seen, so the presence of human rows is verified from
the data itself.

  python src/sample_raid.py            # ~5-15 min, mostly download time
Output: data/raid_sample.jsonl
"""
import csv
import io
import json
import random
import sys
import time
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

URL = "https://dataset.raid-bench.xyz/train_none.csv"
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "raid_sample.jsonl"
PER_CELL = 150
KEEP_DOMAINS = {"news", "books", "abstracts", "reviews", "wiki", "reddit"}
SEED = 13


def main():
    csv.field_size_limit(min(sys.maxsize, 2**31 - 1))
    rng = random.Random(SEED)
    res, seen = defaultdict(list), defaultdict(int)
    models, domains, attacks = Counter(), Counter(), Counter()
    req = urllib.request.Request(URL, headers={"User-Agent": "prompt-genesis-student-project"})
    t0, n = time.time(), 0
    with urllib.request.urlopen(req, timeout=120) as r:
        reader = csv.DictReader(io.TextIOWrapper(r, encoding="utf-8", newline=""))
        print("columns:", reader.fieldnames, flush=True)
        text_col = "generation" if "generation" in reader.fieldnames else reader.fieldnames[-1]
        for row in reader:
            n += 1
            m, d = row.get("model", "?"), row.get("domain", "?")
            models[m] += 1; domains[d] += 1; attacks[row.get("attack", "?")] += 1
            if d not in KEEP_DOMAINS or row.get("attack", "none") not in ("none", ""):
                continue
            text = row.get(text_col) or ""
            if len(text.split()) < 100:
                continue
            key = (m, d)
            seen[key] += 1
            rec = {"model": m, "domain": d, "decoding": row.get("decoding"),
                   "repetition_penalty": row.get("repetition_penalty"),
                   "source_id": row.get("source_id"), "title": row.get("title"), "text": text}
            if len(res[key]) < PER_CELL:
                res[key].append(rec)
            else:                                   # reservoir sampling
                j = rng.randint(0, seen[key] - 1)
                if j < PER_CELL:
                    res[key][j] = rec
            if n % 200000 == 0:
                print(f"{n:,} rows read ({(time.time()-t0)/60:.1f} min)", flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        for recs in res.values():
            for rec in recs:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"\nread {n:,} rows in {(time.time()-t0)/60:.1f} min")
    print("model labels seen:", dict(models))
    print("domains seen:", dict(domains))
    print("attacks seen:", dict(attacks))
    print(f"saved {sum(map(len, res.values()))} rows -> {OUT}")
    print("human rows kept:", sum(len(v) for k, v in res.items() if k[0] == "human"))


if __name__ == "__main__":
    main()
