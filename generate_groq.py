"""
DATA EXPANSION via Groq — run on the LOCAL machine (needs internet + .env).

  python src/generate_groq.py --list-models      # see what your key can use
  python src/generate_groq.py --pilot            # 20 essays, then stop + summary
  python src/generate_groq.py                    # full run (resumable, Ctrl+C safe)

What it generates (per generator FAMILY, same job specs for every family so
leave-one-generator-out is balanced):
  plain      15 PERSUADE prompts x 4 phrasings (incl. the bare "Write an essay about X.")
  factorial  reduced tone x persona x evasion grid on the 15 prompts (135 jobs)
  newtopic   37 topics OUTSIDE the 15 (and outside raw_llm_essays.json, which is
             the held-out OOD set), plain + styled
  paraphrase hard negatives: an existing gpt-oss essay paraphrased / lightly
             edited by this family
Lengths vary: explicit targets 120/180/250/400/550 words on most jobs.

Rules: temperature 0.9-1.0, varied angle per call, NO fixed appended text.
Families are discovered live from client.models.list() (gpt-oss excluded: it is
already in the dataset). Output: data/generated/groq_essays.jsonl (one JSON per
line, appended immediately -> crash/limit safe; reruns skip finished jobs).
"""
import argparse
import hashlib
import itertools
import json
import os
import random
import re
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "generated" / "groq_essays.jsonl"
SEED = 7

PERSUADE_TOPICS = {
    '"A Cowboy Who Rode the Waves"': "joining the Seagoing Cowboys program to help after World War II",
    "Car-free cities": "limiting car usage in cities",
    "Cell phones at school": "whether students should be allowed to use cell phones at school",
    "Community service": "whether community service should be required for students",
    "Distance learning": "whether students benefit from attending classes from home through distance learning",
    "Does the electoral college work?": "whether the United States should keep the Electoral College or switch to a popular vote",
    "Driverless cars": "the development of driverless cars",
    "Exploring Venus": "whether studying Venus is a worthy pursuit despite the dangers",
    "Facial action coding system": "using facial expression recognition technology in classrooms",
    "Grades for extracurricular activities": "requiring at least a B average for students to take part in sports and activities",
    "Mandatory extracurricular activities": "requiring every student to take part in at least one extracurricular activity",
    "Phones and driving": "whether drivers should be able to use cell phones while driving",
    "Seeking multiple opinions": "why seeking advice from more than one person helps you make better choices",
    "Summer projects": "whether summer projects should be teacher-designed or student-designed",
    "The Face on Mars": "whether the Face on Mars is a natural landform or created by aliens",
}

# Outside the 15 prompts AND outside raw_llm_essays.json topics (kept clean as OOD test)
NEW_TOPICS = {
    "debate": ["universal basic income", "nuclear power as a climate solution", "school uniforms",
               "moving to a cashless society", "abolishing daylight saving time",
               "whether video games make players more aggressive", "removing books from school libraries",
               "compulsory voting", "animal testing for cosmetics", "opt-out organ donation",
               "tipping culture in restaurants", "government subsidies for electric vehicles",
               "paying college athletes", "screen time limits for young children",
               "open-plan offices", "overtourism in popular cities", "banning junk food advertising to children"],
    "narrative": ["starting at a new school", "learning to swim", "your first summer job",
                  "a story a grandparent told you", "volunteering at a shelter", "losing something important",
                  "a concert you will never forget", "a long train journey",
                  "building something with your own hands", "a friendship that slowly faded"],
    "explainer": ["how ocean tides work", "why leaves change color in autumn", "how the stock market works",
                  "how airplanes stay in the air", "how plastic recycling actually works", "why humans need sleep",
                  "how batteries store energy", "why earthquakes happen", "how antibiotics work",
                  "how data travels across the internet"],
}

PLAIN_PHRASINGS = [
    "Write an essay about {t}.",                      # bare: what a teacher will type
    "Write an essay on {t}.",
    "Can you write me an essay about {t}?",
    "Write a short essay discussing {t}.",
]
ANGLES = ["Take a clear position.", "Focus on the practical consequences.", "Consider both sides before concluding.",
          "Use a concrete example to open.", "Focus on who is most affected.", "Think about the long-term picture.",
          "Ground it in everyday life.", "Challenge a common assumption.", ""]
TONE = {
    "formal": ["Use a formal tone.", "Write in a formal, polished register.", "Keep the tone formal and measured."],
    "casual": ["Use a casual, conversational tone.", "Keep it relaxed and conversational.", "Write casually, like you're talking to a friend."],
    "academic": ["Use an academic tone.", "Write in an academic style with careful reasoning.", "Adopt a scholarly, analytical tone."],
}
PERSONA = {
    "student": ["Write it as a high school student.", "Write from the perspective of a student.", "You are a student writing this for class."],
    "non-native speaker": ["Write as a non-native English speaker would.", "Write it as someone for whom English is a second language.", "You are an English learner writing this."],
    "expert": ["Write as an expert on the subject.", "Write from the perspective of a specialist in this area.", "You are an experienced professional in this field."],
}
EVASION = {
    "sound_human": ["Make it sound like a human wrote it, not an AI.", "Avoid sounding like AI; make it read naturally human.",
                    "Write so it won't be detected as AI-generated."],
    "none": [""],
}
LENGTH_TARGETS = [120, 180, 250, 400, 550]
EXCLUDE = ("whisper", "tts", "guard", "compound", "gpt-oss", "playai", "orpheus", "distil", "safeguard",
           "embedding", "imagen", "veo", "aqa", "image", "audio", "live", "lyria", "learnlm", "nano", "robotics",
           "computer-use", "native", "1b", "e2b", "e4b")
PROVIDERS = {  # env var -> OpenAI-compatible base URL (Gemini: ai.google.dev/gemini-api/docs/openai)
    "GROQ_API_KEY": "https://api.groq.com/openai/v1",
    "GEMINI_API_KEY": "https://generativelanguage.googleapis.com/v1beta/openai/",
}


def family_of(model_id):
    m = model_id.lower()
    for key, fam in [("llama", "meta-llama"), ("qwen", "qwen"), ("kimi", "moonshot"), ("moonshot", "moonshot"),
                     ("gemma", "google-gemma"), ("gemini", "google-gemini"), ("mistral", "mistral"),
                     ("mixtral", "mistral"), ("deepseek", "deepseek"), ("allam", "allam")]:
        if key in m:
            return fam
    return None  # unknown family (e.g. agent/preview products): skip rather than guess


def size_score(model_id):
    m = model_id.lower()
    if "gemini" in m:  # prefer stable flash models (generous free tier) over pro/preview
        return 50 + 10 * ("flash" in m) - 20 * ("preview" in m or "exp" in m) - 30 * ("pro" in m) - 8 * ("lite" in m) \
               + max([float(x) for x in re.findall(r"gemini-(\d+(?:\.\d+)?)", m)] or [0])
    if "gemma" in m:  # newer generation first (gemma-4 > gemma-3), then size
        ver = max([float(x) for x in re.findall(r"gemma-(\d+)", m)] or [0])
        return 100 * ver + max([float(x) for x in re.findall(r"(\d+)b", m)] or [0])
    nums = [float(x) for x in re.findall(r"(\d+(?:\.\d+)?)b", model_id.lower())]
    bonus = 5 if any(k in model_id.lower() for k in ("versatile", "maverick", "instruct")) else 0
    return (max(nums) if nums else 1) + bonus


def pick_models(ids):
    by_fam = {}
    for mid in ids:
        if any(x in mid.lower() for x in EXCLUDE) or family_of(mid) is None:
            continue
        by_fam.setdefault(family_of(mid), []).append(mid)
    return {fam: max(ms, key=size_score) for fam, ms in by_fam.items()}


def rank_models(ids):
    """Per family: all usable models, best first (fallbacks if the best keeps failing)."""
    by_fam = {}
    for mid in ids:
        if any(x in mid.lower() for x in EXCLUDE) or family_of(mid) is None:
            continue
        by_fam.setdefault(family_of(mid), []).append(mid)
    return {fam: sorted(ms, key=size_score, reverse=True) for fam, ms in by_fam.items()}


def build_jobs(gptoss_sources):
    """Same job list for every family. Priority: plain > factorial > newtopic > paraphrase."""
    rnd = random.Random(SEED)
    jobs = []

    def add(kind, topic, prompt, prio, **meta):
        jobs.append(dict(kind=kind, topic=topic, prompt=prompt, prio=prio, **meta))

    for canon, desc in PERSUADE_TOPICS.items():
        for i, ph in enumerate(PLAIN_PHRASINGS):
            p = ph.format(t=desc)
            L = None
            if i >= 2:
                L = rnd.choice(LENGTH_TARGETS)
                p += f" Aim for about {L} words."
            add("plain", canon, p, 0, length_target=L, bare=(i == 0))

    grid = list(itertools.product(TONE, PERSONA, EVASION))           # 18 combos
    topics = list(PERSUADE_TOPICS)
    for ci, (tone, persona, ev) in enumerate(grid):                   # 18 x ~7.5 = 135, every topic covered
        for k in range(8 if ci % 2 == 0 else 7):
            canon = topics[(ci * 7 + k) % 15]
            length = rnd.choice(["concise", "elaborate"])
            L = rnd.choice([150, 200, 250]) if length == "concise" else rnd.choice([400, 500, 600])
            parts = [f"Write an essay about {PERSUADE_TOPICS[canon]}.", rnd.choice(TONE[tone]),
                     rnd.choice(PERSONA[persona]), rnd.choice(EVASION[ev]), rnd.choice(ANGLES),
                     f"Aim for about {L} words."]
            add("factorial", canon, " ".join(x for x in parts if x), 1,
                tone=tone, persona=persona, evasion=ev, length=length, length_target=L)

    for genre, ts in NEW_TOPICS.items():
        for t in ts:
            add("newtopic", t, f"Write an essay about {t}.", 2, genre=genre, bare=True)
            L = rnd.choice(LENGTH_TARGETS)
            add("newtopic", t, f"{rnd.choice(PLAIN_PHRASINGS[1:]).format(t=t)} {rnd.choice(ANGLES)} Aim for about {L} words.".replace("  ", " "),
                2, genre=genre, length_target=L)
            tone, persona = rnd.choice(list(TONE)), rnd.choice(list(PERSONA))
            add("newtopic", t, f"Write an essay about {t}. {rnd.choice(TONE[tone])} {rnd.choice(PERSONA[persona])}",
                2, genre=genre, tone=tone, persona=persona)

    for i, src in enumerate(gptoss_sources[:80]):
        if i % 2 == 0:
            p = ("Paraphrase the following essay completely in your own words. Keep the meaning and roughly the "
                 "same length, but change the wording and sentence structure. Return only the essay.\n\n" + src["text"])
            kind = "paraphrase"
        else:
            p = ("Lightly edit the following essay so it reads more naturally, the way a real person would revise "
                 "their own draft. Keep most of it. Return only the edited essay.\n\n" + src["text"])
            kind = "light_edit"
        add(kind, src["topic"], p, 3, source_ref=src["ref"])

    for j in jobs:
        key = f"{j['kind']}|{j['topic']}|{j.get('source_ref', '')}|{j['prompt']}"
        j["job_key"] = hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]
    assert len({j["job_key"] for j in jobs}) == len(jobs), "duplicate job keys"
    return jobs


_THINK = re.compile(r"<think>.*?</think>", re.S)
_PREAMBLE = re.compile(r"^\s*(sure|certainly|here(?:'s| is)|of course)[^\n]*:\s*\n", re.I)


def clean_output(s):
    s = _THINK.sub("", s or "").strip()
    return _PREAMBLE.sub("", s).strip()


class Writer:
    def __init__(self, path):
        self.path, self.lock = path, threading.Lock()
        path.parent.mkdir(parents=True, exist_ok=True)

    def done_keys(self):
        if not self.path.exists():
            return set()
        keys = set()
        with open(self.path, encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                    if r.get("text"):
                        keys.add((r.get("family") or family_of(r["generator"]), r["job_key"]))
                except json.JSONDecodeError:
                    pass
        return keys

    def write(self, rec):
        with self.lock, open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _thinking_options(model):
    """Disable hidden 'thinking' (it burns rate-limit tokens and can return empty text).
    Groq qwen3.x: reasoning_effort='none' = instruct mode (console.groq.com/docs/model/qwen/qwen3.8-27b).
    Falls back step by step if a provider rejects the parameter."""
    m = model.lower()
    if "qwen3" in m or "gemini-2.5" in m or "gemini-3" in m or "deepseek-r1" in m:
        return [{"reasoning_effort": "none"}, {"reasoning_format": "hidden"}, None]
    return [None]


def call(client, model, prompt, max_tokens, stats, log):
    opts = _thinking_options(model)
    delay = 5
    for attempt in range(10):
        extra = opts[0]
        try:
            kw = dict(model=model, messages=[{"role": "user", "content": prompt}],
                      temperature=random.uniform(0.9, 1.0), max_tokens=max_tokens)
            if extra:
                kw["extra_body"] = extra
            r = client.chat.completions.create(**kw)
            text = clean_output(r.choices[0].message.content)
            u = getattr(r, "usage", None)
            if u:
                stats["in"] += u.prompt_tokens or 0
                stats["out"] += u.completion_tokens or 0
            if not text and r.choices[0].finish_reason == "length" and max_tokens < 4000:
                max_tokens *= 2
                continue
            return text, r.choices[0].finish_reason
        except Exception as e:
            code = getattr(e, "status_code", None)
            if code == 400 and len(opts) > 1:      # provider rejected the thinking option: try the next one
                log(f"[{model}] option {opts[0]} rejected, falling back")
                opts = opts[1:]
                continue
            if code in (400, 401, 403, 404):       # invalid request / no access: retrying cannot help
                log(f"[{model}] {code}: {str(e)[:120]}")
                return None, "bad_request"
            wait = delay
            resp = getattr(e, "response", None)
            if resp is not None:
                ra = resp.headers.get("retry-after")
                if ra:
                    try:
                        wait = float(ra) + 1
                    except ValueError:
                        pass
            if code == 429 and wait > 600:
                log(f"[{model}] daily/long rate limit (retry-after {wait:.0f}s); pausing this model")
                return None, "rate_limited_long"
            transient = code in (429, 500, 502, 503, 504) or "Timeout" in type(e).__name__
            if transient and attempt >= 2:            # 3 rate-limit/overload/timeout hits on one essay: give up fast
                return None, "rate_limited" if code == 429 else "server_error"
            log(f"[{model}] {code or type(e).__name__}: retry {attempt + 1} in {wait:.0f}s")
            time.sleep(wait)
            delay = min(delay * 2, 120)
    return None, "failed"


def worker(model_client, fam, models, jobs, writer, done, stats, limit, log):
    """One thread per family. If the current model fails 3 jobs in a row (rate limit wall,
    timeouts, invalid requests), switch to the next model of the SAME family."""
    mi, n, fails = 0, 0, 0
    model = models[mi]
    log(f"[{fam}] using {model}" + (f" (fallbacks: {', '.join(models[1:3])})" if len(models) > 1 else ""))
    for j in jobs:
        if (fam, j["job_key"]) in done:
            continue
        if limit is not None and n >= limit:
            break
        mt = 1800 if j["kind"] in ("paraphrase", "light_edit") or (j.get("length_target") or 0) >= 400 else 1100
        text, finish = call(model_client[model], model, j["prompt"], mt, stats, log)
        if not text:
            fails += 1
            if fails >= 2 or finish == "rate_limited_long":
                mi += 1
                if mi >= len(models):   # all models of this family exhausted: rest, then start over
                    log(f"[{fam}] all models rate-limited; pausing 10 min, then retrying from {models[0]}")
                    time.sleep(600)
                    mi = 0
                model, fails = models[mi], 0
                log(f"[{fam}] now using {model}")
            continue
        fails = 0
        rec = {k: v for k, v in j.items() if k != "prio"}
        rec.update(generator=model, family=fam, text=text, finish_reason=finish, ts=time.time())
        writer.write(rec)
        n += 1
        stats["n"] += 1
        if stats["n"] % 10 == 0:
            log(f"progress: {stats['n']} essays | tokens in={stats['in']:,} out={stats['out']:,}")
    log(f"[{fam}:{model}] finished this run with {n} new essays")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list-models", action="store_true")
    ap.add_argument("--pilot", action="store_true", help="20 essays total, then summary")
    ap.add_argument("--models", help="comma-separated model ids to force (default: auto one per family)")
    ap.add_argument("--max-families", type=int, default=6)
    args = ap.parse_args()

    try:
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env")
    except ImportError:
        print("Tip: pip install python-dotenv  (or set GROQ_API_KEY in the environment)")
    from openai import OpenAI
    clients, ids, model_client = {}, [], {}
    for env, url in PROVIDERS.items():
        key = os.environ.get(env)
        if not key:
            continue
        c = OpenAI(base_url=url, api_key=key, timeout=240, max_retries=0)
        try:
            mids = sorted(m.id.replace("models/", "") for m in c.models.list().data)
        except Exception as e:
            print(f"[{env}] could not list models: {e}")
            continue
        print(f"[{env}] {len(mids)} models")
        ids += mids
        for m in mids:
            model_client[m] = c
    if not model_client:
        sys.exit("No working API key found (.env needs GROQ_API_KEY and/or GEMINI_API_KEY).")
    ranked = rank_models(ids)
    if args.models:
        ranked = {family_of(m): [m] for m in args.models.split(",")}
    ranked = dict(sorted(ranked.items(), key=lambda kv: -size_score(kv[1][0]))[: args.max_families])
    chosen = {f: ms[0] for f, ms in ranked.items()}
    if args.list_models:
        print("All models:\n  " + "\n  ".join(ids))
        print("\nChosen (one per family, gpt-oss excluded):")
        for f, ms in ranked.items():
            print(f"  {f:14s} -> {ms[0]}   fallbacks: {ms[1:4]}")
        return

    src = ROOT / "data" / "factorial_essays_real_api_final.csv"
    import csv
    gptoss = []
    with open(src, encoding="utf-8") as f:
        for i, r in enumerate(csv.DictReader(f)):
            if "gpt-oss" in r["model_used"]:
                gptoss.append({"text": r["text"], "topic": r["topic"], "ref": f"factorial_api_row{i}"})
    random.Random(SEED).shuffle(gptoss)
    jobs = sorted(build_jobs(gptoss), key=lambda j: j["prio"])

    writer = Writer(OUT)
    done = writer.done_keys()
    total = len(jobs) * len(chosen)
    print(f"Families/models: {chosen}")
    print(f"Jobs per family: {len(jobs)}  | total: {total}  | already done: {len(done)}")
    print("Rough token budget: ~{:,} output tokens total (free tier = $0; time is limited by rate limits)".format(total * 500))

    lock = threading.Lock()

    def log(msg):
        with lock:
            print(time.strftime("%H:%M:%S"), msg, flush=True)

    stats = {"n": 0, "in": 0, "out": 0}
    per_family_limit = max(1, 20 // len(chosen)) if args.pilot else None
    if args.pilot:  # pilot: spread over job kinds
        rnd = random.Random(1)
        jobs = rnd.sample([j for j in jobs if j["kind"] == "plain"], 2) + \
               rnd.sample([j for j in jobs if j["kind"] == "factorial"], 2) + \
               rnd.sample([j for j in jobs if j["kind"] == "newtopic"], 1) + \
               rnd.sample([j for j in jobs if j["kind"] in ("paraphrase", "light_edit")], 1)
    threads = [threading.Thread(target=worker, args=(model_client, f, ms, jobs, writer, done, stats, per_family_limit, log))
               for f, ms in ranked.items()]
    try:
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    except KeyboardInterrupt:
        print("\nStopped. Progress is saved; rerun the same command to resume.")
        os._exit(0)

    recs = [json.loads(l) for l in open(OUT, encoding="utf-8")] if OUT.exists() else []
    print(f"\nDone this run: {stats['n']} new | file now has {len(recs)} essays")
    from collections import Counter
    print("by family:", dict(Counter(r["family"] for r in recs)))
    print("by kind  :", dict(Counter(r["kind"] for r in recs)))
    if recs:
        wc = [len(r["text"].split()) for r in recs]
        print(f"words: min {min(wc)}  median {sorted(wc)[len(wc)//2]}  max {max(wc)}")
    if args.pilot:
        print("\nPilot OK if every family produced essays with sensible lengths. Then run without --pilot.")


if __name__ == "__main__":
    main()
