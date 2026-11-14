"""
Small LLM helper for the app: discovers which chat models the provided keys can use
(never a hard-coded model name), prefers stronger families, and falls back automatically
on rate limits / overload. Reuses the generator's tested call + ranking logic.
"""
import os
import random
import time

import generate_groq as G

FAMILY_PREFERENCE = ["google-gemini", "google-gemma", "qwen", "allam", "meta-llama"]


class LLM:
    def __init__(self, keys=None):
        from openai import OpenAI
        keys = keys or {}
        self.model_client, ids = {}, []
        self.providers, self.log = {}, []      # diagnostics shown in the app (never contains keys)
        for env, url in G.PROVIDERS.items():
            key = keys.get(env) or os.environ.get(env)
            if not key:
                continue
            c = OpenAI(base_url=url, api_key=key, timeout=45, max_retries=0)
            try:
                mids = [m.id.replace("models/", "") for m in c.models.list().data]
                self.providers[env] = f"ok ({len(mids)} models)"
            except Exception as e:
                self.providers[env] = f"key rejected or unreachable: {getattr(e, 'status_code', None) or type(e).__name__}"
                continue
            ids += mids
            for m in mids:
                self.model_client[m] = c
        ranked = G.rank_models(ids)
        fams = [f for f in FAMILY_PREFERENCE if ranked.get(f)] + [f for f in ranked if f not in FAMILY_PREFERENCE and ranked[f]]
        # round-robin across families: best model of EACH family first, so one provider's exhausted
        # quota (e.g. every Gemini variant rate-limited) falls through to another provider quickly
        depth = max((len(ranked[f]) for f in fams), default=0)
        self.models = [ranked[f][i] for i in range(depth) for f in fams if i < len(ranked[f])]
        self.stats = {"n": 0, "in": 0, "out": 0}
        self.dead = {}                    # model -> time it failed; retried after 10 minutes
        self.last_model = None

    @property
    def available(self):
        return bool(self.models)

    def chat(self, prompt, max_tokens=1100, max_models=6):
        from build_corpus import clean_generated   # strip <thought> traces / cut-off tails, as in training data
        tried = 0
        for m in self.models:
            if time.time() - self.dead.get(m, 0) < 600:   # failed recently (quota / errors): skip, stay fast
                continue
            if tried >= max_models:
                break
            tried += 1
            try:
                text, fin = G.call(self.model_client[m], m, prompt, max_tokens, self.stats,
                                   lambda msg: self.log.append(str(msg)[:160]))
            except Exception as e:
                self.log.append(f"[{m}] {type(e).__name__}"); self.dead[m] = time.time()
                continue
            text = clean_generated(text, fin) if text else text
            if text and len(text.split()) >= 3:
                self.last_model = m
                return text
            self.log.append(f"[{m}] no usable text ({fin})")
            self.dead[m] = time.time()
        return None

    def diagnostics(self, n=8):
        lines = [f"{k}: {v}" for k, v in self.providers.items()]
        lines.append("models in use (first 6): " + ", ".join(self.models[:6]) if self.models else "no usable models")
        return lines + self.log[-n:]

    def topic(self, text):
        out = self.chat("What assignment topic was this essay written on? Phrase it the way it would appear in "
                        "an essay assignment, in 4 to 14 words, e.g. 'whether homework should be banned in "
                        "primary schools' or 'the impact of social media on teenagers' sleep'. Reply with ONLY "
                        "the topic phrase, no quotes, no full stop.\n\n" + text[:3000], max_tokens=60)
        if not out:
            return None
        t = out.strip().strip('"').strip("'").rstrip(".").splitlines()[0][:80]
        return tidy_topic(t)


WRONG_PROMPTS = [
    "Write an essay about the history of the bicycle. Use a formal tone.",
    "Write an essay about why people enjoy gardening. Write it as an expert on the subject.",
    "Write an essay about the importance of sleep for teenagers. Use a casual tone.",
    "Write an essay about whether museums should be free. Use an academic tone.",
]


def wrong_prompt_for(topic, seed=0):
    rnd = random.Random(seed)
    options = [p for p in WRONG_PROMPTS if not any(w in p.lower() for w in str(topic).lower().split()[:2])]
    return rnd.choice(options or WRONG_PROMPTS)


def tidy_topic(t):
    """'Climate migration: A multidimensional challenge...' -> 'climate migration'.
    Drop a title-style subtitle after a colon and lowercase the first word unless it is an acronym/name-like."""
    t = t.lstrip("# ").strip()
    if ":" in t and len(t.split(":")[0].split()) >= 2 and len(t.split(":", 1)[1].split()) >= 5:
        t = t.split(":")[0].strip()
    w = t.split(" ", 1)
    if w and w[0][:1].isupper() and not w[0].isupper() and not (len(w[0]) > 1 and w[0][1:].lower() != w[0][1:]):
        t = w[0][0].lower() + w[0][1:] + (" " + w[1] if len(w) > 1 else "")
    return t
