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
        for env, url in G.PROVIDERS.items():
            key = keys.get(env) or os.environ.get(env)
            if not key:
                continue
            c = OpenAI(base_url=url, api_key=key, timeout=90, max_retries=0)
            try:
                mids = [m.id.replace("models/", "") for m in c.models.list().data]
            except Exception:
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
                text, fin = G.call(self.model_client[m], m, prompt, max_tokens, self.stats, lambda *_: None)
            except Exception:
                continue
            text = clean_generated(text, fin) if text else text
            if text and len(text.split()) >= 3:
                self.last_model = m
                return text
            self.dead[m] = time.time()
        return None

    def topic(self, text):
        out = self.chat("In 3 to 8 words, what is the topic of this essay? Reply with ONLY the topic phrase, "
                        "no quotes, no full stop.\n\n" + text[:3000], max_tokens=40)
        if not out:
            return None
        return out.strip().strip('"').strip("'").rstrip(".").splitlines()[0][:80]


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
