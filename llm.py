"""
Small LLM helper for the app: discovers which chat models the provided keys can use
(never a hard-coded model name), prefers stronger families, and falls back automatically
on rate limits / overload. Reuses the generator's tested call + ranking logic.
"""
import os
import random

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
        self.models = [m for fam in FAMILY_PREFERENCE for m in ranked.get(fam, [])]
        self.models += [m for fam, ms in ranked.items() if fam not in FAMILY_PREFERENCE for m in ms]
        self.stats = {"n": 0, "in": 0, "out": 0}
        self.last_model = None

    @property
    def available(self):
        return bool(self.models)

    def chat(self, prompt, max_tokens=1100, max_models=4):
        for m in self.models[:max_models]:
            text, _ = G.call(self.model_client[m], m, prompt, max_tokens, self.stats, lambda *_: None)
            if text:
                self.last_model = m
                return text
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
