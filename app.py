"""
Prompt Genesis: Streamlit app.
  Page 1  Detect & Reconstruct   Guard verdict (calibrated, can abstain) + plain-English reasons,
                                 Detective settings + reconstructed prompt
  Page 2  Verify                 regenerate x3 with the reconstructed prompt, Match Score on the
                                 floor -> ceiling scale
Demo mode replays cached, pre-run examples (no models / API needed), the live-demo backup.
All inference goes through src/pipeline.py, the same code used for the reported results.
"""
import json
import os
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))            # flat layout (all files in one folder) also works
sys.path.insert(0, str(ROOT / "src"))
ART = ROOT / "artifacts" if (ROOT / "artifacts").exists() else ROOT

st.set_page_config(page_title="Prompt Genesis", page_icon="P", layout="wide", initial_sidebar_state="expanded")

# ------------------------------------------------------------------ style
st.markdown("""
<style>
:root { --bg:#000; --card:#0b0b0b; --line:#dc2626; --dim:rgba(220,38,38,.35); --txt:#fff; --mut:#a3a3a3; --red:#ef4444; }
.stApp { background: var(--bg); color: var(--txt); }
#MainMenu, footer, [data-testid="stToolbarActions"], [data-testid="stDecoration"] { visibility: hidden; }
header[data-testid="stHeader"] { background: transparent; }   /* keep it: it holds the open-sidebar arrow */
[data-testid="stSidebarCollapsedControl"], [data-testid="stExpandSidebarButton"] { visibility: visible !important; }
.block-container { padding-top: 1.6rem; max-width: 1150px; }
[data-testid="stSidebar"] { background:#050505; border-right:1px solid var(--dim); }
h1,h2,h3,h4,p,li,label,span { color: var(--txt); }
.wm { font-weight:800; letter-spacing:.14em; font-size:1.9rem; margin:0; }
.wm .b { color: var(--red); }
.tag { color: var(--mut); margin:.15rem 0 1.2rem 0; font-size:.98rem; }
.card { background:var(--card); border:1px solid var(--dim); border-radius:12px; padding:1.1rem 1.25rem; margin-bottom:1rem; }
.card h4 { margin:0 0 .6rem 0; font-size:.78rem; letter-spacing:.14em; text-transform:uppercase; color:var(--mut); }
.verdict { border-left:4px solid var(--line); }
.verdict.human { border-left-color:#fff; }
.verdict.unsure { border-left-color:#737373; }
.vt { font-size:1.6rem; font-weight:800; margin:0; }
.vt.big { font-size:3rem; line-height:1.1; color:#dc2626; }
textarea:disabled { color:#e5e5e5 !important; -webkit-text-fill-color:#e5e5e5 !important; opacity:1 !important; }
.vs { color:var(--mut); margin:.2rem 0 0 0; }
.gauge { position:relative; height:14px; border-radius:7px; margin:1rem 0 .3rem 0;
         background: linear-gradient(90deg,#fff 0%, #525252 50%, #dc2626 100%); }
.gauge .band { position:absolute; top:-3px; height:20px; background:rgba(120,120,120,.45); border:1px dashed #a3a3a3; border-radius:4px; }
.gauge .mk { position:absolute; top:-6px; width:4px; height:26px; background:#facc15; border-radius:2px; }
.gl { display:flex; justify-content:space-between; color:var(--mut); font-size:.78rem; }
.reason { padding:.55rem 0; border-bottom:1px solid #1f1f1f; }
.reason:last-child { border-bottom:none; }
.pill { display:inline-block; padding:.1rem .55rem; border-radius:999px; font-size:.74rem; font-weight:700; margin-right:.5rem; }
.pill.ai { background:rgba(239,68,68,.18); color:#f87171; border:1px solid rgba(239,68,68,.5); }
.pill.hu { background:rgba(255,255,255,.08); color:#e5e5e5; border:1px solid #525252; }
.prompt { font-family: ui-monospace, Menlo, Consolas, monospace; background:#111; border:1px solid #262626;
          border-radius:8px; padding:.8rem 1rem; font-size:.95rem; }
.badges > span { display:inline-block; margin:.25rem .4rem .25rem 0; padding:.25rem .65rem; border-radius:6px;
               border:1px solid var(--dim); font-size:.85rem; }
.badges > span b { color: var(--mut); font-weight:600; margin-right:.3rem; }
.badges > span.unsure { opacity:.45; border-style:dashed; }
.scale { position:relative; height:16px; background:#171717; border:1px solid #333; border-radius:8px; margin:1.6rem 0 .4rem 0; }
.scale .fill { position:absolute; height:100%; border-radius:8px; background:linear-gradient(90deg,#3f3f46,#dc2626); }
.scale .lab { position:absolute; top:-1.35rem; font-size:.72rem; color:var(--mut); transform:translateX(-50%); white-space:nowrap; }
.scale .tick { position:absolute; top:-4px; width:2px; height:24px; background:#a3a3a3; }
.scale .you { position:absolute; top:-7px; width:5px; height:30px; background:#facc15; border-radius:2px; }
.bar { margin:.45rem 0; }
.bar .t { display:flex; justify-content:space-between; font-size:.88rem; }
.bar .tr { height:8px; background:#1c1c1c; border-radius:4px; overflow:hidden; margin-top:.25rem; }
.bar .tr div { height:100%; background:#dc2626; }
.small { color:var(--mut); font-size:.85rem; }
.stButton>button { background:#dc2626; color:#fff; border:none; font-weight:700; border-radius:8px; padding:.5rem 1.3rem; }
.stButton>button:hover { background:#b91c1c; color:#fff; }
textarea { background:#0a0a0a !important; color:#fff !important; border:1px solid var(--dim) !important; }
</style>
""", unsafe_allow_html=True)


def card(title, body_html, extra_cls=""):
    st.markdown(f'<div class="card {extra_cls}"><h4>{title}</h4>{body_html}</div>', unsafe_allow_html=True)


# ------------------------------------------------------------------ resources
def secret(name):
    try:
        if name in st.secrets:
            return st.secrets[name]
    except Exception:
        pass
    return os.environ.get(name)


@st.cache_resource(show_spinner=False)
def load_models():
    import pipeline
    return pipeline.Models()


@st.cache_resource(show_spinner=False)
def build_llm(gemini_key, groq_key):
    from llm import LLM
    keys = {"GEMINI_API_KEY": gemini_key, "GROQ_API_KEY": groq_key}
    return LLM({k: v for k, v in keys.items() if v})


def load_llm(gemini_key, groq_key, _status=False):
    """None when no usable model; _status=True returns the object anyway so the sidebar can say why."""
    llm = build_llm(gemini_key, groq_key)
    return llm if (llm.available or _status) else None


@st.cache_data(show_spinner=False, max_entries=200)
def analyze_cached(text, use_llm_topic, _gk, _qk):     # keys start with "_" so they are never hashed/stored
    """Same essay (and same key setup) -> instant second time; results are deterministic except the LLM topic."""
    import pipeline
    llm = load_llm(_gk, _qk) if use_llm_topic else None
    return pipeline.analyze(text, load_models(), topic_fn=llm.topic if llm else None)


@st.cache_data(show_spinner=False)
def load_demo():
    p = ART / "demo_cache.json"
    return json.load(open(p, encoding="utf-8")) if p.exists() else {}


# ------------------------------------------------------------------ sidebar
st.sidebar.markdown('<p class="wm" style="font-size:1.15rem">PROMPT<span class="b">GENESIS</span></p>', unsafe_allow_html=True)
page = st.sidebar.radio("Page", ["Detect & Reconstruct", "Verify", "About the model"], label_visibility="collapsed")
demo = load_demo()
use_demo = st.sidebar.toggle("Demo mode (pre-run examples)", value=False,
                             help="Replays cached results: works without models or API keys.") if demo else False
with st.sidebar.expander("API keys (for topic + regeneration)"):
    gk = (st.text_input("Gemini API key", type="password", value="") or secret("GEMINI_API_KEY") or "").strip()
    qk = (st.text_input("Groq API key", type="password", value="") or secret("GROQ_API_KEY") or "").strip()
    st.caption("Keys stay in this session only. Without a key, the topic comes from keywords and Verify is disabled.")
    if gk or qk:
        _l = load_llm(gk, qk, _status=True)
        for k, v in (_l.providers.items() if _l else []):
            st.caption(f"{'Gemini' if 'GEMINI' in k else 'Groq'}: {v}")

st.markdown('<p class="wm">PROMPT<span class="b">GENESIS</span></p>'
            '<p class="tag">Was this essay written by a person or an AI? If AI, what prompt most likely produced it, '
            'and can we rebuild it?</p>', unsafe_allow_html=True)

ss = st.session_state
ss.setdefault("analysis", None)
ss.setdefault("text", "")
ss.setdefault("verify", None)
ss.setdefault("demo_key", None)


# ------------------------------------------------------------------ renderers
def pct(x):
    """Calibrated probabilities can hit exactly 0/1; no detector is certain, so cap the wording."""
    return "> 99%" if x > 0.99 else ("< 1%" if x < 0.01 else f"{x:.0%}")


def render_verdict(a):
    p, band = a["p_ai"], a["band"]
    v = a["verdict"]
    if v == "AI":
        cls, title, sub = "", "Likely AI-generated", f"{pct(p)} probability of AI (calibrated)"
    elif v == "human":
        cls, title, sub = "human", "Likely human-written", f"{pct(1 - p)} probability of human (calibrated)"
    else:
        cls, title, sub = "unsure", "Insufficient evidence", (f"P(AI) = {p:.0%}, inside the uncertainty band "
                                                              f"[{0.5 - band:.0%}, {0.5 + band:.0%}]. The model abstains "
                                                              "rather than guess.")
    lo, hi = (0.5 - band) * 100, (0.5 + band) * 100
    g = (f'<div class="gauge"><div class="band" style="left:{lo}%;width:{max(hi - lo, 0.6)}%"></div>'
         f'<div class="mk" style="left:calc({p * 100}% - 2px)"></div></div>'
         '<div class="gl"><span>Human</span><span>uncertain</span><span>AI</span></div>')
    card("Verdict", f'<p class="vt">{title}</p><p class="vs">{sub} &middot; {a["n_words"]} words</p>{g}', f"verdict {cls}")


def render_reasons(a):
    rows = "".join(
        f'<div class="reason"><span class="pill {"ai" if r["toward"] == "AI" else "hu"}">'
        f'{"points to AI" if r["toward"] == "AI" else "points to human"}</span>{r["sentence"]}'
        f'<span class="small"> &nbsp;({r["label"]})</span></div>' for r in a["reasons"])
    card("Why: strongest evidence (SHAP)", rows + '<p class="small" style="margin-top:.6rem">Each line is one '
         "feature's contribution to this verdict, not a rule. Several weak signals add up.</p>")


def render_prompt(a):
    if "settings" not in a:
        return
    s = a["settings"]
    src = "detected by the LLM" if a.get("topic_source") == "llm" else "main noun phrases (add an API key for a better topic)"
    measured = "".join(f"<li>{c}</li>" for c in a.get("style_clauses", []))
    guesses = "".join(
        f'<span class="unsure"><b>{k}</b>{d["value"]} <span class="small">({d["prob"]:.0%})</span></span>'
        for k, d in s.items() if k in ("tone", "persona"))
    card("Reconstructed prompt",
         f'<div class="prompt">{a["prompt"]}</div>'
         f'<p class="small" style="margin-top:.7rem"><b>Topic</b> ({src}): {a["topic"]}</p>'
         f'<p class="small"><b>Measured from the text</b> (counted, not guessed):</p><ul class="small">{measured}</ul>'
         f'<p class="small"><b>Style guesses</b> (classifier, NOT used in the prompt):</p><div class="badges">{guesses}</div>'
         f'<p class="small">The guesses are about 85% accurate on essays made through AI APIs but mostly wrong on essays '
         f'copied from chatbot apps such as ChatGPT, which add their own hidden instructions. So the prompt only uses '
         f'what can be measured; the guesses are shown for context.</p>')


def render_verify(r, original):
    if "error" in r:
        st.error(r["error"]); return
    s = r["summary"]
    bars = "".join(
        f'<div class="bar"><div class="t"><span>{lab}</span><span>{s[k]:.0%}</span></div>'
        f'<div class="tr"><div style="width:{s[k] * 100:.1f}%"></div></div></div>'
        for k, lab in (("words", "Words (TF-IDF + ROUGE-L)"), ("structure", "Structure (style profile)"),
                       ("ideas", "Ideas (sentence meaning, MiniLM)")) if k in s)
    fl, ce, m = r.get("floor"), r.get("ceiling"), s["match"]
    if fl is not None and ce is not None:
        lo, hi = min(fl, ce, m), max(fl, ce, m)
        span = max(hi - lo, 1e-6)
        pos = lambda x: 5 + 90 * (x - lo) / span
        scale = (f'<div class="scale"><div class="fill" style="left:{pos(fl)}%;width:{max(pos(ce) - pos(fl), 0)}%"></div>'
                 f'<div class="tick" style="left:{pos(fl)}%"></div><div class="lab" style="left:{pos(fl)}%">floor {fl:.2f}</div>'
                 f'<div class="tick" style="left:{pos(ce)}%"></div><div class="lab" style="left:{pos(ce)}%">ceiling {ce:.2f}</div>'
                 f'<div class="you" style="left:calc({pos(m)}% - 2px)"></div></div>'
                 f'<p class="small">Floor = rewrite from a deliberately WRONG prompt. Ceiling = two rewrites from the SAME '
                 f'prompt compared with each other: an AI never writes the same essay twice, so this is as high as any '
                 f'prompt can realistically score (even the true prompt averages about 0.57 in our tests).</p>')
    else:
        scale = ""
    p = r.get("position")
    if p is not None:
        verdict = ("very close to the best possible" if p >= .8 else "close" if p >= .5 else
                   "partial match" if p >= .2 else "no better than a wrong prompt")
        head = (f'<p class="vt big">{p:.0%}</p><p class="tag">of the way from a wrong prompt (0%) to the best an AI '
                f'can match itself (100%): <b>{verdict}</b>. Raw Match Score {m:.2f}.</p>')
    else:
        head = f'<p class="vt">{m:.2f}</p><p class="small">Raw Match Score (the wrong-prompt control failed, so no scale).</p>'
    card("How well the reconstructed prompt rebuilds the essay", f'{head}{scale}{bars}')
    cols = st.columns(2)
    with cols[0]:
        st.markdown("**Original**")
        st.text_area("orig", original, height=360, label_visibility="collapsed", disabled=True)
    with cols[1]:
        tabs = st.tabs([f"Regeneration {i + 1}" for i in range(len(r["regenerations"]))])
        for t, txt, sc in zip(tabs, r["regenerations"], r["scores"]):
            with t:
                st.caption(f"Match {sc['match']:.2f}")
                st.text_area("re", txt, height=330, label_visibility="collapsed", disabled=True, key=f"re{hash(txt)}")


# ------------------------------------------------------------------ pages
if page == "Detect & Reconstruct":
    if use_demo:
        key = st.selectbox("Pre-run example", list(demo))
        ex = demo[key]
        ss.text, ss.analysis, ss.demo_key = ex["text"], ex["analysis"], key
        ss.verify = ex.get("verify")
        st.text_area("Essay", ss.text, height=220, disabled=True)
        st.caption(f"Ground truth: {ex.get('truth', 'unknown')}")
    else:
        # bind to a fixed key: passing the stored text back as `value` makes Streamlit treat
        # every change as a NEW widget and silently drop what the user typed
        st.caption("Built for **student essays** (100+ words, English). Not validated for professional or "
                   "adult writing such as news or published articles: on those it raises false alarms "
                   "(see About the model). A verdict is evidence, never proof.")
        st.text_area("Essay", key="essay_input", height=240,
                     placeholder="Paste an essay (at least 100 words, English)...")
        ss.text = ss.get("essay_input", "")
        go = st.button("Analyze")
        if go:
            ss.verify, ss.demo_key = None, None
            try:
                with st.spinner("Loading models (first run can take a minute)..."):
                    models = load_models()
                llm = load_llm(gk, qk) if (gk or qk) else None
                with st.spinner("Reading the text (about 10-30 s on the free server)..."):
                    ss.analysis = analyze_cached(ss.text, bool(llm), gk, qk)
            except FileNotFoundError:
                ss.analysis = {"error": "Model files are missing. Switch on demo mode, or run the training scripts."}
            except Exception as e:  # never show a raw traceback in a demo
                ss.analysis = {"error": f"Something went wrong while analyzing ({type(e).__name__}). Try again or use demo mode."}
    a = ss.analysis
    if a:
        if "error" in a:
            st.warning(a["error"])
        else:
            c1, c2 = st.columns([1.05, 1])
            with c1:
                render_verdict(a)
                render_prompt(a)
            with c2:
                render_reasons(a)
            if a["verdict"] == "AI":
                st.info("Go to **Verify** to regenerate the essay from this prompt and score the match.")

elif page == "Verify":
    a = ss.analysis
    if not a or "error" in a:
        st.info("Analyze an essay first (page 1).")
    elif a["verdict"] != "AI":
        st.info("Verification applies to essays judged AI-generated.")
    else:
        card("Prompt being tested", f'<div class="prompt">{a["prompt"]}</div>')
        if use_demo and ss.verify:
            render_verify(ss.verify, ss.text)
        else:
            llm = load_llm(gk, qk) if (gk or qk) else None
            if not llm:
                st.warning("Add a Gemini or Groq API key in the sidebar to regenerate (or use demo mode).")
            elif st.button("Regenerate x3 and score"):
                import pipeline
                from llm import wrong_prompt_for
                with st.spinner("Asking the LLM for 3 rewrites + 1 wrong-prompt control, all at once (usually under a minute)..."):
                    try:
                        ss.verify = pipeline.verify(ss.text, a["prompt"], llm.chat, wrong_prompt_for(a["topic"]), n=3)
                    except Exception as e:
                        ss.verify = {"error": f"Regeneration failed ({type(e).__name__}). Rate limits are common on free "
                                              "keys: wait a minute and try again."}
            if ss.verify and "error" in ss.verify and llm:
                with st.expander("Details (what the AI services answered)"):
                    st.code("\n".join(llm.diagnostics()), language=None)
            if ss.verify:
                render_verify(ss.verify, ss.text)

else:
    p = ART / "model_card.json"
    mc = json.load(open(p)) if p.exists() else {}
    st.markdown("### How it works")
    st.markdown(
        "- **Guard**: gradient-boosted trees on writing-style features (sentence rhythm, vocabulary, punctuation, "
        "syntax) and frozen GPT-2 / DistilGPT-2 predictability features (perplexity, token rank, Binoculars-style ratio). "
        "Probabilities are isotonic-calibrated; inside an uncertainty band it abstains.\n"
        "- **Detective**: one classifier per prompt setting (tone, persona, evasion, length), trained on essays whose "
        "prompts are known.\n"
        "- **Verifier**: regenerates the essay from the reconstructed prompt and scores Words, Structure and Ideas "
        "between a wrong-prompt floor and a self-consistency ceiling.\n"
        "- All text is cleaned by the same `normalize()` used in training, so formatting cannot give the answer away.")
    if mc:
        st.markdown("### Evaluation (real numbers, see RESULTS.md)")
        st.json(mc)
    st.markdown('<p class="small">Limitations: short or heavily edited texts, mixed human/AI authorship, and '
                'generators unlike any in training are harder. A verdict is evidence, not proof.</p>',
                unsafe_allow_html=True)
