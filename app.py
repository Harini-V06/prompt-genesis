"""
Prompt Genesis: Streamlit app.
  Page 1  Detect & Reconstruct   Guard verdict (calibrated, can abstain) + plain-English reasons,
                                 Detective settings + reconstructed prompt
  Page 2  Verify                 regenerate x3 with the reconstructed prompt, Match Score on the
                                 floor -> ceiling scale
  Page 3  About the model        how it works + the evaluation numbers from model_card.json
Demo mode replays cached, pre-run examples (no models / API needed), the live-demo backup.
All inference goes through src/pipeline.py, the same code used for the reported results.

UI: light, glass-surface design (frosted top bar, white cards, pill buttons). All styling lives in
the CSS block below; the analysis logic is unchanged from the previous (dark) version, which is
kept as app_dark_backup.py.
"""
import html as _html
import json
import os
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))            # flat layout (all files in one folder) also works
sys.path.insert(0, str(ROOT / "src"))
ART = ROOT / "artifacts" if (ROOT / "artifacts").exists() else ROOT

st.set_page_config(page_title="Prompt Genesis", page_icon=":material/fingerprint:", layout="wide",
                   initial_sidebar_state="collapsed")

# ------------------------------------------------------------------ style
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Google+Sans+Flex:opsz,slnt,wdth,wght,ROND@8..144,-10..0,25..150,400..700,0..100&display=swap');
@import url('https://fonts.googleapis.com/css2?family=Google+Sans+Code:ital@0;1&display=swap');

:root {
  --palette-grey-1200: #121317; --palette-grey-1100: #18191d; --palette-grey-1000: #212226;
  --palette-grey-900: #2f3034;  --palette-grey-800: #45474d;  --palette-grey-300: #b2bbc5;
  --palette-grey-200: #cdd4dc;  --palette-grey-100: #e1e6ec;  --palette-grey-50: #e6eaf0;
  --palette-grey-20: #eff2f7;   --palette-grey-15: #f0f1f5;   --palette-grey-10: #f8f9fc;
  --palette-grey-0: #fff;       --palette-grey-400: #b7bfd9;  --palette-blue-600: #52b788;
  --surface: #fff; --on-surface: #121317; --on-surface-variant: #45474d;
  --container: #f8f9fc; --container-high: #eff2f7; --container-higher: #e6eaf0;
  --outline: #2122261F; --outline-variant: #2122260F;
  --accent-wash: #b7bfd917; --accent-wash-2: #b7bfd933;
  --sh-1: #0000000A 0px 3px 10px 0px;
  --sh-card: #00000024 0px 14px 36px -4px, #0000000F 0px 4px 12px -2px;
  --sh-pop: #0000001F 0px 16px 40px 0px;
  --ai: #c5221f; --ai-wash: #fce8e6; --hu: #137333; --hu-wash: #e6f4ea;
  --font: 'Google Sans Flex', 'Google Sans Text', 'Google Sans', system-ui, -apple-system, 'Segoe UI', Roboto, sans-serif;
  --mono: 'Google Sans Code', ui-monospace, 'SFMono-Regular', Menlo, Consolas, monospace;
  --ease: cubic-bezier(.2, 0, 0, 1);
}

/* ---------- base ---------- */
html, body, .stApp { background: var(--surface) !important; color: var(--on-surface); }
.stApp, .stApp *:not(code):not(pre):not(kbd) { font-family: var(--font); }
.stApp p, .stApp li { font-size: 16px; line-height: 24px; }
.stApp [data-testid="stIconMaterial"], .stApp .pg-note .ic { font-family: 'Material Symbols Rounded' !important; }
.stApp .pg-prompt, .stApp .pg-code, .stApp .pg-tile .n, .stApp .pg-scale .lab, .stApp .pg-bar .t span:last-child { font-family: var(--mono) !important; }
header[data-testid="stHeader"], #MainMenu, footer, [data-testid="stDecoration"], [data-testid="stSidebar"],
[data-testid="stSidebarCollapsedControl"], [data-testid="stStatusWidget"] { display: none !important; }
[data-testid="stMainBlockContainer"], .block-container { max-width: 1200px; padding: 16px 24px 72px 24px; }
.stApp a { color: #202124; text-decoration: none; }
::selection { background: #52b78833; }

/* soft ambient backdrop */
.stApp::before { content: ""; position: fixed; inset: -20% -10% auto -10%; height: 70vh; z-index: 0; pointer-events: none;
  background: radial-gradient(60% 60% at 20% 20%, #b7bfd933 0%, transparent 60%),
              radial-gradient(50% 50% at 85% 10%, #52b78814 0%, transparent 60%); }

@keyframes agy-fade-in-settle { from { opacity: 0; transform: translateY(8px); } to { opacity: 1; transform: none; } }
@keyframes agy-fade-in { from { opacity: 0; } to { opacity: 1; } }

/* ---------- top bar (frosted glass) ---------- */
[data-testid="stLayoutWrapper"]:has(> .st-key-topbar), .stElementContainer:has(> .st-key-topbar) {
  position: sticky; top: 12px; z-index: 999; }
.st-key-topbar { position: relative; margin: 0 0 8px 0;
  background: #FFFFFFE6; backdrop-filter: blur(24px) saturate(1.6); -webkit-backdrop-filter: blur(20px) saturate(1.6);
  border: 1px solid var(--outline-variant); border-radius: 9999px; padding: 8px 10px 8px 22px;
  box-shadow: var(--sh-1); }
.st-key-topbar [data-testid="stHorizontalBlock"] { gap: 8px; }
.st-key-tb_right { align-items: flex-end; }
.pg-brand { display: flex; align-items: center; align-self: center; gap: 10px; margin: 0; white-space: nowrap; }
.pg-word { font-size: 17.5px; font-weight: 500; letter-spacing: -0.01em; color: var(--on-surface); }
.pg-chip { display: inline-flex; align-items: center; gap: 6px; font-size: 13px; font-weight: 500; color: var(--on-surface-variant);
  background: var(--container-high); border-radius: 9999px; padding: 2px 10px; }
.pg-chip .d { width: 6px; height: 6px; border-radius: 50%; background: var(--palette-blue-600); }

/* segmented page nav */
.st-key-topbar [data-testid="stButtonGroup"] { display: flex; justify-content: center; width: 100%; }
.st-key-topbar [data-testid="stButtonGroup"] [role="radiogroup"] { background: var(--palette-grey-15); border-radius: 9999px;
  padding: 4px; gap: 2px; flex-wrap: nowrap; width: fit-content; margin: 0 auto; }
.st-key-topbar button[data-variant="segmented_control"] { border: none !important; border-radius: 9999px !important; margin: 0 !important;
  background: transparent !important; color: var(--on-surface-variant) !important; min-height: 34px; padding: 6px 18px !important;
  box-shadow: none !important; transition: background .2s var(--ease), color .2s var(--ease), box-shadow .2s var(--ease); }
.st-key-topbar button[data-variant="segmented_control"]:hover { color: var(--on-surface) !important; background: #ffffffaa !important; }
.st-key-topbar button[data-variant="segmented_control"][aria-checked="true"] { background: #fff !important; color: var(--on-surface) !important;
  box-shadow: #00000014 0 1px 2px, var(--sh-1) !important; }
.st-key-topbar button[data-variant="segmented_control"] p { font-size: 14.5px; font-weight: 450; letter-spacing: .11px; color: inherit; }

/* ---------- buttons ---------- */
.stButton > button, [data-testid="stPopover"] button, [data-testid="stBaseButton-primary"], [data-testid="stBaseButton-secondary"] {
  border-radius: 9999px !important; font-size: 14.5px; font-weight: 450; letter-spacing: .11px; min-height: 40px;
  padding: 6px 18px !important; transition: background .2s var(--ease), border-color .2s var(--ease), box-shadow .2s var(--ease), transform .15s var(--ease); }
[data-testid="stBaseButton-primary"] { background: #121317 !important; color: #fff !important; border: 1px solid transparent !important; }
[data-testid="stBaseButton-primary"]:hover { background: #2f3034 !important; box-shadow: #00000026 0px 4px 12px 0px; }
[data-testid="stBaseButton-primary"]:active { transform: scale(.98); }
[data-testid="stBaseButton-primary"] p, [data-testid="stBaseButton-primary"] span { color: #fff !important; }
[data-testid="stBaseButton-secondary"], [data-testid="stPopover"] button { background: #fff !important; color: var(--on-surface) !important;
  border: 1px solid var(--outline) !important; }
[data-testid="stBaseButton-secondary"]:hover, [data-testid="stPopover"] button:hover { background: var(--container) !important;
  border-color: #21222633 !important; }
button:focus-visible { outline: 2px solid var(--palette-blue-600) !important; outline-offset: 2px; }
[data-testid="stPopoverBody"] { border-radius: 16px !important; border: 1px solid var(--outline) !important; box-shadow: var(--sh-pop) !important;
  background: #FFFFFFF2 !important; backdrop-filter: blur(20px); padding: 16px !important; min-width: 340px; }

/* ---------- inputs ---------- */
[data-baseweb="textarea"], [data-baseweb="input"], [data-baseweb="select"] > div {
  background: var(--container) !important; border: 1px solid var(--outline) !important; border-radius: 14px !important;
  transition: border-color .2s var(--ease), box-shadow .2s var(--ease), background .2s var(--ease); }
[data-baseweb="textarea"]:focus-within, [data-baseweb="input"]:focus-within, [data-baseweb="select"] > div:focus-within {
  border-color: var(--palette-blue-600) !important; box-shadow: 0 0 0 3px #52b78826 !important; background: #fff !important; }
[data-baseweb="textarea"] textarea, [data-baseweb="input"] input { background: transparent !important; color: #121317 !important;
  font-size: 16px; line-height: 24px; caret-color: var(--palette-blue-600); }
[data-baseweb="textarea"] textarea { padding: 14px 16px !important; }
textarea::placeholder, input::placeholder { color: #45474d99 !important; }
textarea:disabled { color: var(--on-surface-variant) !important; -webkit-text-fill-color: var(--on-surface-variant) !important; opacity: 1 !important;
  cursor: default !important; }
.stApp label p { font-size: 14px !important; font-weight: 500; color: var(--on-surface-variant) !important; }
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p { color: var(--on-surface-variant) !important; font-size: 13px !important; }

/* theme-proof: force the light look even if .streamlit/config.toml is dark or missing (Streamlit 1.66 test ids) */
.stApp, [data-testid="stPopoverBody"] { color-scheme: light; }
.stApp label p, [data-testid="stWidgetLabel"] p { color: #45474d !important; }
[data-testid="stPopoverBody"] p { color: #121317; }
[data-testid="stPopoverBody"] [data-testid="stCaptionContainer"] p { color: #45474d !important; }
[data-testid="stTextAreaRootElement"], [data-testid="stTextInputRootElement"], [data-testid="stSelectbox"] [role="group"] {
  background: #f8f9fc !important; border: 1px solid #2122261F !important; border-radius: 14px !important; box-shadow: none !important;
  transition: border-color .2s var(--ease), box-shadow .2s var(--ease), background .2s var(--ease); }
[data-testid="stTextAreaRootElement"]:focus-within, [data-testid="stTextInputRootElement"]:focus-within,
[data-testid="stSelectbox"] [role="group"]:focus-within {
  border-color: #52b788 !important; box-shadow: 0 0 0 3px #52b78826 !important; background: #fff !important; }
[data-testid="stTextAreaRootElement"] textarea, [data-testid="stTextInputField"], [data-testid="stSelectbox"] [role="group"] input,
[data-testid="stSelectbox"] [role="group"] div, [data-testid="stSelectbox"] [role="group"] span {
  background: transparent !important; color: #121317 !important; -webkit-text-fill-color: #121317; caret-color: #52b788; }
[data-testid="stTextAreaRootElement"] textarea::placeholder, [data-testid="stTextInputField"]::placeholder {
  color: #45474d99 !important; -webkit-text-fill-color: #45474d99 !important; }
[data-testid="stTextAreaRootElement"] textarea { padding: 14px 16px !important; font-size: 16px; line-height: 24px; }
[data-testid="stTextAreaRootElement"] textarea:disabled { color: #45474d !important; -webkit-text-fill-color: #45474d !important; opacity: 1 !important; }
[data-testid="stTextInputRootElement"] button, [data-testid="stTextInputRootElement"] svg,
[data-testid="stSelectbox"] svg { background: transparent !important; color: #45474d !important; fill: #45474d !important; }
[data-testid="stSelectboxVirtualDropdown"] { background: #fff !important; border: 1px solid #2122261F; border-radius: 12px; }
[data-testid="stSelectboxVirtualDropdown"] [role="option"] { color: #121317 !important; background: transparent !important; }
[data-testid="stSelectboxVirtualDropdown"] [role="option"]:hover, [data-testid="stSelectboxVirtualDropdown"] [role="option"][aria-selected="true"] {
  background: #f0f1f5 !important; }
[data-testid="stCheckbox"] label > div:first-of-type, .stCheckbox label > div:first-of-type { background: #cdd4dc !important; }
[data-testid="stCheckbox"] label:has(input:checked) > div:first-of-type, .stCheckbox label:has(input:checked) > div:first-of-type {
  background: #121317 !important; }
[data-testid="stCheckbox"] label > div:first-of-type > div, .stCheckbox label > div:first-of-type > div { background: #fff !important; }
[data-testid="stTooltipIcon"] svg { color: #45474d !important; }
[data-testid="stExpander"] summary, [data-testid="stExpander"] summary p { color: #121317 !important; }
[data-testid="stSpinner"], [data-testid="stSpinner"] p { color: #45474d !important; }
.stApp [data-testid="stCode"] pre, .stApp [data-testid="stCode"] code { background: #f8f9fc !important; color: #121317 !important; }

/* tabs */
[data-baseweb="tab-list"] { gap: 4px; }
[data-baseweb="tab"] { font-size: 14.5px; color: var(--on-surface-variant); padding: 8px 4px !important; }
[data-baseweb="tab"][aria-selected="true"] { color: var(--on-surface); }
[data-baseweb="tab-highlight"] { background: #121317 !important; height: 2px !important; }
[data-baseweb="tab-border"] { background: var(--outline) !important; }

/* expander / alerts */
[data-testid="stExpander"] details { border: 1px solid var(--outline) !important; border-radius: 14px !important; background: #fff; }
[data-testid="stAlert"] { border-radius: 14px !important; }

/* ---------- surfaces ---------- */
.pg-card, .st-key-composer, .st-key-verify_panel {
  background: #FFFFFFFA; border: 1px solid var(--outline); border-radius: 16px; box-shadow: var(--sh-1);
  animation: agy-fade-in-settle .45s var(--ease) both; }
.pg-card { padding: 22px 24px; margin-bottom: 16px; color: var(--on-surface-variant); }
.st-key-composer, .st-key-verify_panel { padding: 20px 22px 18px 22px; }
.pg-card-h { display: flex; align-items: center; justify-content: space-between; gap: 8px; margin: 0 0 14px 0;
  font-size: 13px; font-weight: 500; letter-spacing: .08em; text-transform: uppercase; color: var(--on-surface-variant); }
.pg-card p { margin: 0; }

/* ---------- typography blocks ---------- */
.pg-hero { padding: 56px 0 28px 0; animation: agy-fade-in-settle .6s var(--ease) both; position: relative; }
.pg-eyebrow { display: inline-flex; align-items: center; gap: 8px; font-size: 14px; font-weight: 500; color: var(--on-surface-variant);
  background: #FFFFFFB8; border: 1px solid var(--outline); border-radius: 9999px; padding: 4px 12px 4px 8px; backdrop-filter: blur(12px); }
.pg-eyebrow .d { width: 18px; height: 18px; border-radius: 50%; background: #121317; display: inline-grid; place-items: center; }
.pg-eyebrow .d::after { content: ""; width: 6px; height: 6px; border-radius: 50%; background: #fff; }
.pg-h1 { font-size: clamp(44px, 6.2vw, 80px); line-height: 1.04; font-weight: 450; letter-spacing: -0.035em; color: #121317;
  margin: 22px 0 18px 0; }
.pg-h1 .soft { color: #121317; background: linear-gradient(90deg, #121317 10%, #45474d 55%, #52b788 100%);
  -webkit-background-clip: text; background-clip: text; -webkit-text-fill-color: transparent; }
.pg-h2 { font-size: clamp(32px, 4vw, 42px); line-height: 1.12; font-weight: 450; letter-spacing: -0.025em; color: #121317; margin: 14px 0 10px 0; }
.stApp .pg-lead { font-size: 17.5px !important; line-height: 28px !important; color: var(--on-surface-variant); max-width: 640px; margin: 0; }
.pg-page-head { padding: 40px 0 20px 0; animation: agy-fade-in-settle .5s var(--ease) both; }
.pg-section-label { font-size: 13px; font-weight: 500; letter-spacing: .08em; text-transform: uppercase; color: var(--on-surface-variant);
  margin: 40px 0 14px 0; }
.stApp .pg-muted { color: var(--on-surface-variant); font-size: 14px; line-height: 21px; }

/* feature tiles */
.pg-tiles { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 16px; margin-top: 8px; }
.pg-tiles.four { grid-template-columns: repeat(4, minmax(0, 1fr)); }
.pg-tile { background: var(--container); border: 1px solid var(--outline-variant); border-radius: 16px; padding: 22px;
  transition: background .25s var(--ease), box-shadow .25s var(--ease), transform .25s var(--ease); }
.pg-tile:hover { background: #fff; box-shadow: var(--sh-card); transform: translateY(-2px); }
.pg-tile .n { font-family: var(--mono); font-size: 13px; color: var(--on-surface-variant); }
.pg-tile .t { font-size: 20px; font-weight: 500; color: #121317; margin: 28px 0 6px 0; letter-spacing: -0.01em; }
.pg-tile .s { font-size: 14.5px; line-height: 22px; color: var(--on-surface-variant); }
@media (max-width: 900px) { .pg-tiles, .pg-tiles.four { grid-template-columns: 1fr; } }

/* ---------- verdict ---------- */
.pg-status { display: inline-flex; align-items: center; gap: 8px; font-size: 13px; font-weight: 500; letter-spacing: 0;
  text-transform: none; border-radius: 9999px; padding: 3px 10px; }
.pg-status .d { width: 7px; height: 7px; border-radius: 50%; background: currentColor; }
.pg-status.ai { background: var(--ai-wash); color: var(--ai); }
.pg-status.human { background: var(--hu-wash); color: var(--hu); }
.pg-status.unsure { background: var(--container-high); color: var(--on-surface-variant); }
.stApp .pg-vt { font-size: 34px; line-height: 40px; font-weight: 450; letter-spacing: -0.02em; color: #121317 !important; margin: 0 0 6px 0 !important; }
.stApp .pg-vs { color: var(--on-surface-variant); font-size: 15px; }
.pg-gauge { position: relative; height: 10px; border-radius: 9999px; margin: 30px 0 10px 0;
  background: linear-gradient(90deg, #e6eaf0 0%, #b7bfd9 50%, #121317 100%); }
.pg-gauge .band { position: absolute; top: -4px; height: 18px; border-radius: 6px; border: 1px dashed #45474d66;
  background: repeating-linear-gradient(135deg, #ffffff88 0 4px, transparent 4px 8px); }
.pg-gauge .mk { position: absolute; top: 50%; width: 20px; height: 20px; border-radius: 50%; transform: translate(-50%, -50%);
  background: #fff; border: 5px solid var(--palette-blue-600); box-shadow: #52b7884d 0 2px 8px; }
.pg-gl { display: flex; justify-content: space-between; color: var(--on-surface-variant); font-size: 13px; }

/* reasons */
.pg-reason { display: grid; grid-template-columns: auto 1fr; gap: 12px; align-items: start; padding: 12px 0;
  border-top: 1px solid var(--outline-variant); }
.pg-reason:first-of-type { border-top: none; padding-top: 0; }
.pg-tag { font-size: 12px; font-weight: 500; border-radius: 9999px; padding: 2px 10px; white-space: nowrap; margin-top: 1px; }
.pg-tag.ai { background: #121317; color: #fff; }
.pg-tag.hu { background: var(--container-high); color: var(--on-surface-variant); border: 1px solid var(--outline); }
.pg-reason .s { color: #121317; font-size: 15px; line-height: 22px; }
.pg-reason .l { color: var(--on-surface-variant); font-size: 13px; margin-top: 2px; }
.stApp .pg-foot { color: var(--on-surface-variant); font-size: 13px; line-height: 19px; margin-top: 14px !important; padding-top: 12px;
  border-top: 1px solid var(--outline-variant); }

/* prompt */
.pg-prompt { font-family: var(--mono); font-size: 14.5px; line-height: 23px; color: #121317; background: var(--container);
  border: 1px solid var(--outline-variant); border-radius: 12px; padding: 16px 18px; white-space: pre-wrap; word-break: break-word; }
.stApp .pg-kvline { font-size: 14px; color: var(--on-surface-variant); margin-top: 14px !important; }
.pg-kvline b { color: #121317; font-weight: 500; }
.pg-checks { list-style: none; padding: 0; margin: 8px 0 0 0; }
.stApp .pg-checks li { position: relative; padding: 4px 0 4px 26px; font-size: 14.5px !important; line-height: 21px !important; color: #121317; }
.pg-checks li::before { content: ""; position: absolute; left: 2px; top: 8px; width: 14px; height: 14px; border-radius: 50%;
  background: var(--container-high); border: 1px solid var(--outline); }
.pg-checks li::after { content: ""; position: absolute; left: 7px; top: 11px; width: 3px; height: 6px; border: solid #121317;
  border-width: 0 1.5px 1.5px 0; transform: rotate(45deg); }
.pg-badges { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 8px; }
.pg-badge { font-size: 13px; color: var(--on-surface-variant); border: 1px dashed var(--palette-grey-300); border-radius: 9999px; padding: 3px 12px; }
.pg-badge b { color: #121317; font-weight: 500; margin-right: 4px; }

/* notes */
.stApp .pg-note { display: grid; grid-template-columns: 22px 1fr; gap: 12px; align-items: start; border-radius: 14px; padding: 14px 16px;
  margin: 0 0 14px 0; font-size: 14.5px; line-height: 22px; border: 1px solid transparent; animation: agy-fade-in .4s var(--ease) both; }
.pg-note .ic { font-family: 'Material Symbols Rounded'; font-size: 20px; line-height: 22px; font-weight: 400;
  font-variation-settings: 'FILL' 1; -webkit-font-feature-settings: 'liga'; }
.pg-note.info { background: #edf7f1; color: #1d5e3c; border-color: #52b7881f; }
.pg-note.warn { background: #fef7e0; color: #7a4a00; border-color: #f9ab0026; }
.pg-note.danger { background: var(--ai-wash); color: #8c1d18; border-color: #c5221f1f; }
.pg-note.plain { background: var(--container); color: var(--on-surface-variant); border-color: var(--outline-variant); }
.pg-note b { font-weight: 600; }

/* ---------- verify ---------- */
.stApp .pg-big { font-size: clamp(56px, 7vw, 72px); line-height: 1; font-weight: 450; letter-spacing: -0.04em; color: #121317 !important; margin: 4px 0 10px 0 !important; }
.pg-scale { position: relative; height: 10px; background: var(--container-high); border-radius: 9999px; margin: 44px 0 12px 0; }
.pg-scale .fill { position: absolute; height: 100%; border-radius: 9999px; background: linear-gradient(90deg, #cdd4dc, #121317); }
.pg-scale .lab { position: absolute; top: -26px; font-size: 12px; color: var(--on-surface-variant); transform: translateX(-50%); white-space: nowrap;
  font-family: var(--mono); }
.pg-scale .tick { position: absolute; top: -4px; width: 2px; height: 18px; background: #45474d; border-radius: 2px; transform: translateX(-50%); }
.pg-scale .you { position: absolute; top: 50%; width: 20px; height: 20px; border-radius: 50%; transform: translate(-50%, -50%);
  background: #fff; border: 5px solid var(--palette-blue-600); box-shadow: #52b7884d 0 2px 8px; }
.pg-bar { margin: 14px 0; }
.pg-bar .t { display: flex; justify-content: space-between; font-size: 14.5px; color: #121317; }
.pg-bar .t span:last-child { font-family: var(--mono); font-size: 13px; color: var(--on-surface-variant); }
.pg-bar .tr { height: 6px; background: var(--container-high); border-radius: 9999px; overflow: hidden; margin-top: 8px; }
.pg-bar .tr div { height: 100%; background: #121317; border-radius: 9999px; transition: width .6s var(--ease); }
.pg-colhead { font-size: 13px; font-weight: 500; letter-spacing: .08em; text-transform: uppercase; color: var(--on-surface-variant); margin: 8px 0 8px 0; }

/* ---------- about ---------- */
.pg-kv { display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 1.25fr); gap: 24px; padding: 16px 0; border-top: 1px solid var(--outline-variant); }
.pg-kv:first-of-type { border-top: none; padding-top: 0; }
.pg-kv .k { font-size: 15px; color: #121317; font-weight: 450; }
.pg-kv .v { font-size: 14.5px; color: var(--on-surface-variant); line-height: 22px; }
.pg-kv .sub { display: flex; justify-content: space-between; gap: 16px; padding: 2px 0; }
.pg-kv .sub span:first-child { color: #121317; text-transform: capitalize; }
@media (max-width: 720px) { .pg-kv { grid-template-columns: 1fr; gap: 6px; } }
.pg-code { font-family: var(--mono); font-size: 13px; background: var(--container-high); border-radius: 6px; padding: 1px 6px; color: #121317; }

/* footer */
.pg-footer { display: flex; justify-content: space-between; flex-wrap: wrap; gap: 12px; margin-top: 72px; padding-top: 24px;
  border-top: 1px solid var(--outline-variant); color: var(--on-surface-variant); font-size: 13px; }
.pg-footer .pg-word { font-size: 14.5px; }

@media (max-width: 640px) {
  [data-testid="stMainBlockContainer"], .block-container { padding: 12px 16px 48px 16px; }
  .st-key-topbar { border-radius: 22px; padding: 10px 12px; top: 8px; }
  .st-key-tb_right { align-items: stretch; }
  .pg-hero { padding: 32px 0 20px 0; }
  .pg-card { padding: 18px; }
}
</style>
""", unsafe_allow_html=True)

E = _html.escape


def md(s):
    st.markdown(s, unsafe_allow_html=True)


def card(title, body_html, extra_cls="", right=""):
    md(f'<div class="pg-card {extra_cls}"><div class="pg-card-h"><span>{title}</span>{right}</div>{body_html}</div>')


NOTE_ICONS = {"info": "info", "warn": "warning", "danger": "error", "plain": "arrow_forward"}


def note(body_html, kind="info"):
    md(f'<div class="pg-note {kind}"><span class="ic">{NOTE_ICONS[kind]}</span><div>{body_html}</div></div>')


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
    """None when no usable model; _status=True returns the object anyway so the settings panel can say why."""
    llm = build_llm(gemini_key, groq_key)
    return llm if (llm.available or _status) else None


@st.cache_data(show_spinner=False, max_entries=200)
def analyze_cached(text, use_llm_topic, _gk, _qk, force=False):     # keys start with "_" so they are never hashed/stored
    """Same essay (and same key setup) -> instant second time; results are deterministic except the LLM topic."""
    import pipeline
    llm = load_llm(_gk, _qk) if use_llm_topic else None
    return pipeline.analyze(text, load_models(), topic_fn=llm.topic if llm else None, force=force)


@st.cache_data(show_spinner=False)
def load_demo():
    p = ART / "demo_cache.json"
    return json.load(open(p, encoding="utf-8")) if p.exists() else {}


# ------------------------------------------------------------------ state
ss = st.session_state
ss.setdefault("analysis", None)
ss.setdefault("text", "")
ss.setdefault("verify", None)
ss.setdefault("demo_key", None)
ss.setdefault("nav", "Detect & Reconstruct")
ss.setdefault("last_page", "Detect & Reconstruct")
if "essay_input" in ss:                 # keep the draft essay when the Detect page is not on screen
    ss.essay_input = ss.essay_input

PAGES = {"Detect & Reconstruct": "Detect", "Verify": "Verify", "About the model": "About"}


def _keep_nav():
    """A segmented control can be clicked 'off'; never leave the app without a page."""
    if ss.nav is None:
        ss.nav = ss.last_page
    else:
        ss.last_page = ss.nav


def go_to(page_name):
    ss.nav = ss.last_page = page_name


# ------------------------------------------------------------------ top bar
demo = load_demo()
with st.container(key="topbar"):
    c_brand, c_nav, c_right = st.columns([1.15, 2, 1.15], vertical_alignment="center")
    with c_right:
        with st.container(key="tb_right"):
            with st.popover("Settings", icon=":material/tune:"):
                md('<div class="pg-card-h" style="margin-bottom:6px"><span>Session settings</span></div>')
                use_demo = st.toggle("Demo mode (pre-run examples)", value=False, key="use_demo",
                                     help="Replays cached results: works without models or API keys.") if demo else False
                gk = (st.text_input("Gemini API key", type="password", value="", key="gk_in") or secret("GEMINI_API_KEY") or "").strip()
                qk = (st.text_input("Groq API key", type="password", value="", key="qk_in") or secret("GROQ_API_KEY") or "").strip()
                st.caption("Keys stay in this session only. Without a key, the topic comes from keywords and Verify is disabled.")
                if gk or qk:
                    _l = load_llm(gk, qk, _status=True)
                    for k, v in (_l.providers.items() if _l else []):
                        st.caption(f"{'Gemini' if 'GEMINI' in k else 'Groq'}: {v}")
    with c_brand:
        chip = '<span class="pg-chip"><span class="d"></span>Demo</span>' if use_demo else ""
        md(f'<div class="pg-brand"><span class="pg-word">Prompt Genesis</span>{chip}</div>')
    with c_nav:
        st.segmented_control("Page", list(PAGES), format_func=PAGES.get, key="nav", on_change=_keep_nav,
                             label_visibility="collapsed")

page = ss.nav or ss.last_page


# ------------------------------------------------------------------ renderers
def pct(x):
    """Calibrated probabilities can hit exactly 0/1; no detector is certain, so cap the wording."""
    return "&gt; 99%" if x > 0.99 else ("&lt; 1%" if x < 0.01 else f"{x:.0%}")


def page_head(eyebrow, title, lead):
    md(f'<div class="pg-page-head"><div class="pg-eyebrow"><span class="d"></span>{eyebrow}</div>'
       f'<div class="pg-h2" role="heading" aria-level="1">{title}</div><p class="pg-lead">{lead}</p></div>')


def render_verdict(a):
    p, band = a["p_ai"], a["band"]
    v = a["verdict"]
    if v == "out of scope":
        card("Verdict",
             '<p class="pg-vt">Out of scope: no verdict</p>'
             f'<p class="pg-vs">This does not read like a student essay (scope check {a["scope"]["p_out"]:.0%} '
             f'&gt; {a["scope"]["threshold"]:.0%}) &middot; {a["n_words"]} words</p>'
             '<p class="pg-foot">Prompt Genesis was trained on school and university student essays. On other adult '
             'writing (work reports, news, reviews, encyclopedia-style text) it wrongly flagged about 10% of genuinely '
             'human texts as AI in our tests, so it gives no verdict here rather than risk a false accusation. The scope '
             'check also refuses about 1% of school essays and 8% of university essays (see About the model).</p>',
             right='<span class="pg-status unsure"><span class="d"></span>Abstained</span>')
        return
    if v == "AI":
        cls, status, title, sub = "ai", "AI", "Likely AI-generated", f"{pct(p)} probability of AI (calibrated)"
    elif v == "human":
        cls, status, title, sub = "human", "Human", "Likely human-written", f"{pct(1 - p)} probability of human (calibrated)"
    else:
        cls, status, title = "unsure", "Abstained", "Insufficient evidence"
        sub = (f"P(AI) = {p:.0%}, inside the uncertainty band [{0.5 - band:.0%}, {0.5 + band:.0%}]. "
               "The model abstains rather than guess.")
    lo, hi = (0.5 - band) * 100, (0.5 + band) * 100
    g = (f'<div class="pg-gauge"><div class="band" style="left:{lo}%;width:{max(hi - lo, 0.6)}%"></div>'
         f'<div class="mk" style="left:{min(max(p * 100, 1.5), 98.5)}%"></div></div>'
         '<div class="pg-gl"><span>Human</span><span>Uncertain</span><span>AI</span></div>')
    card("Verdict", f'<p class="pg-vt">{title}</p><p class="pg-vs">{sub} &middot; {a["n_words"]} words</p>{g}',
         right=f'<span class="pg-status {cls}"><span class="d"></span>{status}</span>')


def render_reasons(a):
    rows = "".join(
        f'<div class="pg-reason"><span class="pg-tag {"ai" if r["toward"] == "AI" else "hu"}">'
        f'{"Points to AI" if r["toward"] == "AI" else "Points to human"}</span>'
        f'<div><div class="s">{E(r["sentence"])}</div><div class="l">{E(r["label"])}</div></div></div>'
        for r in a["reasons"])
    card("Why: strongest evidence", rows + '<p class="pg-foot">Each line is one feature\'s contribution to this verdict '
         '(SHAP), not a rule. Several weak signals add up.</p>', right='<span class="pg-chip">SHAP</span>')


def render_prompt(a):
    if "settings" not in a:
        return
    s = a["settings"]
    src = "detected by the LLM" if a.get("topic_source") == "llm" else "main noun phrases (add an API key for a better topic)"
    measured = "".join(f"<li>{E(c)}</li>" for c in a.get("style_clauses", []))
    guesses = "".join(
        f'<span class="pg-badge"><b>{E(k)}</b>{E(str(d["value"]))} &middot; {d["prob"]:.0%}</span>'
        for k, d in s.items() if k in ("tone", "persona"))
    card("Reconstructed prompt",
         f'<div class="pg-prompt">{E(a["prompt"])}</div>'
         f'<p class="pg-kvline"><b>Topic</b> &middot; {src}: {E(a["topic"])}</p>'
         f'<p class="pg-kvline"><b>Measured from the text</b> &middot; counted, not guessed</p><ul class="pg-checks">{measured}</ul>'
         f'<p class="pg-kvline"><b>Style guesses</b> &middot; classifier, not used in the prompt</p><div class="pg-badges">{guesses}</div>'
         f'<p class="pg-foot">The guesses are about 85% accurate on essays made through AI APIs but mostly wrong on essays '
         f'copied from chatbot apps such as ChatGPT, which add their own hidden instructions. So the prompt only uses '
         f'what can be measured; the guesses are shown for context.</p>')


def render_verify(r, original):
    if "error" in r:
        note(E(r["error"]), "danger")
        return
    s = r["summary"]
    bars = "".join(
        f'<div class="pg-bar"><div class="t"><span>{lab}</span><span>{s[k]:.0%}</span></div>'
        f'<div class="tr"><div style="width:{s[k] * 100:.1f}%"></div></div></div>'
        for k, lab in (("words", "Words &middot; TF-IDF + ROUGE-L"), ("structure", "Structure &middot; style profile"),
                       ("ideas", "Ideas &middot; sentence meaning, MiniLM")) if k in s)
    fl, ce, m = r.get("floor"), r.get("ceiling"), s["match"]
    if fl is not None and ce is not None:
        lo, hi = min(fl, ce, m), max(fl, ce, m)
        span = max(hi - lo, 1e-6)
        pos = lambda x: 5 + 90 * (x - lo) / span
        scale = (f'<div class="pg-scale"><div class="fill" style="left:{pos(fl)}%;width:{max(pos(ce) - pos(fl), 0)}%"></div>'
                 f'<div class="tick" style="left:{pos(fl)}%"></div><div class="lab" style="left:{pos(fl)}%">floor {fl:.2f}</div>'
                 f'<div class="tick" style="left:{pos(ce)}%"></div><div class="lab" style="left:{pos(ce)}%">ceiling {ce:.2f}</div>'
                 f'<div class="you" style="left:{pos(m)}%"></div></div>'
                 f'<p class="pg-foot">Floor = rewrite from a deliberately WRONG prompt. Ceiling = two rewrites from the SAME '
                 f'prompt compared with each other: an AI never writes the same essay twice, so this is as high as any '
                 f'prompt can realistically score (even the true prompt averages about 0.57 in our tests).</p>')
    else:
        scale = ""
    p = r.get("position")
    if p is not None:
        verdict = ("very close to the best possible" if p >= .8 else "close" if p >= .5 else
                   "partial match" if p >= .2 else "no better than a wrong prompt")
        head = (f'<p class="pg-big">{p:.0%}</p><p class="pg-vs">of the way from a wrong prompt (0%) to the best an AI '
                f'can match itself (100%): <b style="color:#121317">{verdict}</b>. Raw Match Score {m:.2f}.</p>')
    else:
        head = (f'<p class="pg-big">{m:.2f}</p><p class="pg-vs">Raw Match Score (the wrong-prompt control failed, '
                f'so no scale).</p>')
    c1, c2 = st.columns([1.15, 1], gap="large")
    with c1:
        card("How well the reconstructed prompt rebuilds the essay", f'{head}{scale}')
    with c2:
        card("Score breakdown", bars or '<p class="pg-muted">No component scores.</p>')
    md('<div class="pg-section-label">Side by side</div>')
    cols = st.columns(2, gap="large")
    with cols[0]:
        md('<div class="pg-colhead">Original</div>')
        st.text_area("orig", original, height=380, label_visibility="collapsed", disabled=True)
    with cols[1]:
        tabs = st.tabs([f"Regeneration {i + 1}" for i in range(len(r["regenerations"]))])
        for t, txt, sc in zip(tabs, r["regenerations"], r["scores"]):
            with t:
                st.caption(f"Match {sc['match']:.2f}")
                st.text_area("re", txt, height=330, label_visibility="collapsed", disabled=True, key=f"re{hash(txt)}")


TILES = (('01', 'Guard', 'A calibrated person-vs-AI verdict from writing style and language-model predictability. '
          'It abstains when the evidence is weak.'),
         ('02', 'Detective', 'Rebuilds the most likely prompt from what can actually be measured in the text: '
          'topic, length, sentence rhythm, reading level.'),
         ('03', 'Verifier', 'Regenerates the essay from that prompt and scores how close it lands, between a '
          'wrong-prompt floor and a self-consistency ceiling.'))


def tiles(items, extra=""):
    md(f'<div class="pg-tiles {extra}">' + "".join(
        f'<div class="pg-tile"><div class="n">{n}</div><div class="t">{t}</div><div class="s">{s}</div></div>'
        for n, t, s in items) + '</div>')


# ------------------------------------------------------------------ pages
if page == "Detect & Reconstruct":
    md('<section class="pg-hero"><div class="pg-eyebrow"><span class="d"></span>AI essay forensics &middot; '
       'detect, reconstruct, verify</div>'
       '<div class="pg-h1" role="heading" aria-level="1">Find the prompt<br><span class="soft">behind the essay.</span></div>'
       '<p class="pg-lead">Was this essay written by a person or an AI? If AI, what prompt most likely produced it, '
       'and can we rebuild it?</p></section>')

    with st.container(key="composer"):
        if use_demo:
            key = st.selectbox("Pre-run example", list(demo))
            ex = demo[key]
            ss.text, ss.analysis, ss.demo_key = ex["text"], ex["analysis"], key
            ss.verify = ex.get("verify")
            st.text_area("Essay", ss.text, height=220, disabled=True)
            st.caption(f"Ground truth: {ex.get('truth', 'unknown')}  ·  {len(ss.text.split())} words")
        else:
            # bind to a fixed key: passing the stored text back as `value` makes Streamlit treat
            # every change as a NEW widget and silently drop what the user typed
            st.text_area("Essay", key="essay_input", height=240,
                         placeholder="Paste an essay (at least 100 words, English)...")
            ss.text = ss.get("essay_input", "")
            n_words = len(ss.text.split())
            cl, cr = st.columns([3, 1], vertical_alignment="center")
            with cl:
                st.caption(f"{n_words} words  ·  Built for **student essays, school or university** (100+ words, "
                           "English). Not validated for work reports, technical writing, news or published articles: "
                           "on those it can raise false alarms, so it refuses text that does not look like a student "
                           "essay. A verdict is evidence, never proof.")
            with cr:
                go = st.button("Analyze essay", type="primary", icon=":material/arrow_forward:", width="stretch")
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
                    ss.analysis = {"error": f"Something went wrong while analyzing ({type(e).__name__}). "
                                            "Try again or use demo mode."}

    a = ss.analysis
    if a:
        md('<div class="pg-section-label">Analysis</div>')
        if "error" in a:
            note(E(a["error"]), "warn")
        else:
            if a.get("scope", {}).get("out") and a["verdict"] != "out of scope":
                note("You asked for this verdict although the scope check says this is <b>not</b> a student essay. "
                     "Treat it as unreliable.", "danger")
            c1, c2 = st.columns([1.05, 1], gap="large")
            with c1:
                render_verdict(a)
                render_prompt(a)
            with c2:
                if a["verdict"] != "out of scope":
                    render_reasons(a)
                if a["verdict"] == "AI":
                    note("A verdict is evidence, not proof. In our tests the model wrongly and confidently called "
                         "<b>0 of 60</b> held-out university essays AI, but about <b>5%</b> of human texts in harder "
                         "groups (non-native students' essays; adult writing that passed the scope check). Treat an AI "
                         "verdict as a reason to talk to the writer, never as proof.", "warn")
                    with st.container(key="verify_panel"):
                        md('<div class="pg-card-h"><span>Next step</span></div>'
                           '<p class="pg-muted" style="margin:0 0 12px 0">Regenerate the essay from this prompt and '
                           'score how closely it matches the original.</p>')
                        st.button("Go to Verify", type="primary", icon=":material/arrow_forward:",
                                  on_click=go_to, args=("Verify",))
                if a["verdict"] == "out of scope":
                    if st.button("Show the detector's verdict anyway (unreliable for this kind of text)",
                                 icon=":material/visibility:"):
                        llm = load_llm(gk, qk) if (gk or qk) else None
                        ss.analysis = analyze_cached(ss.text, bool(llm), gk, qk, force=True)
                        st.rerun()
    else:
        md('<div class="pg-section-label">How it works</div>')
        tiles(TILES)

elif page == "Verify":
    page_head("Step 03 &middot; Verify", "Does the prompt rebuild the essay?",
              "We ask an LLM to write the essay again from the reconstructed prompt, three times, plus one rewrite "
              "from a deliberately wrong prompt as a control, and score the match.")
    a = ss.analysis
    if not a or "error" in a:
        note("Analyze an essay first on the <b>Detect</b> page.", "plain")
        st.button("Go to Detect", icon=":material/arrow_back:", on_click=go_to, args=("Detect & Reconstruct",))
    elif a["verdict"] != "AI":
        note("Verification applies to essays judged AI-generated.", "plain")
    else:
        card("Prompt being tested", f'<div class="pg-prompt">{E(a["prompt"])}</div>')
        if use_demo and ss.verify:
            render_verify(ss.verify, ss.text)
        else:
            llm = load_llm(gk, qk) if (gk or qk) else None
            if not llm:
                note("Add a Gemini or Groq API key under <b>Settings</b> (top right) to regenerate, or switch on demo mode.",
                     "info")
            elif st.button("Regenerate x3 and score", type="primary", icon=":material/autorenew:"):
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
    page_head("About the model", "How Prompt Genesis works",
              "Three models in a row, each trained and evaluated on held-out data. Every number below comes from "
              "the saved model card.")
    tiles((('01', 'Guard', 'Gradient-boosted trees on writing-style features (sentence rhythm, vocabulary, punctuation, '
            'syntax) and frozen GPT-2 / DistilGPT-2 predictability features (perplexity, token rank, Binoculars-style '
            'ratio). Probabilities are isotonic-calibrated; inside an uncertainty band it abstains.'),
           ('02', 'Detective', 'One classifier per prompt setting (tone, persona, evasion, length), trained on essays '
            'whose prompts are known.'),
           ('03', 'Verifier', 'Regenerates the essay from the reconstructed prompt and scores Words, Structure and Ideas '
            'between a wrong-prompt floor and a self-consistency ceiling.'),
           ('04', 'Same cleaning', 'All text is cleaned by the same <span class="pg-code">normalize()</span> used in '
            'training, so formatting cannot give the answer away.')), "four")
    p = ART / "model_card.json"
    mc = json.load(open(p, encoding="utf-8")) if p.exists() else {}
    if mc:
        md('<div class="pg-section-label">Evaluation &middot; real numbers, see RESULTS.md</div>')
        rows = ""
        for k, v in mc.items():
            if isinstance(v, dict):
                val = "".join(f'<div class="sub"><span>{E(str(sk))}</span><span>{E(str(sv))}</span></div>'
                              for sk, sv in v.items())
            else:
                val = E(str(v))
            rows += f'<div class="pg-kv"><div class="k">{E(str(k))}</div><div class="v">{val}</div></div>'
        card("Model card", rows)
    note("<b>Limitations:</b> short or heavily edited texts, mixed human/AI authorship, and generators unlike any in "
         "training are harder. A verdict is evidence, not proof.", "plain")

# ------------------------------------------------------------------ footer
md('<div class="pg-footer"><div class="pg-brand">'
   '<span class="pg-word">Prompt Genesis</span></div>'
   '<span>A verdict is evidence, never proof.</span></div>')
