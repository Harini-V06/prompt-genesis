# PROGRESS

**Status 14 Nov 2026 02:30: complete.** See CHECKLIST.md for every deliverable and where it lives.
Deployed: https://promptgenesis.streamlit.app (repo github.com/Harini-V06/prompt-genesis, flat layout).
Deployed models: Guard = fixed-config XGBoost on 68 stylometric + 23 LM features, isotonic, band 0.04-0.96;
Detective = tuned per-label heads with confidence cut-offs. All XGBoost models stored portably (src/portable.py).

Future work (in priority order): style-only Guard (equal accuracy, no PyTorch); labelled essays from real chatbot
apps for the Detective; non-native human essays in training; teacher_sim evaluation.
