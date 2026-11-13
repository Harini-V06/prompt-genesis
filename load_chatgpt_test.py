"""
Parse the hand-collected real-chatbot essays (data/chatgpt_test/{chatgpt,claude}_essays.txt).
Format: "PROMPT n:" line, then the essay. n indexes PROMPTS below (the exact text pasted into the chatbot,
new chat per essay, essays unedited). TEST ONLY: never used for training, tuning or thresholds;
scored once with the final model. Topics deliberately differ from all training topics.

Claude file notes (data/chatgpt_test/claude_essays_original.txt is the untouched paste):
- 8 of 15 prompts collected; pasted out of order, renumbered by topic (1->2, 2->1, 4->8, 5->9, 6->5, 8->4).
- prompt 4 (four-day week): Claude used web search; the paste contained citation-chip lines (source names
  like "kutv", "+3"). Those lines were removed; no essay sentence was changed.
- prompt 7 (zoos): the paste starts at "The case against" (intro not copied). Kept as is.
- prompt 8 (second language): mentions Abu Dhabi (the chatbot personalised it). Kept as is.
"""
import re
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROMPTS = {
 1: "Write an essay about whether homework should be banned in primary schools.",
 2: "Write an essay about the impact of social media on teenagers' sleep.",
 3: "Write an essay about why cities should invest in public libraries.",
 4: "Write an essay about the pros and cons of a four-day school week.",
 5: "Write an essay about how video games can teach problem solving.",
 6: "Write an essay about the role of volunteering in building character. Make it sound like a high school student wrote it.",
 7: "Write an essay about whether zoos should still exist.",
 8: "Write an essay about the importance of learning a second language. Keep it around 300 words.",
 9: "Write an essay about how fast fashion affects the environment.",
 10: "Write a personal essay about a time you failed and what you learned.",
 11: "Write an essay arguing for or against school uniforms. Make it sound human, not like an AI.",
 12: "Write an essay about whether students should be allowed to use AI for homework.",
 13: "Write an essay about the benefits of walking or cycling to school.",
 14: "Write an essay about how space exploration benefits life on Earth. Use a formal tone.",
 15: "Write an essay about what makes a good teacher.",
}


def load():
    rows = []
    for src in ("chatgpt", "claude"):
        f = ROOT / "data" / "chatgpt_test" / f"{src}_essays.txt"
        if not f.exists():
            continue
        parts = re.split(r"^\s*PROMPT\s+(\d+)\s*:\s*$", f.read_text(encoding="utf-8"), flags=re.M)
        for n, body in zip(parts[1::2], parts[2::2]):
            body = body.strip()
            if body:
                rows.append({"uid": f"realchat_{src}_{int(n):02d}", "source": src, "prompt_id": int(n),
                             "prompt": PROMPTS[int(n)], "text_raw": body, "label": 1,
                             "n_words": len(body.split())})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    d = load()
    print(d.groupby("source").agg(n=("uid", "size"), words_min=("n_words", "min"), words_med=("n_words", "median")))
    assert not d.duplicated(["source", "prompt_id"]).any()
