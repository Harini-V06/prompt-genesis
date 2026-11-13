"""
LM + embedding features from FROZEN pretrained models (feature extractors only,
nothing is trained). Run on the LOCAL machine (needs internet once to download
the models, ~1 GB). Shared with the app via lm_features().

  python src/features_lm.py                 # scores artifacts/texts_for_lm.parquet
  python src/features_lm.py --batch 2       # if you run out of RAM

Models
  observer  = gpt2          (124M)  } same GPT-2 tokenizer -> cross-perplexity
  performer = distilgpt2    (82M)   } (Binoculars-style, Hans et al. 2024)
  embedder  = sentence-transformers/all-MiniLM-L6-v2

Every text is scored on its FIRST 128 GPT-2 tokens only, so all statistics are
per-token averages over a fixed window (no length leakage).

Features (per text)
  lm_{obs,perf}_lp_mean/std/skew   token log-prob distribution
  lm_{obs,perf}_ppl_log            mean NLL (log perplexity)
  lm_{obs,perf}_entropy_mean       mean predictive entropy
  lm_obs_rank_{1,10,100,1000,gt1000}  GLTR rank histogram (shares)
  lm_obs_logrank_mean
  lm_xent                          cross-entropy: -sum p_perf * log p_obs (mean over tokens)
  lm_binoculars                    lm_obs_ppl_log / lm_xent
  lm_obs_perf_lp_gap               mean(lp_obs - lp_perf)
  emb_coh_adj_mean/std             cosine of adjacent sentence embeddings
  emb_coh_centroid_mean/std        cosine of each sentence to the document centroid
Also saves the 384-d document embedding (artifacts/cache/emb_doc_*.npy) for the
Verifier's Ideas score and the MiniLM near-duplicate check (never used as Guard
features: raw embeddings mostly encode TOPIC, which would reintroduce the topic
confound).

Checkpointed in chunks of 256 rows (artifacts/cache/lm_chunk_XXXX.parquet);
rerun to resume. Final: artifacts/features_lm.parquet
"""
import argparse
import re
import sys
import time
from pathlib import Path

import os

ROOT = Path(__file__).resolve().parents[1]
# Keep the ~1 GB model cache and temp files on the PROJECT drive, not C:
# (the user's C: drive is full). Respects an HF_HOME the user already set.
# On a server where that folder is not writable (e.g. Streamlit Community Cloud) the
# library defaults are kept.
_CACHE = ROOT.parent / "hf_cache"
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
try:
    (_CACHE / "tmp").mkdir(parents=True, exist_ok=True)
    _probe = _CACHE / "tmp" / ".write_test"
    _probe.write_text("ok"); _probe.unlink()
    os.environ.setdefault("HF_HOME", str(_CACHE))
    for _v in ("TMP", "TEMP", "TMPDIR"):
        os.environ[_v] = str(_CACHE / "tmp")
except OSError:
    pass

import numpy as np
import pandas as pd
MAXTOK = 128  # = app minimum (100 words ~ 128 GPT-2 tokens): same window for every text
_M = {}


THREADS = None  # set from --threads


def load(device=None):
    if _M:
        return _M
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from sentence_transformers import SentenceTransformer
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    # default: HALF the cores. Using every core overheated the user's laptop (thermal throttling, then shutdown)
    torch.set_num_threads(THREADS or max(1, (os.cpu_count() or 2) // 2))
    tok = AutoTokenizer.from_pretrained("gpt2")
    tok.pad_token = tok.eos_token
    obs = AutoModelForCausalLM.from_pretrained("gpt2").to(dev).eval()
    perf = AutoModelForCausalLM.from_pretrained("distilgpt2").to(dev).eval()
    emb = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=dev)
    _M.update(torch=torch, tok=tok, obs=obs, perf=perf, emb=emb, dev=dev)
    return _M


def _skew(x):
    s = x.std()
    return float(((x - x.mean()) ** 3).mean() / s ** 3) if s > 0 else 0.0


def _token_stats(texts, batch, progress=None):
    M = load()
    torch, tok = M["torch"], M["tok"]
    rows = []
    t0 = time.time()
    for i in range(0, len(texts), batch):
        if progress and i and i % (batch * 8) == 0:
            rate = i / (time.time() - t0)
            print(f"   {progress}: {i}/{len(texts)} texts  ({rate:.2f} texts/s)", flush=True)
        enc = tok([tok.bos_token + t for t in texts[i:i + batch]], return_tensors="pt", padding=True,
                  truncation=True, max_length=MAXTOK + 1).to(M["dev"])
        ids, att = enc["input_ids"], enc["attention_mask"]
        with torch.no_grad():
            lo = torch.log_softmax(M["obs"](**enc).logits[:, :-1].float(), -1)
            lp_ = torch.log_softmax(M["perf"](**enc).logits[:, :-1].float(), -1)
        tgt, mask = ids[:, 1:], att[:, 1:].bool()
        tlo = lo.gather(-1, tgt[..., None])[..., 0]
        tlp = lp_.gather(-1, tgt[..., None])[..., 0]
        rank = (lo > tlo[..., None]).sum(-1) + 1
        ent_o = -(lo.exp() * lo).sum(-1)
        ent_p = -(lp_.exp() * lp_).sum(-1)
        xent = -(lp_.exp() * lo).sum(-1)
        for b in range(ids.shape[0]):
            m = mask[b]
            o, p, r = tlo[b][m].cpu().numpy(), tlp[b][m].cpu().numpy(), rank[b][m].cpu().numpy()
            f = {}
            for nm, v, e in (("obs", o, ent_o[b][m]), ("perf", p, ent_p[b][m])):
                f[f"lm_{nm}_lp_mean"] = float(v.mean())
                f[f"lm_{nm}_lp_std"] = float(v.std())
                f[f"lm_{nm}_lp_skew"] = _skew(v)
                f[f"lm_{nm}_ppl_log"] = float(-v.mean())
                f[f"lm_{nm}_entropy_mean"] = float(e.mean().item())
            for k, lo_, hi in (("1", 1, 1), ("10", 2, 10), ("100", 11, 100), ("1000", 101, 1000)):
                f[f"lm_obs_rank_{k}"] = float(((r >= lo_) & (r <= hi)).mean())
            f["lm_obs_rank_gt1000"] = float((r > 1000).mean())
            f["lm_obs_logrank_mean"] = float(np.log(r).mean())
            f["lm_xent"] = float(xent[b][m].mean().item())
            f["lm_binoculars"] = f["lm_obs_ppl_log"] / f["lm_xent"] if f["lm_xent"] else np.nan
            f["lm_obs_perf_lp_gap"] = float((o - p).mean())
            rows.append(f)
    return rows


_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")


def _emb_stats(texts):
    M = load()
    out, docs = [], []
    for t in texts:
        sents = [s for s in _SENT.split(t) if len(s.split()) >= 3][:40] or [t]
        E = M["emb"].encode(sents, normalize_embeddings=True, show_progress_bar=False)
        c = E.mean(0); c /= np.linalg.norm(c) + 1e-9
        adj = (E[1:] * E[:-1]).sum(1) if len(E) > 1 else np.array([1.0])
        cen = E @ c
        out.append({"emb_coh_adj_mean": float(adj.mean()), "emb_coh_adj_std": float(adj.std()),
                    "emb_coh_centroid_mean": float(cen.mean()), "emb_coh_centroid_std": float(cen.std())})
        docs.append(c.astype(np.float16))
    return out, np.stack(docs)


def lm_features(texts, batch=4):
    """For the app / any caller: list[str] -> DataFrame of LM + embedding-stat features."""
    tok_rows = _token_stats(list(texts), batch)
    emb_rows, _ = _emb_stats(list(texts))
    return pd.DataFrame([{**a, **b} for a, b in zip(tok_rows, emb_rows)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=str(ROOT / "artifacts" / "texts_for_lm.parquet"))
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--chunk", type=int, default=128)
    ap.add_argument("--threads", type=int, default=None, help="CPU threads (default: half the cores)")
    a = ap.parse_args()
    global THREADS
    THREADS = a.threads
    df = pd.read_parquet(a.input)
    stem = Path(a.input).stem
    cache = ROOT / "artifacts" / "cache" / stem
    cache.mkdir(parents=True, exist_ok=True)
    out_name = "features_lm.parquet" if stem == "texts_for_lm" else f"features_lm_{stem}.parquet"
    emb_name = "emb_doc" if stem == "texts_for_lm" else f"emb_doc_{stem}"
    M = load()
    print(f"device={M['dev']} | threads={M['torch'].get_num_threads()} | rows={len(df)} | chunks of {a.chunk}", flush=True)
    t0, n_chunks = time.time(), (len(df) + a.chunk - 1) // a.chunk
    skipped = 0
    for ci in range(n_chunks):
        fp = cache / f"lm_chunk_{ci:04d}.parquet"
        if fp.exists():
            skipped += 1
            continue
        part = df.iloc[ci * a.chunk:(ci + 1) * a.chunk]
        tok_rows = _token_stats(part.text.tolist(), a.batch, progress=f"chunk {ci+1}/{n_chunks}")
        emb_rows, E = _emb_stats(part.text.tolist())
        F = pd.DataFrame([{**x, **y} for x, y in zip(tok_rows, emb_rows)])
        F.insert(0, "uid", part.uid.values)
        F.to_parquet(fp, index=False)
        np.save(cache / f"emb_doc_{ci:04d}.npy", E)
        done = ci + 1
        el = time.time() - t0
        left = (n_chunks - done) * el / max(1, done - skipped)
        print(f"chunk {done}/{n_chunks} saved  ({el/60:.1f} min elapsed, ~{left/3600:.1f} h left)", flush=True)
    F = pd.concat([pd.read_parquet(cache / f"lm_chunk_{ci:04d}.parquet") for ci in range(n_chunks)], ignore_index=True)
    F.to_parquet(ROOT / "artifacts" / out_name, index=False)
    E = np.concatenate([np.load(cache / f"emb_doc_{ci:04d}.npy") for ci in range(n_chunks)])
    np.save(ROOT / "artifacts" / f"{emb_name}.npy", E)
    pd.Series(df.uid.values).to_csv(ROOT / "artifacts" / f"{emb_name}_uids.csv", index=False, header=["uid"])
    print(f"DONE: {len(F)} rows x {F.shape[1]-1} features -> artifacts/{out_name}; embeddings saved")


if __name__ == "__main__":
    main()
