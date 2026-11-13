"""
Calibration (reliability diagram) and abstention (accuracy vs coverage) plots from the
Guard's out-of-fold predictions. Every point is an outer-test prediction: no plot uses
training data.

  python src/plots.py reports/guard_oof_logo_stylo_lm_essay.parquet
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RED, GREY = "#dc2626", "#737373"


def reliability(P, ax, bins=10):
    y, p = P.label.values, P.p_guard.values
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    xs, ys, ns = [], [], []
    for b in range(bins):
        m = idx == b
        if m.sum():
            xs.append(p[m].mean()); ys.append(y[m].mean()); ns.append(m.sum())
    ax.plot([0, 1], [0, 1], ls="--", color=GREY, lw=1, label="perfect calibration")
    ax.plot(xs, ys, marker="o", color=RED, label="Guard (out-of-fold)")
    for x, yy, n in zip(xs, ys, ns):
        ax.annotate(str(n), (x, yy), textcoords="offset points", xytext=(4, -10), fontsize=7, color=GREY)
    ece = sum(abs(a - b) * n for a, b, n in zip(xs, ys, ns)) / sum(ns)
    ax.set_xlabel("predicted P(AI)"); ax.set_ylabel("observed fraction AI")
    ax.set_title(f"Reliability diagram  (ECE {ece:.3f}, n={len(y)})", fontsize=10)
    ax.legend(fontsize=8, loc="upper left")


def coverage_curve(P, ax):
    y, p = P.label.values, P.p_guard.values
    conf = np.abs(p - 0.5)
    order = np.argsort(-conf)
    correct = ((p >= 0.5) == y)[order]
    cov = np.arange(1, len(y) + 1) / len(y)
    acc = np.cumsum(correct) / np.arange(1, len(y) + 1)
    ax.plot(cov * 100, acc * 100, color=RED)
    for c in (1.0, 0.9, 0.8):
        k = int(c * len(y)) - 1
        ax.scatter(cov[k] * 100, acc[k] * 100, color="black", s=12, zorder=3)
        ax.annotate(f"{acc[k]*100:.1f}% at {c*100:.0f}% answered", (cov[k] * 100, acc[k] * 100),
                    textcoords="offset points", xytext=(-135, 4), fontsize=8)
    ax.set_xlabel("% of essays answered (most confident first)"); ax.set_ylabel("accuracy on answered (%)")
    ax.set_title("Abstention: accuracy vs coverage", fontsize=10)
    ax.set_xlim(0, 101)


def main(path):
    P = pd.read_parquet(path)
    stem = Path(path).stem.replace("guard_oof_", "")
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.4))
    reliability(P, axs[0]); coverage_curve(P, axs[1])
    for a in axs:
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
    fig.suptitle(f"Guard calibration and abstention ({stem})", fontsize=11)
    plt.tight_layout()
    out = f"reports/plot_calib_abstain_{stem}.png"
    plt.savefig(out, dpi=160)
    print("saved", out)


if __name__ == "__main__":
    main(sys.argv[1])
