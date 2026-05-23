#!/usr/bin/env python3
"""Regenerate fig1_frontier_scatter.pdf as vertically-stacked single-column figure.

Top: per-cell Delta accuracy vs pairwise disagreement (frontier headline).
Bottom: per-cell Delta ECE vs pairwise disagreement.
Both colored by family (encoder vs decoder).
"""
from __future__ import annotations

import glob
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

DECODER_TAGS = ("qwen", "tinyllama", "smollm", "pythia", "llama", "mistral")

def family(pool_id):
    p = pool_id.lower()
    return "decoder" if any(t in p for t in DECODER_TAGS) else "encoder"

# 1) Load per-pool disagreement from diversity files
disagree = {}
for fn in glob.glob("analysis/pool_*_diversity.json"):
    try:
        d = json.load(open(fn))
        pid = d.get("pool_name") or Path(fn).stem.replace("_diversity", "")
        ps = d.get("prediction_space", {})
        if isinstance(ps, dict) and "disagreement_rate" in ps:
            disagree[pid] = ps["disagreement_rate"]
    except Exception:
        pass

# 2) Load per-cell Delta from cm files (accuracy + ece)
pts_acc, pts_ece = [], []   # each (disagreement, delta, family, pool_id)
for fn in glob.glob("analysis/pool_*_cm_*.json"):
    try:
        d = json.load(open(fn))
        pid = d["ensemble"]["pool_id"]
        if pid not in disagree:
            continue
        x = disagree[pid]
        f = family(pid)
        metric = d["metric"]
        if metric == "accuracy":
            pts_acc.append((x, d["accuracy_diff"], f, pid))
        elif metric == "ece":
            pts_ece.append((x, d["ece_diff"], f, pid))
    except Exception:
        pass

print(f"loaded {len(pts_acc)} accuracy cells, {len(pts_ece)} ece cells, {len(disagree)} pools")

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6.0, 6.4), sharex=True)

def scatter(ax, pts, ylabel, title):
    enc_x = [x for x, y, f, _ in pts if f == "encoder"]
    enc_y = [y for x, y, f, _ in pts if f == "encoder"]
    dec_x = [x for x, y, f, _ in pts if f == "decoder"]
    dec_y = [y for x, y, f, _ in pts if f == "decoder"]
    ax.scatter(enc_x, enc_y, c="#1f77b4", s=18, alpha=0.55, edgecolor="none",
               label=f"encoder (n={len(enc_x)})")
    ax.scatter(dec_x, dec_y, c="#d62728", s=18, alpha=0.55, edgecolor="none",
               label=f"decoder (n={len(dec_x)})")
    ax.axhline(0, color="gray", linewidth=0.6, linestyle="--")
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=10)
    ax.legend(loc="upper right", fontsize=8, frameon=True)
    ax.grid(alpha=0.3)

scatter(ax1, pts_acc,
        r"$\Delta$ accuracy (ensemble $-$ baseline)",
        "Frontier (accuracy arm): encoder LOPO $R^2{=}0.60$, decoder $R^2{<}0$")
scatter(ax2, pts_ece,
        r"$\Delta$ ECE (positive $=$ lower ECE)",
        "Frontier (calibration arm): both families improve with disagreement")
ax2.set_xlabel("pairwise prediction disagreement", fontsize=10)

plt.tight_layout()
out = Path("paper/figures/fig1_frontier_scatter.pdf")
plt.savefig(out, bbox_inches="tight")
plt.savefig(str(out).replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
print(f"wrote {out}")
