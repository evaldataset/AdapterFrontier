#!/usr/bin/env python3
"""Regenerate fig2_scaling_curves.pdf as vertically-stacked single-column figure.

Top: accuracy vs N (subset size). Bottom: ECE vs N. Both with mean +/- std
across 50 random subsets per N, per (model, task). Plateau by N~8-16.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

d = json.load(open("analysis/scaling_curve_cache.json"))

LABEL = {
    "pool_a_mnli_bert":         "BERT-base",
    "pool_a_mnli_qwen25_05b":   "Qwen-0.5B",
    "pool_a_mnli_roberta_base": "RoBERTa-base",
    "pool_a_mnli_tinyllama_11b": "TinyLlama-1.1B",
}
COLOR = {
    "pool_a_mnli_bert":         "#1f77b4",
    "pool_a_mnli_qwen25_05b":   "#d62728",
    "pool_a_mnli_roberta_base": "#2ca02c",
    "pool_a_mnli_tinyllama_11b": "#ff7f0e",
}

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6.0, 5.6), sharex=True)

for pool_id, v in d.items():
    sizes = sorted(int(k) for k in v["by_size"].keys())
    accs   = [v["by_size"][str(n)]["acc_mean"] for n in sizes]
    acc_sd = [v["by_size"][str(n)]["acc_std"]  for n in sizes]
    eces   = [v["by_size"][str(n)]["ece_mean"] for n in sizes]
    ece_sd = [v["by_size"][str(n)]["ece_std"]  for n in sizes]
    label = LABEL.get(pool_id, pool_id)
    color = COLOR.get(pool_id, "k")
    ax1.errorbar(sizes, accs, yerr=acc_sd, label=label, color=color, marker="o",
                 capsize=2, linewidth=1.2, markersize=4)
    ax2.errorbar(sizes, eces, yerr=ece_sd, label=label, color=color, marker="o",
                 capsize=2, linewidth=1.2, markersize=4)

ax1.set_ylabel("test accuracy", fontsize=10)
ax1.set_title("Accuracy vs.\\ pool size N (mean$\\pm$std over 50 random subsets)", fontsize=10)
ax1.grid(alpha=0.3)
ax1.legend(loc="lower right", fontsize=8, frameon=True)

ax2.set_ylabel("test ECE (lower = better)", fontsize=10)
ax2.set_xlabel("pool size $N$", fontsize=10)
ax2.set_title("ECE vs.\\ pool size N (modest decline continues to N=20)", fontsize=10)
ax2.grid(alpha=0.3)
ax2.set_xticks([1, 2, 3, 5, 8, 12, 16, 20])

plt.tight_layout()
out = Path("paper/figures/fig2_scaling_curves.pdf")
plt.savefig(out, bbox_inches="tight")
plt.savefig(str(out).replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
print(f"wrote {out}")
for pid, v in d.items():
    print(f"  {pid}: sizes={sorted(int(k) for k in v['by_size'].keys())}")
