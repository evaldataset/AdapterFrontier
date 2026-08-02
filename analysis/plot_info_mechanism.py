#!/usr/bin/env python3
"""Regenerate fig4_info_mechanism.pdf as a vertically-stacked single-column figure.

Top panel: per-pool H(c|conf). Bottom: per-pool I(c;conf). Decoder family sits
systematically lower on H and higher on I.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

FAMILY_COLOR = {
    "encoder_bertfam": "#1f77b4",
    "encoder_deberta": "#9467bd",
    "decoder":         "#d62728",
}
FAMILY_LABEL = {
    "encoder_bertfam": "encoder (BERT/RoBERTa)",
    "encoder_deberta": "encoder (DeBERTa-v3)",
    "decoder":         "decoder",
}

d = json.load(open("analysis/info_mechanism.json"))
pools = d["per_pool"]

fams = ["encoder_bertfam", "encoder_deberta", "decoder"]
xs = {f: [] for f in fams}
hs = {f: [] for f in fams}
is_ = {f: [] for f in fams}
for p in pools:
    f = p.get("family", "decoder")
    xs[f].append(p["pool_id"])
    hs[f].append(p["H_corr_given_conf_bits"])
    is_[f].append(p["I_corr_conf_bits"])

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6.0, 5.6), sharex=False)

# Top: H(c|conf)
x_off = 0
for f in fams:
    n = len(hs[f])
    ax1.scatter(np.arange(x_off, x_off + n), hs[f],
                c=FAMILY_COLOR[f], s=36, alpha=0.8, edgecolor="k", linewidth=0.4,
                label=FAMILY_LABEL[f])
    mean = float(np.mean(hs[f])) if hs[f] else 0
    ax1.hlines(mean, x_off - 0.4, x_off + n - 0.6, colors=FAMILY_COLOR[f],
               linestyles="--", linewidth=1.4)
    x_off += n
ax1.set_ylabel(r"$H(c\,|\,\mathrm{conf})$  [bits]", fontsize=10)
ax1.set_title("Per-pool conditional entropy of correctness given confidence", fontsize=10)
ax1.set_xticks([])
ax1.legend(loc="upper right", fontsize=8, frameon=True)
ax1.grid(axis="y", alpha=0.3)

# Bottom: I(c;conf)
x_off = 0
for f in fams:
    n = len(is_[f])
    ax2.scatter(np.arange(x_off, x_off + n), is_[f],
                c=FAMILY_COLOR[f], s=36, alpha=0.8, edgecolor="k", linewidth=0.4,
                label=FAMILY_LABEL[f])
    mean = float(np.mean(is_[f])) if is_[f] else 0
    ax2.hlines(mean, x_off - 0.4, x_off + n - 0.6, colors=FAMILY_COLOR[f],
               linestyles="--", linewidth=1.4)
    x_off += n
ax2.set_ylabel(r"$I(c\,;\,\mathrm{conf})$  [bits]", fontsize=10)
ax2.set_xlabel("pool index (grouped by family)", fontsize=10)
ax2.set_title("Per-pool mutual information between correctness and confidence", fontsize=10)
ax2.set_xticks([])
ax2.grid(axis="y", alpha=0.3)

plt.tight_layout()
out = Path("paper/figures/fig4_info_mechanism.pdf")
plt.savefig(out, bbox_inches="tight")
plt.savefig(str(out).replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
print(f"wrote {out}")

fs = d["family_summary"]
for f in fams:
    if f in fs:
        s = fs[f]
        print(f"  {f}: n={s['n_pools']} mean_H={s['mean_H_corr_given_conf_bits']:.4f} mean_I={s['mean_I_corr_conf_bits']:.4f}")
