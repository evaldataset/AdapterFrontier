#!/usr/bin/env python3
"""Figure 1: the apparent PEFT-ensembling benefit is a product of
evaluation protocol.

Panel A (waterfall): the fraction of cells adjudicated SUPPORTED collapses
as evaluation shortcuts are removed — weak baseline (best_of_n, all
families) -> compute-matched baseline (n_rank) -> the clean encoder slice,
where nothing survives. REVERSED rises in mirror image.

Panel B (distribution): per-cell accuracy deltas for the strict encoder
comparison (encoder x n_rank x accuracy, n=92), colored by post-FDR
verdict. The mass sits left of zero; no cell is SUPPORTED.

Reads the same clean subset as analysis/protocol_sensitivity.py
(HellaSwag excluded: stale selection split; GSM8K excluded: contaminated).

Usage: python3 analysis/plot_protocol_sensitivity.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "analysis"))
from protocol_sensitivity import load_accuracy_cells, _verdict_stats  # noqa: E402

C_SUP = "#2E7D32"     # green
C_REV = "#C62828"     # red
C_UNS = "#BDBDBD"     # gray


def main() -> int:
    cells = load_accuracy_cells()
    enc = [c for c in cells if c["family"] == "encoder"]

    stages = [
        ("Weak baseline\n(best-of-N, all)",
         _verdict_stats([c for c in cells if c["baseline_kind"] == "best_of_n"])),
        ("Compute-matched\n(N×rank, all)",
         _verdict_stats([c for c in cells if c["baseline_kind"] == "n_rank"])),
        ("+ clean encoder\nslice",
         _verdict_stats([c for c in enc if c["baseline_kind"] == "n_rank"])),
    ]

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(11, 4.0),
                                   gridspec_kw={"width_ratios": [1.05, 1.0]})

    # ---------------- Panel A: collapse of SUPPORTED ----------------
    x = np.arange(len(stages))
    sup = [s[1]["pct_supported"] for s in stages]
    rev = [s[1]["pct_reversed"] for s in stages]
    w = 0.36
    axA.bar(x - w / 2, sup, w, color=C_SUP, label="SUPPORTED")
    axA.bar(x + w / 2, rev, w, color=C_REV, label="REVERSED")
    for xi, (s, r, st) in enumerate(zip(sup, rev, stages)):
        axA.text(xi - w / 2, s + 0.9, f"{s:.1f}%", ha="center", fontsize=9,
                 color=C_SUP, fontweight="bold")
        axA.text(xi + w / 2, r + 0.9, f"{r:.1f}%", ha="center", fontsize=9,
                 color=C_REV, fontweight="bold")
        axA.text(xi, -3.4, f"n={st[1]['n_cells']}", ha="center", fontsize=8,
                 color="#555555")
    axA.set_xticks(x)
    axA.set_xticklabels([s[0] for s in stages], fontsize=9)
    axA.set_ylabel("% of adjudicated cells", fontsize=10)
    axA.set_title("A  Removing evaluation shortcuts erases the benefit",
                  fontsize=10.5, loc="left", fontweight="bold")
    axA.set_ylim(-5, max(max(sup), max(rev)) * 1.30)
    axA.axhline(0, color="black", lw=0.8)
    axA.legend(frameon=False, fontsize=9, loc="upper right")
    axA.spines[["top", "right"]].set_visible(False)
    axA.annotate("", xy=(2, sup[2] + 2.2), xytext=(0, sup[0] + 2.2),
                 arrowprops=dict(arrowstyle="->", color="#333333", lw=1.2,
                                 connectionstyle="arc3,rad=-0.18"))
    axA.text(1.0, max(sup) * 1.12, "27.2% → 0%", ha="center", fontsize=9.5,
             color="#333333", fontstyle="italic")

    # ---------------- Panel B: strict encoder distribution ----------------
    strict = [c for c in enc if c["baseline_kind"] == "n_rank"]
    d = np.array([c["diff_pp"] for c in strict])
    verd = [c["verdict"] for c in strict]
    colors = [C_REV if v == "reversed" else (C_SUP if v == "supported" else C_UNS)
              for v in verd]
    order = np.argsort(d)
    axB.scatter(d[order], np.arange(len(d)), c=[colors[i] for i in order],
                s=16, edgecolors="none")
    axB.axvline(0, color="black", lw=1.0)
    axB.axvline(d.mean(), color="#1565C0", lw=1.2, ls="--")
    axB.text(d.mean(), len(d) * 1.02, f"mean {d.mean():+.2f}pp",
             color="#1565C0", fontsize=9, ha="center")
    axB.set_xlabel("Δ accuracy vs compute-matched single (pp)", fontsize=10)
    axB.set_ylabel("cells (sorted)", fontsize=10)
    n_rev = sum(1 for v in verd if v == "reversed")
    axB.set_title(f"B  Strict encoder comparison: 0/{len(d)} SUPPORTED, "
                  f"{n_rev} REVERSED", fontsize=10.5, loc="left", fontweight="bold")
    axB.spines[["top", "right"]].set_visible(False)
    axB.set_ylim(-2, len(d) * 1.12)
    # data runs bottom-left -> top-right, so the upper-left corner is free
    axB.legend(handles=[mpatches.Patch(color=C_REV, label="REVERSED"),
                        mpatches.Patch(color=C_UNS, label="unsupported")],
               frameon=False, fontsize=9, loc="upper left")

    fig.tight_layout()
    figdir = ROOT / "paper/figures"
    figdir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(figdir / f"fig1_protocol_sensitivity.{ext}",
                    dpi=200, bbox_inches="tight")

    print("Panel A stages:")
    for label, st in stages:
        print(f"  {label.replace(chr(10), ' '):38s} n={st['n_cells']:4d}  "
              f"SUP {st['pct_supported']:5.1f}%  REV {st['pct_reversed']:5.1f}%")
    print(f"Panel B: n={len(d)} strict encoder cells, mean {d.mean():+.3f}pp, "
          f"reversed {n_rev} ({100*n_rev/len(d):.1f}%), supported "
          f"{sum(1 for v in verd if v=='supported')}")
    print(f"wrote {figdir}/fig1_protocol_sensitivity.{{pdf,png}}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
