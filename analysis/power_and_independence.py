#!/usr/bin/env python3
"""Is "0 of 92" evidence of absence, or absence of evidence?

A null result needs two things the paper did not report: how precise each
cell actually is, and how many independent tests 92 really represents.
Review raised both. This script answers them from the released cells.

(A) Precision and equivalence. For each cell we have a paired bootstrap CI on
    the accuracy difference. A cell whose CI upper bound sits above +1pp has
    not ruled out a meaningful gain -- it is uninformative about the positive
    direction, not evidence against it. We report CI half-widths, stratify by
    test-set size, and run the equivalence test the null actually needs:
    what fraction of cells exclude a gain of delta for delta in {0.5, 1.0}pp.

(B) Independence. The 92 cells are 23 pools x 4 combination methods, and two
    of those methods (soft_vote, logit_avg) are near-identical operations. We
    measure how often they agree, so "the result holds for every combination
    rule" can be stated at its true strength.

Usage: python3 analysis/power_and_independence.py
"""
from __future__ import annotations

import json
import glob
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DECODER = ("qwen", "llama", "mistral", "tinyllama", "pythia", "smollm")
PAT = re.compile(r"^(?P<pool>.+?)_cm_(?P<m>[a-z_]+?)_vs_"
                 r"(?P<k>best_of_n|n_rank|n_steps|n_data)(?P<ece>_ECE)?\.json$")


def load_strict_cells() -> list[dict]:
    out = []
    for f in sorted(glob.glob(str(ROOT / "analysis/*_cm_*.json"))):
        m = PAT.match(Path(f).name)
        if not m or m.group("ece") or m.group("k") != "n_rank":
            continue
        pool = m.group("pool")
        pl = pool.lower()
        if "hellaswag" in pl or "gsm8k" in pl or any(t in pl for t in DECODER):
            continue
        d = json.loads(Path(f).read_text())
        if d.get("accuracy_diff") is None:
            continue
        out.append({
            "pool": pool, "method": m.group("m"),
            "diff_pp": 100 * d["accuracy_diff"],
            "lo_pp": 100 * d["ci_low"], "hi_pp": 100 * d["ci_high"],
            "n_test": d.get("n_test"),
        })
    return out


def main() -> int:
    cells = load_strict_cells()
    n = len(cells)
    half = np.array([(c["hi_pp"] - c["lo_pp"]) / 2 for c in cells])
    hi = np.array([c["hi_pp"] for c in cells])
    ntest = np.array([c["n_test"] or 0 for c in cells])

    print(f"strict encoder x n_rank x accuracy cells: {n} "
          f"from {len(set(c['pool'] for c in cells))} pools\n")

    print("(A) precision")
    print(f"  CI half-width  mean {half.mean():.2f}pp   median {np.median(half):.2f}pp   "
          f"max {half.max():.2f}pp")
    equiv = {}
    for delta in (0.5, 1.0):
        excl = int((hi < delta).sum())
        equiv[str(delta)] = {"excluded": excl, "pct": round(100 * excl / n, 1)}
        print(f"  cells excluding a +{delta}pp gain: {excl}/{n} ({100*excl/n:.1f}%)")

    print("\n  stratified by test-set size")
    strata = [("n_test >= 2000", ntest >= 2000), ("n_test < 2000", ntest < 2000)]
    strat_out = {}
    for name, mask in strata:
        if mask.sum() == 0:
            continue
        h, u = half[mask], hi[mask]
        rec = {"n": int(mask.sum()),
               "median_half_width_pp": float(np.median(h)),
               "excl_0.5pp": int((u < 0.5).sum()),
               "excl_1.0pp": int((u < 1.0).sum())}
        strat_out[name] = rec
        print(f"    {name:16s} n={rec['n']:3d}  median half-width {rec['median_half_width_pp']:5.2f}pp  "
              f"exclude +0.5pp {rec['excl_0.5pp']:3d}  exclude +1.0pp {rec['excl_1.0pp']:3d}")

    print("\n(B) independence of the 92 cells")
    by_pool = defaultdict(dict)
    for c in cells:
        by_pool[c["pool"]][c["method"]] = c["diff_pp"]
    pairs, close, identical = 0, 0, 0
    for pool, m in by_pool.items():
        if "soft_vote" in m and "logit_avg" in m:
            pairs += 1
            d = abs(m["soft_vote"] - m["logit_avg"])
            close += int(d < 0.1)
            identical += int(d == 0.0)
    print(f"  pools with both soft_vote and logit_avg: {pairs}")
    print(f"    agree to <0.1pp: {close}/{pairs}    bit-identical: {identical}/{pairs}")
    print(f"  cells = {len(by_pool)} pools x {n/max(len(by_pool),1):.0f} methods; "
          f"effective independent rules ~= {4 - (close/max(pairs,1)):.1f}")

    out = {
        "n_cells": n, "n_pools": len(set(c["pool"] for c in cells)),
        "ci_half_width_pp": {"mean": float(half.mean()), "median": float(np.median(half)),
                             "max": float(half.max())},
        "equivalence": equiv,
        "by_test_size": strat_out,
        "method_redundancy": {
            "pools_with_both": pairs, "agree_within_0.1pp": close,
            "bit_identical": identical,
        },
        "note": ("The 0/92 null is only as strong as the cells are precise. Cells whose CI "
                 "upper bound exceeds the equivalence margin have not ruled out a gain; we "
                 "report them rather than counting them as evidence of absence. The 92 cells "
                 "are 23 pools x 4 methods and two methods are near-duplicates, so they are "
                 "not 92 independent tests."),
    }
    p = ROOT / "analysis/power_and_independence.json"
    p.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
