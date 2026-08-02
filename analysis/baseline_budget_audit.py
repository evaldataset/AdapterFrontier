#!/usr/bin/env python3
"""Measure what the `n_rank` baseline actually cost, per pool.

The paper described `n_rank` as "compute-matched". Adversarial review
challenged this, and the released artifacts settle it: the baseline sweeps
train at epochs {2,4} (SST-2 {3,6}) while the pools train at a fixed epoch
count, so half the baseline runs see twice the tokens per adapter. This
script sums the per-adapter `train_runtime_s` recorded in every pool
manifest and reports the realised baseline/pool ratio.

It also records two asymmetries that matter for interpreting the null:
  - the baseline is a best-of-N selection over a rank x epoch x seed grid
    (chosen on val_selection); the ensemble arm gets no HP search at all;
  - the baseline serves at 1x inference cost, the ensemble at Nx.

Both make `n_rank` a STRICT UPPER-BOUND competitor rather than a matched
one, which makes the accuracy null conservative and the calibration win
harder-earned.

Usage: python3 analysis/baseline_budget_audit.py
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def gpu_hours(manifest: Path) -> tuple[float, int]:
    """Sum train_runtime_s over a manifest's adapters -> (GPU-hours, n)."""
    try:
        d = json.loads(manifest.read_text())
    except Exception:
        return float("nan"), 0
    total, n = 0.0, 0
    for a in d.get("adapters", []):
        m = a.get("metrics") or {}
        r = m.get("train_runtime_s") or m.get("train_runtime")
        if r:
            total += float(r)
            n += 1
    return total / 3600.0, n


def baseline_for(pool_id: str) -> Path | None:
    """pool_a_mnli_bert -> baseline_compute_matched_mnli_bert."""
    stem = re.sub(r"^pool_[abc]_", "", pool_id)
    stem = re.sub(r"_method_mixed$", "", stem)
    cand = ROOT / f"pools/baseline_compute_matched_{stem}.json"
    return cand if cand.exists() else None


def main() -> int:
    rows = []
    for mf in sorted((ROOT / "pools").glob("pool_*.json")):
        pool_id = mf.stem
        if any(t in pool_id for t in ("hellaswag", "gsm8k", "pilot")):
            continue
        bf = baseline_for(pool_id)
        if bf is None:
            continue
        ph, pn = gpu_hours(mf)
        bh, bn = gpu_hours(bf)
        if not (ph > 0 and bh > 0):
            continue
        rows.append({
            "pool_id": pool_id, "baseline": bf.stem,
            "pool_gpu_h": round(ph, 3), "pool_n": pn,
            "baseline_gpu_h": round(bh, 3), "baseline_n": bn,
            "ratio": round(bh / ph, 3),
        })

    if not rows:
        print("no (pool, baseline) pairs with runtime data")
        return 1

    ratios = np.array([r["ratio"] for r in rows])
    print(f"{'pool':42s} {'pool h':>7s} {'base h':>7s} {'ratio':>6s}  n(pool/base)")
    print("-" * 82)
    for r in sorted(rows, key=lambda x: -x["ratio"]):
        print(f"{r['pool_id']:42s} {r['pool_gpu_h']:7.2f} {r['baseline_gpu_h']:7.2f} "
              f"{r['ratio']:6.2f}  {r['pool_n']}/{r['baseline_n']}")
    print("-" * 82)
    print(f"pairs={len(rows)}  ratio: min={ratios.min():.2f} median={np.median(ratios):.2f} "
          f"mean={ratios.mean():.2f} max={ratios.max():.2f}")
    print(f"pairs where baseline got MORE compute: {(ratios > 1.05).sum()}/{len(rows)}")

    out = {
        "note": ("Realised training cost of the n_rank baseline relative to the pool it "
                 "is compared against, summed from per-adapter train_runtime_s in the "
                 "released pool manifests. Ratios > 1 mean the baseline received more "
                 "training compute than the pool, so it is a strict upper-bound "
                 "competitor rather than a compute-matched one."),
        "n_pairs": len(rows),
        "ratio_min": float(ratios.min()),
        "ratio_median": float(np.median(ratios)),
        "ratio_mean": float(ratios.mean()),
        "ratio_max": float(ratios.max()),
        "n_pairs_baseline_favoured": int((ratios > 1.05).sum()),
        "pairs": sorted(rows, key=lambda x: -x["ratio"]),
    }
    (ROOT / "analysis/baseline_budget_audit.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote {ROOT / 'analysis/baseline_budget_audit.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
