#!/usr/bin/env python3
"""Does computing diversity features on the full eval set leak the test split?

diversity_metrics.py loads each adapter's cached logits with no split
indexing, so the prediction-space features (pairwise disagreement, logit
correlation, ensemble entropy) are computed over ALL evaluation examples --
including the 40% test slice on which the regression target (Delta accuracy)
is measured. Leave-one-pool-out CV holds out pools, not examples, so it does
not remove this overlap. A reviewer raised it; this script measures it.

What we can and cannot do:
  - CANNOT re-fit the published frontier leak-free. The logits cache behind
    the 48 frontier pools was not retained locally and regenerating it needs
    the adapter weights, which live on the cluster.
  - CAN measure how much the features themselves move when restricted to
    val_selection, on every pool whose cache we still have. If the features
    are stable, the leak is immaterial; if they move, the published R^2 is
    suspect and must be re-derived once the cache is regenerated.

Usage: python3 analysis/leakage_check_diversity.py
"""
from __future__ import annotations

import json
import glob
import itertools
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "ensemble_cache"


def softmax(x):
    e = np.exp(x - x.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


def features(logits: np.ndarray) -> dict:
    """logits: (M adapters, N examples, K classes) -> the three
    prediction-space features diversity_metrics.py reports."""
    preds = logits.argmax(-1)
    M = logits.shape[0]
    dis = [float((preds[i] != preds[j]).mean())
           for i, j in itertools.combinations(range(M), 2)]
    cors = []
    for i, j in itertools.combinations(range(M), 2):
        for k in range(logits.shape[-1]):
            a, b = logits[i, :, k], logits[j, :, k]
            if a.std() > 0 and b.std() > 0:
                cors.append(float(np.corrcoef(a, b)[0, 1]))
    p = softmax(logits).mean(0)
    ent = float((-(p * np.log(p + 1e-12)).sum(-1)).mean())
    return {"disagreement_rate": float(np.mean(dis)),
            "logit_correlation": float(np.mean(cors)) if cors else float("nan"),
            "ensemble_entropy": ent}


def main() -> int:
    rows = []
    for rf in sorted(glob.glob(str(ROOT / "ensemble_results/pilot_c1/*.json"))):
        d = json.loads(Path(rf).read_text())
        pool_id = d.get("pool_id")
        sel = d.get("val_selection_indices")
        if not sel:
            continue
        npys = sorted(CACHE.glob(f"{pool_id}__*.npy"))
        if len(npys) < 2:
            continue
        stack = np.stack([np.load(f) for f in npys], axis=0)
        sel = np.asarray(sel)
        full = features(stack)                 # what the paper used
        valonly = features(stack[:, sel, :])   # leak-free
        rows.append({
            "pool_id": pool_id, "n_adapters": int(stack.shape[0]),
            "n_eval": int(stack.shape[1]), "n_val_selection": int(len(sel)),
            "full": full, "val_only": valonly,
            "delta": {k: valonly[k] - full[k] for k in full},
        })

    if not rows:
        print("no pool has both a cached logits stack and stored val_selection indices")
        return 1

    keys = ("disagreement_rate", "logit_correlation", "ensemble_entropy")
    print(f"{'pool':46s} " + " ".join(f"{k[:12]:>26s}" for k in keys))
    print(f"{'':46s} " + " ".join(f"{'full -> val (delta)':>26s}" for _ in keys))
    print("-" * 130)
    for r in rows:
        cells = []
        for k in keys:
            cells.append(f"{r['full'][k]:.4f}->{r['val_only'][k]:.4f} ({r['delta'][k]:+.4f})".rjust(26))
        print(f"{r['pool_id'][:46]:46s} " + " ".join(cells))

    print("-" * 130)
    summary = {}
    for k in keys:
        d = np.array([r["delta"][k] for r in rows])
        f = np.array([r["full"][k] for r in rows])
        rel = np.abs(d) / np.maximum(np.abs(f), 1e-9)
        summary[k] = {"mean_abs_delta": float(np.abs(d).mean()),
                      "max_abs_delta": float(np.abs(d).max()),
                      "mean_relative_shift": float(rel.mean()),
                      "max_relative_shift": float(rel.max())}
        print(f"  {k:20s} mean |delta| {np.abs(d).mean():.4f}   "
              f"max |delta| {np.abs(d).max():.4f}   "
              f"mean relative shift {100*rel.mean():.2f}%")

    # Rank stability matters more than absolute shift: the frontier uses these
    # features to order pools, so if the ordering is unchanged the fit is too.
    ranks = {}
    for k in keys:
        a = np.argsort([r["full"][k] for r in rows])
        b = np.argsort([r["val_only"][k] for r in rows])
        ranks[k] = bool(np.array_equal(a, b))
        print(f"  {k:20s} pool ordering preserved: {ranks[k]}")

    out = {
        "note": ("Prediction-space diversity features recomputed on val_selection only "
                 "versus the full eval set (which includes the test slice used for the "
                 "regression target). Measured on the pools whose logits cache is still "
                 "available; the 48 frontier pools cannot be re-fit without regenerating "
                 "their cache from adapter weights."),
        "n_pools_measured": len(rows),
        "summary": summary,
        "pool_ordering_preserved": ranks,
        "pools": rows,
    }
    p = ROOT / "analysis/leakage_check_diversity.json"
    p.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
