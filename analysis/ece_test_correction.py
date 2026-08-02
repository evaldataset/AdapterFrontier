#!/usr/bin/env python3
"""The ECE arm was adjudicated with a permutation test of the wrong null.

`compute_match._sign_flip_pvalue_ece` swaps the (confidence, correctness) pair
between arms example by example. That tests H0: the two systems share one
confidence-correctness joint distribution -- which ensembling violates by
construction, since averaging probabilities reshapes the confidence
distribution whether or not ECE moves. The symptom is visible in the corpus:
the ECE arm's median p is ~1e-3 with half the cells pinned at the Monte-Carlo
floor, while the accuracy arm -- where the sign-flip on per-example
correctness *is* well specified -- has a median p of ~0.12.

This script re-adjudicates the calibration arm against the null we actually
mean, H0: Delta ECE = 0, by inverting the paired bootstrap that
`_paired_ece_diff_ci` already computes correctly (it recomputes ECE on each
resampled index set for both arms rather than linearising). The achieved
significance level is the usual two-sided bootstrap ASL,
    p = 2 * min( Pr[boot <= 0], Pr[boot >= 0] ),
floored at 1/(B+1).

It also reports, on the same cells and from the same cached per-example
arrays, three things the paper claims but never adjudicates:

  (a) Brier and NLL. ECE is the pre-registered primary, but the proper scores
      are pre-registered alongside it and were collected. Top-label Brier is
      computable per example from (conf, correct) and so gets a paired
      bootstrap; the stored multiclass Brier is reported as a point contrast.
  (b) The per-method decomposition, because `majority_vote`'s confidence is a
      vote fraction, not a probability estimate, and equal-mass binning is not
      well defined on it.
  (c) Counts under the *pre-registered* within-table Holm (prereg S7: "Holm
      step-down across every cell reported in a single table"), alongside BH
      run per arm rather than pooled across arms.

Usage: python3 analysis/ece_test_correction.py
"""
from __future__ import annotations

import json
import re
import zlib
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
import sys
sys.path.insert(0, str(ROOT / "analysis"))
from compute_match import _ece_equal_mass, holm_correct  # noqa: E402

DECODER = ("qwen", "llama", "mistral", "tinyllama", "pythia", "smollm")
METHODS = ("soft_vote", "logit_avg", "greedy_soup", "majority_vote")
POOL_PREFIX = re.compile(r"^pool_[abc]_")
POOL_SUFFIX = re.compile(r"_(method_mixed|hp_diverse|seed_only)$")
B = 5000


def is_clean(pool_id: str) -> bool:
    pl = pool_id.lower()
    return not ("hellaswag" in pl or "gsm8k" in pl
                or any(t in pl for t in DECODER))


def baseline_for(pool_id: str) -> Path:
    stem = POOL_SUFFIX.sub("", POOL_PREFIX.sub("", pool_id))
    return ROOT / f"ensemble_results/baseline_compute_matched_{stem}.json"


def arrays(block: dict, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    conf = np.asarray(block["confidence_test"], dtype=np.float64)
    pred = np.asarray(block["predictions_test"])
    return conf, (pred == labels).astype(np.float64)


def boot_ece(conf_a, corr_a, conf_b, corr_b, seed: int):
    """Returns (obs, lo, hi, p). Positive obs = ensemble (b) better calibrated."""
    rng = np.random.RandomState(seed)
    n = len(conf_a)
    obs = _ece_equal_mass(conf_a, corr_a) - _ece_equal_mass(conf_b, corr_b)
    boots = np.empty(B)
    for i in range(B):
        idx = rng.randint(0, n, size=n)
        boots[i] = (_ece_equal_mass(conf_a[idx], corr_a[idx])
                    - _ece_equal_mass(conf_b[idx], corr_b[idx]))
    lo, hi = np.quantile(boots, [0.025, 0.975])
    p = 2 * min((boots <= 0).mean(), (boots >= 0).mean())
    return float(obs), float(lo), float(hi), float(max(p, 1.0 / (B + 1)))


def boot_mean(d: np.ndarray, seed: int):
    """Paired bootstrap on a per-example difference. Positive = ensemble better."""
    rng = np.random.RandomState(seed)
    n = len(d)
    boots = d[rng.randint(0, n, size=(B, n))].mean(axis=1)
    lo, hi = np.quantile(boots, [0.025, 0.975])
    p = 2 * min((boots <= 0).mean(), (boots >= 0).mean())
    return float(d.mean()), float(lo), float(hi), float(max(p, 1.0 / (B + 1)))


def adjudicate(lo: float, hi: float) -> str:
    if lo > 0:
        return "SUPPORTED"
    if hi < 0:
        return "REVERSED"
    return "UNSUPPORTED"


def main() -> int:
    cells = []
    for pf in sorted((ROOT / "ensemble_results").glob("pool_*.json")):
        pool = json.loads(pf.read_text())
        pid = pool.get("pool_id") or pf.stem
        if not is_clean(pid):
            continue
        bf = baseline_for(pid)
        if not bf.exists():
            continue
        base = json.loads(bf.read_text())
        if base["n_test"] != pool["n_test"]:
            print(f"  !! n_test mismatch, skipping {pid}")
            continue
        labels = np.asarray(pool["labels_test"])
        bconf, bcorr = arrays(base["methods"]["best_single"], labels)
        b_cal = base["methods"]["best_single"]["calibration"]
        for m in METHODS:
            if m not in pool["methods"]:
                continue
            econf, ecorr = arrays(pool["methods"][m], labels)
            # zlib.crc32, not hash(): str hashing is salted per process, so
            # hash() would reseed every bootstrap on every run and the counts
            # would not reproduce.
            seed = zlib.crc32((pid + "|" + m).encode()) % (2**31)
            obs, lo, hi, p = boot_ece(bconf, bcorr, econf, ecorr, seed)
            # top-label Brier: per-example (conf - correct)^2, lower is better
            db = ((bconf - bcorr) ** 2) - ((econf - ecorr) ** 2)
            bobs, blo, bhi, bp = boot_mean(db, seed + 1)
            e_cal = pool["methods"][m]["calibration"]
            cells.append({
                "pool_id": pid, "method": m, "n_test": int(pool["n_test"]),
                "ece_diff": obs, "ece_ci": [lo, hi], "ece_p_bootstrap": p,
                "ece_adjudication_raw": adjudicate(lo, hi),
                "toplabel_brier_diff": bobs, "toplabel_brier_ci": [blo, bhi],
                "toplabel_brier_p": bp,
                "toplabel_brier_adjudication_raw": adjudicate(blo, bhi),
                "multiclass_brier_diff": b_cal["brier"] - e_cal["brier"],
                "nll_diff": b_cal["nll"] - e_cal["nll"],
            })
        print(f"  {pid[:52]:52s} vs {bf.stem[28:]:22s} ok")

    n = len(cells)
    print(f"\nclean encoder x n_rank calibration cells: {n} "
          f"from {len(set(c['pool_id'] for c in cells))} pools\n")

    # ---- corrected test vs the shipped one
    shipped = {}
    for f in (ROOT / "analysis").glob("*_cm_*_vs_n_rank_ECE.json"):
        d = json.loads(f.read_text())
        mm = re.match(r"^(?P<pool>.+?)_cm_(?P<m>[a-z_]+?)_vs_n_rank_ECE\.json$", f.name)
        if mm:
            shipped[(mm.group("pool"), mm.group("m"))] = d
    sp = [shipped[k]["sign_flip_p"] for k in
          ((c["pool_id"], c["method"]) for c in cells) if k in shipped]
    cp = [c["ece_p_bootstrap"] for c in cells]
    print("(1) the test")
    if sp:
        print(f"  shipped sign-flip p    median {np.median(sp):.4f}   "
              f"at MC floor: {sum(x <= 2.001e-4 for x in sp)}/{len(sp)}")
    print(f"  bootstrap-inverted p   median {np.median(cp):.4f}   "
          f"at floor: {sum(x <= 1.0/(B+1)+1e-12 for x in cp)}/{len(cp)}")

    # ---- adjudication under three procedures
    print("\n(2) ECE adjudication, corrected p-values")
    raw_s = sum(c["ece_adjudication_raw"] == "SUPPORTED" for c in cells)
    raw_r = sum(c["ece_adjudication_raw"] == "REVERSED" for c in cells)
    print(f"  raw CI                        SUPPORTED {raw_s:3d}   REVERSED {raw_r:3d}")

    holm = holm_correct([c["ece_p_bootstrap"] for c in cells])
    hs = sum(h < 0.05 and c["ece_adjudication_raw"] == "SUPPORTED"
             for h, c in zip(holm, cells))
    hr = sum(h < 0.05 and c["ece_adjudication_raw"] == "REVERSED"
             for h, c in zip(holm, cells))
    for h, c in zip(holm, cells):
        c["holm_adjusted_p_within_table"] = float(h)
    print(f"  Holm within-table (m={n}, prereg)  SUPPORTED {hs:3d}   REVERSED {hr:3d}")

    order = np.argsort(cp)
    bh_cut = 0.0
    for rank, i in enumerate(order, start=1):
        if cp[i] <= 0.05 * rank / n:
            bh_cut = cp[i]
    bs = sum(c["ece_p_bootstrap"] <= bh_cut and c["ece_adjudication_raw"] == "SUPPORTED"
             for c in cells)
    br = sum(c["ece_p_bootstrap"] <= bh_cut and c["ece_adjudication_raw"] == "REVERSED"
             for c in cells)
    print(f"  BH within-arm (q=.05, cut={bh_cut:.4f})  SUPPORTED {bs:3d}   REVERSED {br:3d}")

    # ---- per method
    print("\n(3) per combination rule (raw CI)")
    by = defaultdict(lambda: [0, 0, 0, [], [], []])
    for c in cells:
        a = by[c["method"]]
        a[0] += c["ece_adjudication_raw"] == "SUPPORTED"
        a[1] += c["ece_adjudication_raw"] == "REVERSED"
        a[2] += 1
        a[3].append(c["ece_diff"])
        a[4].append(c["multiclass_brier_diff"])
        a[5].append(c["toplabel_brier_adjudication_raw"] == "SUPPORTED")
    print(f"  {'rule':14s} {'ECE S/R':>10s} {'meanDECE':>10s} "
          f"{'mcBrier>0':>10s} {'meanDBrier':>11s} {'tlBrier S':>10s}")
    for m, a in sorted(by.items()):
        print(f"  {m:14s} {a[0]:4d}/{a[1]:<5d} {np.mean(a[3]):+10.4f} "
              f"{sum(x > 0 for x in a[4]):5d}/{a[2]:<4d} {np.mean(a[4]):+11.4f} "
              f"{sum(a[5]):5d}/{a[2]:<4d}")

    # ---- proper scores overall
    mcb = np.array([c["multiclass_brier_diff"] for c in cells])
    nll = np.array([c["nll_diff"] for c in cells])
    tlb_s = sum(c["toplabel_brier_adjudication_raw"] == "SUPPORTED" for c in cells)
    tlb_r = sum(c["toplabel_brier_adjudication_raw"] == "REVERSED" for c in cells)
    noMV = [c for c in cells if c["method"] != "majority_vote"]
    print("\n(4) proper scores on the same cells (positive = ensemble better)")
    print(f"  multiclass Brier improved {int((mcb > 0).sum())}/{n}   mean {mcb.mean():+.4f}")
    print(f"  NLL improved              {int((nll > 0).sum())}/{n}   mean {nll.mean():+.4f}")
    print(f"  top-label Brier adjudicated  SUPPORTED {tlb_s}   REVERSED {tlb_r}")
    print(f"  excluding majority_vote: mcBrier improved "
          f"{sum(c['multiclass_brier_diff'] > 0 for c in noMV)}/{len(noMV)}   "
          f"ECE SUPPORTED {sum(c['ece_adjudication_raw'] == 'SUPPORTED' for c in noMV)}"
          f"/{len(noMV)}")

    # ---- pool level (the honest inferential unit, L1d)
    pools = defaultdict(list)
    for c in noMV:
        pools[c["pool_id"]].append(c["ece_adjudication_raw"] == "SUPPORTED")
    allsup = sum(all(v) for v in pools.values())
    print(f"\n(5) pool level, probability-averaging rules only: "
          f"{allsup}/{len(pools)} pools SUPPORTED on every rule")

    out = {
        "note": ("Calibration arm re-adjudicated against H0: Delta ECE = 0 by inverting "
                 "the paired ECE bootstrap, replacing the joint-distribution permutation "
                 "test in compute_match._sign_flip_pvalue_ece, which ensembling violates "
                 "by construction. Proper scores (Brier, NLL) reported on the same cells."),
        "n_cells": n, "n_pools": len(set(c["pool_id"] for c in cells)), "n_boot": B,
        "test_comparison": {
            "shipped_sign_flip_p_median": float(np.median(sp)) if sp else None,
            "shipped_at_mc_floor": int(sum(x <= 2.001e-4 for x in sp)) if sp else None,
            "bootstrap_p_median": float(np.median(cp)),
        },
        "ece_adjudication": {
            "raw_ci": {"supported": raw_s, "reversed": raw_r},
            "holm_within_table_prereg": {"supported": hs, "reversed": hr, "m": n},
            "bh_within_arm": {"supported": bs, "reversed": br, "cutoff": bh_cut},
        },
        "proper_scores": {
            "multiclass_brier_improved": int((mcb > 0).sum()),
            "multiclass_brier_mean_diff": float(mcb.mean()),
            "nll_improved": int((nll > 0).sum()),
            "nll_mean_diff": float(nll.mean()),
            "toplabel_brier_supported": tlb_s, "toplabel_brier_reversed": tlb_r,
        },
        "per_method": {
            m: {
                "n_cells": a[2],
                "ece_supported": a[0], "ece_reversed": a[1],
                "mean_ece_diff": float(np.mean(a[3])),
                "multiclass_brier_improved": int(sum(x > 0 for x in a[4])),
                "mean_multiclass_brier_diff": float(np.mean(a[4])),
            } for m, a in sorted(by.items())
        },
        "ece_reversals_from_majority_vote": sum(
            c["method"] == "majority_vote" and c["ece_adjudication_raw"] == "REVERSED"
            for c in cells),
        "excluding_majority_vote": {
            "n_cells": len(noMV),
            "ece_supported": sum(c["ece_adjudication_raw"] == "SUPPORTED" for c in noMV),
            "multiclass_brier_improved": sum(c["multiclass_brier_diff"] > 0 for c in noMV),
            "pools_supported_on_every_rule": allsup, "n_pools": len(pools),
        },
        "cells": cells,
    }
    p = ROOT / "analysis/ece_test_correction.json"
    p.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
