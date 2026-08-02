#!/usr/bin/env python3
"""Information-theoretic mechanism analysis: why are decoders ECE-easy?

For each pool's best_single method, compute on the test split:
  - H(c | conf)   : conditional entropy of correctness given confidence
                    (low → confidence cleanly separates right vs wrong)
  - I(c ; conf)   : mutual info between correctness and confidence
                    (high → confidence is informative about correctness)
  - H(p_max)      : entropy of the max-prob distribution across examples
                    (sharper / more bimodal = lower)
  - p10/p50/p90 confidence percentiles
  - per-confidence-bin accuracy (the reliability diagram bins)

Hypothesis: decoder pools have lower H(c|conf) than encoder pools — they
are sharply right or sharply wrong, so soft averaging cleanly extracts
the calibration signal. Encoder pools have higher H(c|conf) → mixing
signals. This explains why decoder ensembles win on ECE consistently.

Usage:
    python3 analysis/info_mechanism.py --out analysis/info_mechanism.json
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def family(pool_id: str) -> str:
    s = pool_id.lower()
    if 'qwen' in s or 'llama' in s or 'pythia' in s or 'smollm' in s: return 'decoder'
    if 'deberta' in s: return 'encoder_deberta'
    return 'encoder_bertfam'


def conditional_H_corr_given_conf(conf: np.ndarray, correct: np.ndarray, n_bins: int = 10) -> float:
    """H(c | conf) using equal-mass binning. Lower = confidence cleanly separates."""
    n = len(conf)
    order = np.argsort(conf)
    bins = np.array_split(order, n_bins)
    H_total = 0.0
    for b in bins:
        if len(b) == 0:
            continue
        p = float(correct[b].mean())
        # binary entropy
        h = 0.0
        for x in (p, 1.0 - p):
            if x > 0:
                h -= x * np.log2(x)
        H_total += (len(b) / n) * h
    return float(H_total)


def mutual_info_corr_conf(conf: np.ndarray, correct: np.ndarray, n_bins: int = 10) -> float:
    """I(c; conf) = H(c) - H(c|conf)"""
    p_corr = float(correct.mean())
    Hc = 0.0
    for x in (p_corr, 1.0 - p_corr):
        if x > 0:
            Hc -= x * np.log2(x)
    return Hc - conditional_H_corr_given_conf(conf, correct, n_bins)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    by_family = defaultdict(list)
    rows = []
    for p in sorted(Path("ensemble_results").glob("pool_*.json")):
        if "_OOD_" in p.name:
            continue
        try:
            ens = json.loads(p.read_text())
        except Exception:
            continue
        pool_id = ens.get("pool_id") or ens.get("pool_name") or p.stem
        labels = np.asarray(ens.get("labels_test", []), dtype=np.int64)
        if labels.size == 0:
            continue
        # Use best_single method's confidence + predictions
        m = (ens.get("methods") or {}).get("best_single", {})
        conf = m.get("confidence_test")
        preds = m.get("predictions_test")
        if not conf or not preds:
            continue
        conf = np.asarray(conf, dtype=np.float64)
        preds = np.asarray(preds, dtype=np.int64)
        if conf.size != labels.size:
            continue
        correct = (preds == labels).astype(np.float64)

        H_cgc = conditional_H_corr_given_conf(conf, correct, n_bins=10)
        I_cc = mutual_info_corr_conf(conf, correct, n_bins=10)
        # entropy of confidence bin distribution = balance of confidence
        order = np.argsort(conf)
        bins = np.array_split(order, 10)
        bin_accs = [float(correct[b].mean()) if len(b) else 0.0 for b in bins]
        bin_means = [float(conf[b].mean()) if len(b) else 0.0 for b in bins]

        row = {
            "pool_id": pool_id,
            "family": family(pool_id),
            "n": int(conf.size),
            "p_correct": float(correct.mean()),
            "p10_conf": float(np.quantile(conf, 0.1)),
            "p50_conf": float(np.quantile(conf, 0.5)),
            "p90_conf": float(np.quantile(conf, 0.9)),
            "H_corr_given_conf_bits": H_cgc,
            "I_corr_conf_bits": I_cc,
            "bin_accs": bin_accs,
            "bin_mean_confs": bin_means,
        }
        rows.append(row)
        by_family[family(pool_id)].append(row)

    print(f"Loaded {len(rows)} pools")
    print(f"{'family':<22s} {'n_pools':>7s} {'mean_H(c|conf)':>15s} {'mean_I(c;conf)':>15s} {'mean_p50_conf':>14s}")
    print('=' * 78)
    family_summary = {}
    for fam, items in by_family.items():
        Hcgc_mean = float(np.mean([r["H_corr_given_conf_bits"] for r in items]))
        Icc_mean  = float(np.mean([r["I_corr_conf_bits"] for r in items]))
        p50_mean  = float(np.mean([r["p50_conf"] for r in items]))
        family_summary[fam] = {
            "n_pools": len(items),
            "mean_H_corr_given_conf_bits": Hcgc_mean,
            "mean_I_corr_conf_bits": Icc_mean,
            "mean_p50_conf": p50_mean,
        }
        print(f"{fam:<22s} {len(items):>7d} {Hcgc_mean:>15.4f} {Icc_mean:>15.4f} {p50_mean:>14.4f}")

    print()
    print("=== top 5 lowest H(c|conf) — best confidence-correctness separation ===")
    for r in sorted(rows, key=lambda r: r["H_corr_given_conf_bits"])[:5]:
        print(f"  {r['pool_id']:<48s} fam={r['family']:<14s} H(c|conf)={r['H_corr_given_conf_bits']:.4f} I={r['I_corr_conf_bits']:.4f}")
    print()
    print("=== top 5 highest H(c|conf) — confidence cannot separate correctness ===")
    for r in sorted(rows, key=lambda r: -r["H_corr_given_conf_bits"])[:5]:
        print(f"  {r['pool_id']:<48s} fam={r['family']:<14s} H(c|conf)={r['H_corr_given_conf_bits']:.4f} I={r['I_corr_conf_bits']:.4f}")

    out = {"family_summary": family_summary, "per_pool": rows}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
