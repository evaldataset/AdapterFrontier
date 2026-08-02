#!/usr/bin/env python3
"""Per-feature and per-group ablation of the frontier regression.

For each (metric, baseline_kind) slice that passed Phase 2 gate (vs n_rank),
fit the LOPO-CV regression with each feature/group held out and report the
drop in R²_LOPO. Features with the largest drops carry the predictive
signal — a Phase 3 mechanism finding that informs which diversity
properties to emphasize in the paper.

Usage:
    python3 analysis/feature_ablation.py --out analysis/feature_ablation.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from frontier import (
    load_diversity, load_cm_cells, build_design_matrix,
    lopo_cv_r2, in_sample_fit,
)

FEATURE_GROUPS = {
    "prediction_space": ["pred_disagreement_rate", "pred_logit_correlation", "pred_ensemble_entropy"],
    "parameter_space":  ["par_mean_cosine_distance_lora_A", "par_mean_cosine_distance_lora_B",
                         "par_mean_frobenius_distance"],
    "hp_entropy":       ["hp_seed", "hp_peft_method", "hp_lora_r", "hp_lora_alpha", "hp_lr", "hp_total"],
    "method_indicator": ["method_greedy_soup", "method_soft_vote", "method_logit_avg", "method_majority_vote"],
}


def fit_with_subset(X: np.ndarray, y: np.ndarray, pools: list[str], feat_names: list[str],
                    keep_indices: list[int]) -> dict:
    Xsub = X[:, keep_indices]
    sub_names = [feat_names[i] for i in keep_indices]
    if Xsub.shape[1] == 0:
        # constant-only; LOPO becomes mean prediction → R² = 0
        return {"r2_lopo": 0.0, "r2_in_sample": 0.0, "n_features": 0}
    lopo = lopo_cv_r2(Xsub, y, pools)
    ins = in_sample_fit(Xsub, y, sub_names)
    return {
        "r2_lopo": lopo.get("r2_lopo", float("nan")),
        "r2_in_sample": ins["r2_in_sample"],
        "n_features": Xsub.shape[1],
    }


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--analysis-dir", type=Path, default=Path("analysis"))
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--slices", nargs="+",
                   default=["accuracy__vs__n_rank", "ece__vs__n_rank"],
                   help="Which (metric__vs__baseline_kind) slices to analyze")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    diversity = load_diversity(args.analysis_dir)
    all_cells = load_cm_cells(args.analysis_dir)
    all_cells = [c for c in all_cells if not c["pool_id"].startswith("baseline_")]

    out: dict = {"slices": {}}
    for slice_key in args.slices:
        metric, kind = slice_key.split("__vs__")
        rows = [c for c in all_cells if c["metric"] == metric and c["baseline_kind"] == kind]
        X, y, feat_names, used = build_design_matrix(rows, diversity)
        pools = [r["pool_id"] for r in used]
        if len(used) < 5:
            out["slices"][slice_key] = {"error": f"only {len(used)} rows"}
            continue

        baseline = fit_with_subset(X, y, pools, feat_names, list(range(len(feat_names))))

        # Group ablations
        group_results = {}
        for gname, gcols in FEATURE_GROUPS.items():
            keep = [i for i, f in enumerate(feat_names) if f not in gcols]
            r = fit_with_subset(X, y, pools, feat_names, keep)
            r["dropped"] = [f for f in gcols if f in feat_names]
            r["r2_lopo_drop"] = baseline["r2_lopo"] - r["r2_lopo"]
            group_results[gname] = r

        # Per-feature ablations
        feat_results = {}
        for i, f in enumerate(feat_names):
            keep = [j for j in range(len(feat_names)) if j != i]
            r = fit_with_subset(X, y, pools, feat_names, keep)
            r["r2_lopo_drop"] = baseline["r2_lopo"] - r["r2_lopo"]
            feat_results[f] = r

        # Group-only models (use ONLY this group)
        group_only = {}
        for gname, gcols in FEATURE_GROUPS.items():
            keep = [i for i, f in enumerate(feat_names) if f in gcols]
            r = fit_with_subset(X, y, pools, feat_names, keep)
            group_only[gname] = r

        out["slices"][slice_key] = {
            "n_rows": len(used),
            "n_pools": len(set(pools)),
            "feat_names": feat_names,
            "baseline_full": baseline,
            "group_holdout": group_results,
            "group_only":    group_only,
            "feature_holdout": feat_results,
        }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2, default=float))
    print(f"[feature_ablation] wrote {args.out}")
    for slice_key, sl in out["slices"].items():
        if "error" in sl:
            print(f"  {slice_key}: {sl['error']}")
            continue
        print(f"\n=== {slice_key} (n={sl['n_rows']}, {sl['n_pools']} pools) ===")
        print(f"  full model R²_LOPO = {sl['baseline_full']['r2_lopo']:+.4f}")
        print("  --- holdout group (drop = baseline - holdout) ---")
        for g, r in sorted(sl["group_holdout"].items(), key=lambda kv: -kv[1]["r2_lopo_drop"]):
            print(f"    drop {g:>20s} → R²_LOPO {r['r2_lopo']:+.4f}  (Δ={r['r2_lopo_drop']:+.4f})")
        print("  --- only group (model with this group ALONE) ---")
        for g, r in sorted(sl["group_only"].items(), key=lambda kv: -kv[1]["r2_lopo"]):
            print(f"    only {g:>20s} → R²_LOPO {r['r2_lopo']:+.4f}")
        print("  --- top per-feature drops (most important features) ---")
        top = sorted(sl["feature_holdout"].items(), key=lambda kv: -kv[1]["r2_lopo_drop"])[:5]
        for f, r in top:
            print(f"    drop {f:>30s} → R²_LOPO {r['r2_lopo']:+.4f}  (Δ={r['r2_lopo_drop']:+.4f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
