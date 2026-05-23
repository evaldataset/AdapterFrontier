#!/usr/bin/env python3
"""Diversity–quality frontier regression for the claim A+C adjudication.

Per paper/prereg.md §9, we fit two regressions:
    Δacc ~ diversity_features + method_indicator
    Δece ~ diversity_features + method_indicator

Target Δ = compute_match output's `accuracy_diff` / `ece_diff` (ensemble vs
baseline; positive = ensemble better). One row per (pool × method × baseline_kind).

Features per row come from the pool's diversity JSON (pool-level, shared by
all methods within that pool) plus a method one-hot.

Validation: leave-one-pool-out CV. The reported R² is global across held-out
folds (not in-sample). Phase 1 gate per prereg §10 uses LOPO-CV R² ≥ 0.5 on
at least one arm × one pool type subset.

Usage:
    python3 analysis/frontier.py --out analysis/frontier_phase1.json
    python3 analysis/frontier.py --baseline-kind n_rank --metrics accuracy ece \
        --out analysis/frontier_n_rank_only.json
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

# Feature columns from diversity JSON
PREDICTION_FEATURES = ("disagreement_rate", "logit_correlation", "ensemble_entropy")
PARAMETER_FEATURES = ("mean_cosine_distance_lora_A", "mean_cosine_distance_lora_B",
                      "mean_frobenius_distance")
HP_ENTROPY_FEATURES = ("seed", "peft_method", "lora_r", "lora_alpha", "lr")
METHOD_ONEHOTS = ("greedy_soup", "soft_vote", "logit_avg", "majority_vote")


def load_diversity(div_dir: Path) -> dict[str, dict]:
    """Load all *_diversity.json in div_dir keyed by pool_name/pool_id."""
    out = {}
    for p in sorted(div_dir.glob("*_diversity.json")):
        try:
            d = json.loads(p.read_text())
        except Exception:
            continue
        key = d.get("pool_name") or p.stem.replace("_diversity", "")
        out[key] = d
    return out


def load_cm_cells(analysis_dir: Path) -> list[dict]:
    """Parse compute_match output filenames to extract (pool, method, baseline_kind, metric).

    File naming convention (set by post-sweep + frontier workflow):
        <pool>_cm_<method>_vs_<baseline_kind>.json           (accuracy)
        <pool>_cm_<method>_vs_<baseline_kind>_ECE.json       (ece)
    """
    pat = re.compile(r"^(?P<pool>.+?)_cm_(?P<method>[a-z_]+?)_vs_(?P<kind>best_of_n|n_rank|n_steps|n_data)(?P<ece>_ECE)?\.json$")
    rows = []
    for p in sorted(analysis_dir.glob("*_cm_*.json")):
        m = pat.match(p.name)
        if not m:
            continue
        try:
            j = json.loads(p.read_text())
        except Exception:
            continue
        metric = "ece" if m.group("ece") else "accuracy"
        diff = j.get("ece_diff") if metric == "ece" else j.get("accuracy_diff")
        if diff is None:
            continue
        rows.append({
            "pool_id": m.group("pool"),
            "method": m.group("method"),
            "baseline_kind": m.group("kind"),
            "metric": metric,
            "diff": float(diff),
            "ci_low": j.get("ci_low"),
            "ci_high": j.get("ci_high"),
            "adjudication": j.get("adjudication"),
            "path": str(p),
        })
    return rows


def row_features(pool_div: dict, method: str) -> dict:
    """Extract one feature vector for a (pool, method) cell."""
    f: dict = {}
    pred = pool_div.get("prediction_space") or {}
    par = pool_div.get("parameter_space") or {}
    hpe = pool_div.get("hyperparameter_entropy") or {}
    for k in PREDICTION_FEATURES:
        f[f"pred_{k}"] = pred.get(k)
    for k in PARAMETER_FEATURES:
        f[f"par_{k}"] = par.get(k)
    # HP entropy: both per-feature and summed
    hp_total = 0.0
    for k in HP_ENTROPY_FEATURES:
        v = hpe.get(k)
        f[f"hp_{k}"] = v
        if isinstance(v, (int, float)):
            hp_total += float(v)
    f["hp_total"] = hp_total
    # Method one-hot (minus one baseline for identifiability)
    for m in METHOD_ONEHOTS:
        f[f"method_{m}"] = 1.0 if m == method else 0.0
    return f


def build_design_matrix(rows: list[dict], diversity: dict[str, dict]) -> tuple[np.ndarray, np.ndarray, list[str], list[dict]]:
    """Returns (X, y, feature_names, used_rows). Drops rows with any missing feature."""
    built = []
    for r in rows:
        div = diversity.get(r["pool_id"])
        if div is None:
            continue
        feats = row_features(div, r["method"])
        if any(v is None for v in feats.values()):
            continue
        built.append((r, feats))
    if not built:
        return np.zeros((0, 0)), np.zeros(0), [], []
    feat_names = list(built[0][1].keys())
    X = np.array([[b[1][k] for k in feat_names] for b in built], dtype=np.float64)
    y = np.array([b[0]["diff"] for b in built], dtype=np.float64)
    return X, y, feat_names, [b[0] for b in built]


def lopo_cv_r2(X: np.ndarray, y: np.ndarray, pools: list[str], ridge: float = 0.1) -> dict:
    """Leave-one-pool-out CV. Small ridge for stability with n ~ p."""
    unique_pools = sorted(set(pools))
    if len(unique_pools) < 2:
        return {"error": f"need >=2 pools for LOPO-CV, got {len(unique_pools)}"}
    preds = np.full_like(y, np.nan)
    for p in unique_pools:
        mask = np.array([pp == p for pp in pools])
        Xtr, ytr = X[~mask], y[~mask]
        Xte = X[mask]
        if len(Xtr) < X.shape[1]:
            continue
        mu = Xtr.mean(axis=0)
        sd = Xtr.std(axis=0)
        sd[sd == 0] = 1.0
        Xtr_s = (Xtr - mu) / sd
        Xte_s = (Xte - mu) / sd
        # Ridge normal equations with intercept term
        Xtr_i = np.column_stack([np.ones(len(Xtr_s)), Xtr_s])
        Xte_i = np.column_stack([np.ones(len(Xte_s)), Xte_s])
        I = np.eye(Xtr_i.shape[1]); I[0, 0] = 0  # don't penalize intercept
        beta = np.linalg.solve(Xtr_i.T @ Xtr_i + ridge * I, Xtr_i.T @ ytr)
        preds[mask] = Xte_i @ beta
    # Global R² on held-out predictions
    ok = ~np.isnan(preds)
    if ok.sum() == 0:
        return {"error": "no LOPO predictions produced"}
    y_ok, p_ok = y[ok], preds[ok]
    rss = float(((y_ok - p_ok) ** 2).sum())
    tss = float(((y_ok - y_ok.mean()) ** 2).sum())
    r2 = 1.0 - rss / tss if tss > 0 else float("nan")
    return {
        "r2_lopo": float(r2),
        "n_rows": int(len(y)),
        "n_pools": len(unique_pools),
        "rss": rss,
        "tss": tss,
        "rmse": float(np.sqrt(((y_ok - p_ok) ** 2).mean())),
        "residuals": (y_ok - p_ok).tolist(),
    }


def in_sample_fit(X: np.ndarray, y: np.ndarray, feat_names: list[str], ridge: float = 0.1) -> dict:
    """In-sample fit to report coefficients + in-sample R² (companion to LOPO)."""
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd == 0] = 1.0
    Xs = (X - mu) / sd
    Xi = np.column_stack([np.ones(len(Xs)), Xs])
    I = np.eye(Xi.shape[1]); I[0, 0] = 0
    beta = np.linalg.solve(Xi.T @ Xi + ridge * I, Xi.T @ y)
    y_hat = Xi @ beta
    rss = float(((y - y_hat) ** 2).sum())
    tss = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - rss / tss if tss > 0 else float("nan")
    return {
        "r2_in_sample": float(r2),
        "intercept": float(beta[0]),
        "coefficients": {feat_names[i]: float(beta[i + 1]) for i in range(len(feat_names))},
        "feature_means": {feat_names[i]: float(mu[i]) for i in range(len(feat_names))},
        "feature_sds": {feat_names[i]: float(sd[i]) for i in range(len(feat_names))},
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--analysis-dir", type=Path, default=Path("analysis"))
    p.add_argument("--metrics", nargs="+", default=["accuracy", "ece"], choices=["accuracy", "ece"])
    p.add_argument("--baseline-kinds", nargs="+", default=["best_of_n", "n_rank"],
                   choices=["best_of_n", "n_rank", "n_steps", "n_data"])
    p.add_argument("--pool-filter", nargs="*", default=None,
                   help="If set, only use pools whose id is in this list")
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--ridge", type=float, default=0.1)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    diversity = load_diversity(args.analysis_dir)
    all_cells = load_cm_cells(args.analysis_dir)
    # Filter out baseline pools themselves as ensemble rows (they only have best_single)
    all_cells = [c for c in all_cells if not c["pool_id"].startswith("baseline_")]
    if args.pool_filter:
        all_cells = [c for c in all_cells if c["pool_id"] in args.pool_filter]

    result = {
        "analysis_dir": str(args.analysis_dir.resolve()),
        "n_diversity_pools": len(diversity),
        "n_total_cells": len(all_cells),
        "by_slice": {},
    }

    for metric in args.metrics:
        for kind in args.baseline_kinds:
            rows = [c for c in all_cells if c["metric"] == metric and c["baseline_kind"] == kind]
            X, y, feat_names, used = build_design_matrix(rows, diversity)
            pools = [r["pool_id"] for r in used]
            slice_key = f"{metric}__vs__{kind}"
            if len(used) < 3:
                result["by_slice"][slice_key] = {
                    "error": f"not enough usable rows: {len(used)}",
                    "n_rows_input": len(rows),
                }
                continue
            lopo = lopo_cv_r2(X, y, pools, ridge=args.ridge)
            ins = in_sample_fit(X, y, feat_names, ridge=args.ridge)
            result["by_slice"][slice_key] = {
                "n_rows": len(used),
                "pools": sorted(set(pools)),
                "methods": sorted(set(r["method"] for r in used)),
                "feature_names": feat_names,
                **lopo,
                **ins,
                "target_mean": float(y.mean()),
                "target_sd": float(y.std()),
                "rows": [
                    {"pool": r["pool_id"], "method": r["method"], "diff": r["diff"],
                     "adjudication": r["adjudication"]}
                    for r in used
                ],
            }

    # Phase 1 gate summary per prereg §10
    gate_arms = []
    for key, sl in result["by_slice"].items():
        if "r2_lopo" in sl:
            gate_arms.append({"slice": key, "r2_lopo": sl["r2_lopo"], "n_rows": sl["n_rows"]})
    result["phase1_gate"] = {
        "rule": "GO if LOPO R² >= 0.5 on any arm; PIVOT if < 0.3 on all; EXTEND otherwise",
        "arms": gate_arms,
        "max_r2_lopo": max((a["r2_lopo"] for a in gate_arms), default=None),
        "decision": (
            "GO" if any(a["r2_lopo"] >= 0.5 for a in gate_arms) else
            "PIVOT" if all(a["r2_lopo"] < 0.3 for a in gate_arms) else
            "EXTEND"
        ) if gate_arms else "NO_DATA",
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    print(f"[frontier] wrote {args.out}")
    print(f"  slices analyzed: {len(result['by_slice'])}")
    for key, sl in result["by_slice"].items():
        if "r2_lopo" in sl:
            print(f"  {key}: n={sl['n_rows']} pools={len(sl['pools'])} R²_LOPO={sl['r2_lopo']:+.3f} R²_IS={sl['r2_in_sample']:+.3f}")
        else:
            print(f"  {key}: {sl.get('error')}")
    print(f"  Phase 1 gate decision: {result['phase1_gate']['decision']} (max R²_LOPO={result['phase1_gate']['max_r2_lopo']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
