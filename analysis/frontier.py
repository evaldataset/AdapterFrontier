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


def row_features(pool_div: dict, method: str, include_method_onehots: bool = True) -> dict:
    """Extract one feature vector for a (pool, method) cell.

    Audit fix Critical 3: pass include_method_onehots=False to drop the
    method-identity features; reviewers can then test whether the headline
    R² survives without that confound.
    """
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
    if include_method_onehots:
        # Method one-hot (minus one baseline for identifiability)
        for m in METHOD_ONEHOTS:
            f[f"method_{m}"] = 1.0 if m == method else 0.0
    return f


def build_design_matrix(rows: list[dict], diversity: dict[str, dict],
                        include_method_onehots: bool = True) -> tuple[np.ndarray, np.ndarray, list[str], list[dict]]:
    """Returns (X, y, feature_names, used_rows). Drops rows with any missing feature.

    Audit fix Critical 3: row-drop reasons are now recorded by
    build_design_matrix_diagnostics(); this function delegates to it and
    returns the same 4-tuple for back-compat.
    """
    diag = build_design_matrix_diagnostics(rows, diversity, include_method_onehots)
    return diag["X"], diag["y"], diag["feature_names"], diag["used_rows"]


def build_design_matrix_diagnostics(rows: list[dict], diversity: dict[str, dict],
                                    include_method_onehots: bool = True) -> dict:
    """Build the design matrix and record exactly which rows were dropped
    and why. Used by frontier.main() to emit an audit trail per slice.

    Audit fix Critical 3: previously rows were silently dropped on
    missing diversity files or missing per-feature values, hiding pool/
    method coverage imbalance from the headline R² claim.
    """
    drop_reasons: dict[str, int] = defaultdict(int)
    dropped_per_pool: dict[str, int] = defaultdict(int)
    dropped_per_method: dict[str, int] = defaultdict(int)
    built = []
    for r in rows:
        div = diversity.get(r["pool_id"])
        if div is None:
            drop_reasons["missing_diversity_file"] += 1
            dropped_per_pool[r["pool_id"]] += 1
            dropped_per_method[r["method"]] += 1
            continue
        feats = row_features(div, r["method"], include_method_onehots=include_method_onehots)
        missing = [k for k, v in feats.items() if v is None]
        if missing:
            drop_reasons[f"missing_feature:{missing[0]}"] += 1
            dropped_per_pool[r["pool_id"]] += 1
            dropped_per_method[r["method"]] += 1
            continue
        built.append((r, feats))
    if not built:
        return {
            "X": np.zeros((0, 0)), "y": np.zeros(0),
            "feature_names": [], "used_rows": [],
            "n_rows_input": len(rows), "n_rows_used": 0,
            "n_rows_dropped": len(rows),
            "drop_reasons": dict(drop_reasons),
            "dropped_per_pool": dict(dropped_per_pool),
            "dropped_per_method": dict(dropped_per_method),
            "include_method_onehots": include_method_onehots,
        }
    feat_names = list(built[0][1].keys())
    X = np.array([[b[1][k] for k in feat_names] for b in built], dtype=np.float64)
    y = np.array([b[0]["diff"] for b in built], dtype=np.float64)
    used_rows = [b[0] for b in built]
    return {
        "X": X, "y": y, "feature_names": feat_names, "used_rows": used_rows,
        "n_rows_input": len(rows), "n_rows_used": len(used_rows),
        "n_rows_dropped": len(rows) - len(used_rows),
        "drop_reasons": dict(drop_reasons),
        "dropped_per_pool": dict(dropped_per_pool),
        "dropped_per_method": dict(dropped_per_method),
        "include_method_onehots": include_method_onehots,
    }


def lopo_cv_r2(X: np.ndarray, y: np.ndarray, pools: list[str], ridge: float = 0.1) -> dict:
    """Leave-one-pool-out CV. Small ridge for stability with n ~ p.

    Audit fix Critical 3: skipped folds (too few training rows) are now
    counted and listed instead of silently dropped.
    """
    unique_pools = sorted(set(pools))
    if len(unique_pools) < 2:
        return {"error": f"need >=2 pools for LOPO-CV, got {len(unique_pools)}"}
    preds = np.full_like(y, np.nan)
    skipped_folds: list[dict] = []
    used_folds: list[str] = []
    for p in unique_pools:
        mask = np.array([pp == p for pp in pools])
        Xtr, ytr = X[~mask], y[~mask]
        Xte = X[mask]
        if len(Xtr) < X.shape[1]:
            skipped_folds.append({
                "pool": p,
                "reason": f"n_train_rows ({len(Xtr)}) < n_features ({X.shape[1]})",
                "n_test_rows": int(Xte.shape[0]),
            })
            continue
        used_folds.append(p)
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
        return {"error": "no LOPO predictions produced",
                "n_skipped_folds": len(skipped_folds),
                "skipped_folds": skipped_folds}
    y_ok, p_ok = y[ok], preds[ok]
    rss = float(((y_ok - p_ok) ** 2).sum())
    tss = float(((y_ok - y_ok.mean()) ** 2).sum())
    r2 = 1.0 - rss / tss if tss > 0 else float("nan")
    return {
        "r2_lopo": float(r2),
        "n_rows": int(len(y)),
        "n_pools": len(unique_pools),
        "n_used_folds": len(used_folds),
        "n_skipped_folds": len(skipped_folds),
        "skipped_folds": skipped_folds,
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
            slice_key = f"{metric}__vs__{kind}"

            # Headline fit (with method one-hots).
            diag = build_design_matrix_diagnostics(rows, diversity, include_method_onehots=True)
            X, y, feat_names, used = diag["X"], diag["y"], diag["feature_names"], diag["used_rows"]
            pools = [r["pool_id"] for r in used]
            if len(used) < 3:
                result["by_slice"][slice_key] = {
                    "error": f"not enough usable rows: {len(used)}",
                    "n_rows_input": len(rows),
                    "drop_diagnostics": {
                        k: diag[k] for k in
                        ("n_rows_input", "n_rows_used", "n_rows_dropped",
                         "drop_reasons", "dropped_per_pool", "dropped_per_method")
                    },
                }
                continue
            lopo = lopo_cv_r2(X, y, pools, ridge=args.ridge)
            ins = in_sample_fit(X, y, feat_names, ridge=args.ridge)

            # Audit fix Critical 3: ablation without method one-hots so the
            # paper can show headline R² is not an artifact of method-identity
            # features.
            diag_noml = build_design_matrix_diagnostics(rows, diversity, include_method_onehots=False)
            ablation_block: dict
            if diag_noml["n_rows_used"] >= 3:
                lopo_noml = lopo_cv_r2(diag_noml["X"], diag_noml["y"],
                                       [r["pool_id"] for r in diag_noml["used_rows"]],
                                       ridge=args.ridge)
                ablation_block = {
                    "r2_lopo": lopo_noml.get("r2_lopo"),
                    "r2_lopo_delta_vs_headline": (
                        (lopo_noml.get("r2_lopo", 0.0) - lopo.get("r2_lopo", 0.0))
                        if lopo.get("r2_lopo") is not None and lopo_noml.get("r2_lopo") is not None
                        else None
                    ),
                    "n_features": int(diag_noml["X"].shape[1]),
                    "n_used_folds": lopo_noml.get("n_used_folds"),
                    "n_skipped_folds": lopo_noml.get("n_skipped_folds"),
                }
            else:
                ablation_block = {"error": f"ablation insufficient rows: {diag_noml['n_rows_used']}"}

            result["by_slice"][slice_key] = {
                "n_rows": len(used),
                "pools": sorted(set(pools)),
                "methods": sorted(set(r["method"] for r in used)),
                "feature_names": feat_names,
                **lopo,
                **ins,
                "target_mean": float(y.mean()),
                "target_sd": float(y.std()),
                "drop_diagnostics": {
                    "n_rows_input": diag["n_rows_input"],
                    "n_rows_used": diag["n_rows_used"],
                    "n_rows_dropped": diag["n_rows_dropped"],
                    "drop_reasons": diag["drop_reasons"],
                    "dropped_per_pool": diag["dropped_per_pool"],
                    "dropped_per_method": diag["dropped_per_method"],
                },
                "ablation_no_method_onehots": ablation_block,
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
            dd = sl.get("drop_diagnostics", {})
            ab = sl.get("ablation_no_method_onehots", {}) or {}
            ab_r2 = ab.get("r2_lopo")
            ab_str = f" no-method-1hot R²={ab_r2:+.3f}" if isinstance(ab_r2, (int, float)) else ""
            print(f"  {key}: n_in={dd.get('n_rows_input','?')} n_used={dd.get('n_rows_used','?')} "
                  f"n_dropped={dd.get('n_rows_dropped','?')} pools={len(sl['pools'])} "
                  f"R²_LOPO={sl['r2_lopo']:+.3f} R²_IS={sl['r2_in_sample']:+.3f}{ab_str}")
        else:
            print(f"  {key}: {sl.get('error')}")
    print(f"  Phase 1 gate decision: {result['phase1_gate']['decision']} (max R²_LOPO={result['phase1_gate']['max_r2_lopo']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
