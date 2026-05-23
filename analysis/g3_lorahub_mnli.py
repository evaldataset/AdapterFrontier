#!/usr/bin/env python3
"""G.3: LoRAHub-style learned-weight ensemble on Pool-A MNLI pools (extends paper §3).

Uses the same L-BFGS-B optimizer that LoRAHub's library uses (alongside their
Nevergrad option) on the val_combine split, then evaluates on test. Compares
to greedy_soup / soft_vote / best_single from the existing ensemble_results.

Output: writes `methods.learned_weight` into the ensemble JSON, and prints
a one-row summary suitable for a paper table.
"""
from __future__ import annotations
import argparse, glob, json
from pathlib import Path
import numpy as np
from scipy.optimize import minimize


def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x); return e / e.sum(axis=axis, keepdims=True)


def neg_log_lik(alpha, P_val, y_val):
    w = np.maximum(alpha, 0); w = w / max(w.sum(), 1e-12)
    p = (w[:, None, None] * P_val).sum(axis=0)
    p = np.clip(p, 1e-12, 1.0)
    return float(-np.log(p[np.arange(len(y_val)), y_val] + 1e-12).mean())


def ece_15bin_equalmass(conf, correct):
    order = np.argsort(conf)
    bins = np.array_split(order, 15)
    e = 0.0
    for b in bins:
        if len(b) == 0: continue
        e += (len(b) / len(conf)) * abs(conf[b].mean() - correct[b].mean())
    return e


def run_pool(ensemble_path: Path, pool_path: Path, cache_dir: Path) -> dict:
    e = json.loads(ensemble_path.read_text())
    pool = json.loads(pool_path.read_text())
    pool_id = pool["pool_id"]

    splits = e.get("splits", {})
    val_idx = np.array(splits.get("val_combine", []), dtype=np.int64)
    test_idx = np.array(splits.get("test", []), dtype=np.int64)
    if len(val_idx) == 0 or len(test_idx) == 0:
        return {"pool_id": pool_id, "error": "splits missing"}

    labels_full = np.array(e.get("labels_full", e.get("labels_test", [])), dtype=np.int64)
    if len(labels_full) == 0:
        return {"pool_id": pool_id, "error": "labels_full / labels_test missing"}
    y_val = labels_full[val_idx] if len(labels_full) >= val_idx.max()+1 else None
    y_test = np.array(e["labels_test"], dtype=np.int64)
    if y_val is None:
        return {"pool_id": pool_id, "error": "val labels unavailable"}

    # Find per-adapter logits in cache
    L = []
    matched_ids = []
    for a in pool["adapters"]:
        candidates = list(cache_dir.glob(f"{pool_id}__*{a['adapter_id']}*.npy"))
        if not candidates:
            continue
        L.append(np.load(candidates[0]))
        matched_ids.append(a["adapter_id"])
    if len(L) < 2:
        return {"pool_id": pool_id, "error": f"only {len(L)} cached adapters"}
    L = np.stack(L, axis=0)  # (M, N_total, K)
    M, N_total, K = L.shape
    P = softmax(L, axis=-1)
    P_val = P[:, val_idx]
    P_test = P[:, test_idx]

    # L-BFGS-B (LoRAHub-compatible algorithm)
    alpha0 = np.ones(M) / M
    res = minimize(neg_log_lik, alpha0, args=(P_val, y_val),
                   method="L-BFGS-B", bounds=[(0, None)] * M,
                   options={"maxiter": 300, "ftol": 1e-8})
    w = np.maximum(res.x, 0); w = w / max(w.sum(), 1e-12)
    eff_M = int((w > 0.01).sum())

    p_test = (w[:, None, None] * P_test).sum(axis=0)
    pred = p_test.argmax(axis=1)
    conf = p_test.max(axis=1)
    correct = (pred == y_test).astype(np.float64)
    acc = float(correct.mean())
    ece = ece_15bin_equalmass(conf, correct)

    methods = e["methods"]
    bs_acc = methods.get("best_single", {}).get("accuracy_test")
    sv_acc = methods.get("soft_vote", {}).get("accuracy_test")
    gs_acc = methods.get("greedy_soup", {}).get("accuracy_test")
    bs_ece = (methods.get("best_single", {}).get("calibration") or {}).get("ece_15bin_equalmass")
    sv_ece = (methods.get("soft_vote", {}).get("calibration") or {}).get("ece_15bin_equalmass")
    gs_ece = (methods.get("greedy_soup", {}).get("calibration") or {}).get("ece_15bin_equalmass")

    result = {
        "pool_id": pool_id,
        "M": M, "effective_M": eff_M,
        "learned_weight_acc": acc, "learned_weight_ece": ece,
        "best_single_acc": bs_acc, "best_single_ece": bs_ece,
        "soft_vote_acc": sv_acc, "soft_vote_ece": sv_ece,
        "greedy_soup_acc": gs_acc, "greedy_soup_ece": gs_ece,
        "delta_vs_best_single_acc": (acc - bs_acc) if bs_acc is not None else None,
        "delta_vs_greedy_soup_acc": (acc - gs_acc) if gs_acc is not None else None,
        "val_combine_nll": float(res.fun),
        "optimizer": "L-BFGS-B (LoRAHub-compatible)",
    }
    # Persist
    e["methods"]["learned_weight_lorahub"] = {
        "accuracy_test": acc, "calibration": {"ece_15bin_equalmass": ece},
        "predictions_test": pred.tolist(), "confidence_test": conf.tolist(),
        "weights": w.tolist(), "effective_M": eff_M,
        "optimizer": "L-BFGS-B", "val_combine_nll": float(res.fun),
        "matched_adapter_ids": matched_ids,
    }
    ensemble_path.write_text(json.dumps(e, indent=2))
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pools", nargs="+", required=True, type=Path)
    ap.add_argument("--ensemble-dir", default=Path("ensemble_results"), type=Path)
    ap.add_argument("--cache-dir", default=Path("ensemble_cache"), type=Path)
    ap.add_argument("--out", default=Path("analysis/g3_lorahub_results.json"), type=Path)
    args = ap.parse_args()

    results = []
    for pool_path in args.pools:
        pool_id = json.loads(pool_path.read_text())["pool_id"]
        ensemble_path = args.ensemble_dir / f"{pool_id}.json"
        if not ensemble_path.exists():
            print(f"SKIP {pool_id}: no ensemble result at {ensemble_path}")
            continue
        r = run_pool(ensemble_path, pool_path, args.cache_dir)
        results.append(r)
        if "error" in r:
            print(f"  {pool_id}: ERROR {r['error']}")
        else:
            print(f"  {pool_id}: lw_acc={r['learned_weight_acc']:.4f} "
                  f"(bs={r['best_single_acc']:.4f}, sv={r['soft_vote_acc'] or 0:.4f}, "
                  f"gs={r['greedy_soup_acc'] or 0:.4f}) "
                  f"eff_M={r['effective_M']}/{r['M']}")
    args.out.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
