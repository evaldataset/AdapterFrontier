#!/usr/bin/env python3
"""E2: Cross-corpus generalization stress test of the diversity-quality
frontier regression.

The headline LOPO-CV R²=0.60 is computed over ALL 48 encoder accuracy
pools. But how well does the regression GENERALIZE to a NEW pool TYPE
that wasn't in the training set?

  H1. Pool-type holdout: train on Pool-A, test on Pool-B + Pool-C
  H2. Family holdout:    train on encoder pools, test on decoder pools
  H3. Task-family holdout: train on v1.0 classification, test on v1.1
  H4. Random 50/50 within encoder (control)

Usage:
    python3 analysis/frontier_cross_corpus.py \\
        --analysis-dir analysis/ \\
        --slice accuracy__n_rank \\
        --out analysis/frontier_cross_corpus.json
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import sys

sys.path.insert(0, str(Path(__file__).parent))
from frontier import load_diversity, load_cm_cells, build_design_matrix


DECODER_TAGS = ("qwen", "tinyllama", "smollm", "pythia", "llama", "mistral")


def is_decoder(pool_id: str) -> bool:
    p = pool_id.lower()
    return any(t in p for t in DECODER_TAGS)


def is_v11(pool_id: str) -> bool:
    p = pool_id.lower()
    return ("hellaswag" in p) or ("gsm8k" in p)


def pool_type(pool_id: str) -> str:
    if "pool_a_" in pool_id: return "A"
    if "pool_b_" in pool_id: return "B"
    if "pool_c_" in pool_id: return "C"
    return "other"


def ridge_fit_predict(X_tr, y_tr, X_te, lam=1.0, min_std=0.01):
    """Ridge with feature standardization. Drops features with near-zero
    variance in TRAIN (else standardization explodes when train σ ≈ 0)."""
    sigma = X_tr.std(axis=0)
    keep = sigma > min_std
    if keep.sum() == 0:
        return np.full(X_te.shape[0], y_tr.mean())
    X_tr = X_tr[:, keep]
    X_te = X_te[:, keep]
    mu = X_tr.mean(axis=0)
    sigma = X_tr.std(axis=0) + 1e-8
    X_tr_n = (X_tr - mu) / sigma
    X_te_n = (X_te - mu) / sigma
    n_feat = X_tr_n.shape[1]
    A = X_tr_n.T @ X_tr_n + lam * np.eye(n_feat)
    b = X_tr_n.T @ (y_tr - y_tr.mean())
    w = np.linalg.solve(A, b)
    intercept = y_tr.mean()
    return X_te_n @ w + intercept


def r2_score(y_true, y_pred, y_null_mean=None):
    sse = float(((y_true - y_pred) ** 2).sum())
    null = y_null_mean if y_null_mean is not None else y_true.mean()
    sst = float(((y_true - null) ** 2).sum())
    return float("nan") if sst < 1e-12 else 1.0 - sse / sst


def run_split(rows, diversity, train_mask, test_mask, label):
    train_rows = [r for r, m in zip(rows, train_mask) if m]
    test_rows = [r for r, m in zip(rows, test_mask) if m]
    if len(train_rows) < 5 or len(test_rows) < 5:
        return {"label": label, "n_train": len(train_rows), "n_test": len(test_rows),
                "error": "insufficient data (n<5 in train or test)"}
    X_tr, y_tr, _, _ = build_design_matrix(train_rows, diversity)
    X_te, y_te, _, _ = build_design_matrix(test_rows, diversity)
    if X_tr.shape[1] != X_te.shape[1]:
        return {"label": label, "error": "feature mismatch"}
    y_pred = ridge_fit_predict(X_tr, y_tr, X_te, lam=1.0)
    return {
        "label": label,
        "n_train_cells": len(train_rows),
        "n_test_cells": len(test_rows),
        "train_pools": len(set(r["pool_id"] for r in train_rows)),
        "test_pools": len(set(r["pool_id"] for r in test_rows)),
        "test_r2": float(r2_score(y_te, y_pred, y_null_mean=y_tr.mean())),
        "y_tr_mean": float(y_tr.mean()),
        "y_te_mean": float(y_te.mean()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--analysis-dir", default=Path("analysis"), type=Path)
    ap.add_argument("--slice", default="accuracy__n_rank")
    ap.add_argument("--out", default=Path("analysis/frontier_cross_corpus.json"), type=Path)
    args = ap.parse_args()

    diversity = load_diversity(args.analysis_dir)
    cells = load_cm_cells(args.analysis_dir)
    print(f"[E2] loaded {len(cells)} cm cells, {len(diversity)} diversity files")

    metric, baseline_kind = args.slice.split("__")
    rows = [c for c in cells
            if c.get("metric") == metric
            and c.get("baseline_kind") == baseline_kind
            and c.get("pool_id") in diversity]
    print(f"[E2] slice {args.slice}: {len(rows)} cells from {len(set(r['pool_id'] for r in rows))} pools")

    rows_w = rows
    results = []

    # H1. Pool-A → Pool-B+C
    train_mask = [pool_type(r["pool_id"]) == "A" for r in rows_w]
    test_mask = [pool_type(r["pool_id"]) in ("B", "C") for r in rows_w]
    results.append(run_split(rows_w, diversity, train_mask, test_mask,
                              "H1: Pool-A → Pool-B+C"))

    # H2. encoder → decoder
    train_mask = [not is_decoder(r["pool_id"]) for r in rows_w]
    test_mask = [is_decoder(r["pool_id"]) for r in rows_w]
    results.append(run_split(rows_w, diversity, train_mask, test_mask,
                              "H2: encoder → decoder"))

    # H3. v1.0 → v1.1
    train_mask = [not is_v11(r["pool_id"]) for r in rows_w]
    test_mask = [is_v11(r["pool_id"]) for r in rows_w]
    results.append(run_split(rows_w, diversity, train_mask, test_mask,
                              "H3: v1.0 classification → v1.1 (HellaSwag/GSM8K)"))

    # H4. Random 50/50 within encoder (control)
    rng = np.random.default_rng(0)
    enc_idx = [i for i, r in enumerate(rows_w) if not is_decoder(r["pool_id"])]
    rng.shuffle(enc_idx)
    half = len(enc_idx) // 2
    train_set = set(enc_idx[:half])
    test_set = set(enc_idx[half:])
    train_mask = [i in train_set for i in range(len(rows_w))]
    test_mask = [i in test_set for i in range(len(rows_w))]
    results.append(run_split(rows_w, diversity, train_mask, test_mask,
                              "H4 (control): random 50/50 within encoder cells"))

    print(f"\n=== E2 Cross-Corpus Generalization (slice={args.slice}) ===")
    for r in results:
        if "error" in r:
            print(f"  {r['label']:55s}  ERROR: {r['error']}")
        else:
            print(f"  {r['label']:55s}  R²_test={r['test_r2']:+.3f}  "
                  f"(n_tr={r['n_train_cells']:3d}/{r['train_pools']:2d}p → "
                  f"n_te={r['n_test_cells']:3d}/{r['test_pools']:2d}p)")

    args.out.write_text(json.dumps({"slice": args.slice, "results": results}, indent=2))
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    raise SystemExit(main())
