#!/usr/bin/env python3
"""LoRAHub-style learned-weight ensemble baseline for HellaSwag pools.

LoRAHub (Huang et al., 2023) optimizes a real-valued weight vector α∈R^M
on a held-out set to minimize cross-entropy of a weighted ensemble; we
implement the same family of methods using gradient-based L-BFGS on
val_combine, then evaluate on test. This adds a published-baseline
adjudication cell to v1.1 multi-choice pools.

Usage:
    python3 analysis/learned_weight_baseline.py \\
        --ensemble ensemble_results/pool_a_hellaswag_roberta.json \\
        --cache-dir ensemble_cache/ \\
        --pool pools/pool_a_hellaswag_roberta.json
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
from scipy.optimize import minimize


def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x); return e / e.sum(axis=axis, keepdims=True)


def neg_log_lik(alpha, P_val, y_val):
    """Weighted ensemble NLL on val_combine. P_val: (M, N_val, K)."""
    w = np.maximum(alpha, 0)
    s = w.sum()
    if s <= 0: return 1e10
    w = w / s
    p_avg = (w[:, None, None] * P_val).sum(axis=0)
    p_avg = np.clip(p_avg, 1e-12, 1.0)
    return -np.log(p_avg[np.arange(len(y_val)), y_val]).mean()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--ensemble', required=True, type=Path)
    ap.add_argument('--cache-dir', required=True, type=Path)
    ap.add_argument('--pool', required=True, type=Path)
    ap.add_argument('--task', default='hellaswag')
    args = ap.parse_args()

    e = json.loads(args.ensemble.read_text())
    pool = json.loads(args.pool.read_text())
    pool_id = pool['pool_id']
    splits = e['splits']
    val_idx = np.array(splits['val_combine'], dtype=np.int64)
    test_idx = np.array(splits['test'], dtype=np.int64)
    labels = np.array(e['labels_test'], dtype=np.int64)

    # full labels (for val_combine)
    from datasets import load_dataset
    ds = load_dataset(args.task, split='validation').shuffle(seed=0).select(range(5000))
    full_labels = np.array([int(x or 0) for x in ds['label']], dtype=np.int64)

    # Load per-adapter logits from cache
    L = []
    for a in pool['adapters']:
        f = args.cache_dir / f"{pool_id}__{a['adapter_id']}.npy"
        if f.exists():
            L.append(np.load(f))
    L = np.stack(L, axis=0)  # (M, N_total, K)
    M, N_total, K = L.shape
    P = softmax(L, axis=-1)
    P_val = P[:, val_idx]
    P_test = P[:, test_idx]
    y_val = full_labels[val_idx]
    y_test = labels  # already test-aligned

    # L-BFGS-B with α >= 0 box constraint
    alpha0 = np.ones(M) / M
    res = minimize(neg_log_lik, alpha0, args=(P_val, y_val),
                   method='L-BFGS-B', bounds=[(0, None)] * M,
                   options={'maxiter': 200, 'ftol': 1e-7})
    w = np.maximum(res.x, 0); w /= max(w.sum(), 1e-12)

    # Apply on test
    p_test = (w[:, None, None] * P_test).sum(axis=0)
    pred = p_test.argmax(axis=1)
    acc = float((pred == y_test).mean())
    # ECE
    conf = p_test.max(axis=1)
    correct = (pred == y_test).astype(np.float64)
    order = np.argsort(conf)
    bins = np.array_split(order, 15)
    ece = 0.0
    for b in bins:
        if len(b) == 0: continue
        ece += (len(b) / len(conf)) * abs(conf[b].mean() - correct[b].mean())

    # Compare to existing methods
    print(f'\\n=== {pool_id} learned-weight baseline (LoRAHub-style) ===')
    print(f'  M={M} val_n={len(val_idx)} test_n={len(test_idx)}')
    print(f'  weight stats: min={w.min():.3f} max={w.max():.3f} effective_M={(w>0.01).sum()}')
    print(f'  learned_weight  acc={acc:.4f}  ECE={ece:.4f}')
    print(f'  best_single     acc={e["methods"]["best_single"]["accuracy"]:.4f}  ECE={e["methods"]["best_single"]["ece_15bin"]:.4f}')
    print(f'  soft_vote       acc={e["methods"]["soft_vote"]["accuracy"]:.4f}  ECE={e["methods"]["soft_vote"]["ece_15bin"]:.4f}')
    print(f'  greedy_soup     acc={e["methods"]["greedy_soup"]["accuracy"]:.4f}  ECE={e["methods"]["greedy_soup"]["ece_15bin"]:.4f}')

    # Persist into ensemble JSON for reuse
    e['methods']['learned_weight'] = {
        'accuracy': acc, 'ece_15bin': ece,
        'predictions_test': pred.tolist(),
        'confidence_test': conf.tolist(),
        'weights': w.tolist(),
        'optimizer': 'L-BFGS-B', 'val_combine_loss': float(res.fun),
    }
    args.ensemble.write_text(json.dumps(e, indent=2))
    print(f'  → updated {args.ensemble}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
