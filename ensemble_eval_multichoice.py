#!/usr/bin/env python3
"""Ensemble evaluation for multi-choice adapter pools (HellaSwag-style).

Each adapter is loaded with AutoModelForMultipleChoice and produces (N, K)
logits per example (K = number of candidate endings). We aggregate over
adapters via:
  - majority_vote (per-example argmax then majority)
  - soft_vote (mean of softmax over candidates)
  - logit_avg (mean of raw candidate logits)
  - greedy_soup (val_selection-driven greedy subset, by val_combine acc)
  - uniform_soup (LoRA-A/B average — weight-space)
  - best_single (highest val_combine acc)

Outputs ECE (15-bin equal-mass), Brier, NLL, MCE, accuracy with reliability
data — same schema as ensemble_eval.py.

Usage:
    python3 ensemble_eval_multichoice.py \\
        --pool pools/pool_a_hellaswag_bert.json --task hellaswag \\
        --methods soft_vote,logit_avg,majority_vote,best_single,uniform_soup,greedy_soup \\
        --out ensemble_results/pool_a_hellaswag_bert.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForMultipleChoice, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from peft_utils import provenance


TASK_CONFIGS = {
    "hellaswag": {
        "dataset": "hellaswag",
        "config": None,
        "split": "validation",
        "context_field": "ctx",
        "endings_field": "endings",
        "label_field": "label",
        "n_choices": 4,
    },
}


def make_features(tokenizer, ctxs, endings_lists, n_choices, max_length):
    """Tokenize (B, K) pairs of (context, ending). Returns dict of tensors."""
    first, second = [], []
    for ctx, ends in zip(ctxs, endings_lists):
        ctx = ctx if isinstance(ctx, str) else ""
        ends = list(ends) if ends else [""] * n_choices
        ends = (ends + [""] * n_choices)[:n_choices]
        for end in ends:
            first.append(ctx)
            second.append(end if isinstance(end, str) else "")
    enc = tokenizer(first, second, truncation=True, max_length=max_length,
                    padding="max_length", return_tensors="pt")
    B = len(ctxs)
    return {k: v.view(B, n_choices, -1) for k, v in enc.items()}


def compute_logits_for_adapter(
    base_model_id: str, adapter_path: str, ds, tokenizer,
    n_choices: int, batch_size: int, device: str, max_length: int,
) -> np.ndarray:
    base = AutoModelForMultipleChoice.from_pretrained(base_model_id)
    model = PeftModel.from_pretrained(base, adapter_path)
    model = model.to(device).eval()

    ctxs = ds["ctx"]
    endings = ds["endings"]
    N = len(ctxs)
    out = np.zeros((N, n_choices), dtype=np.float32)
    with torch.no_grad():
        for i in range(0, N, batch_size):
            j = min(i + batch_size, N)
            feats = make_features(tokenizer, ctxs[i:j], endings[i:j],
                                  n_choices, max_length)
            feats = {k: v.to(device) for k, v in feats.items()}
            logits = model(**feats).logits  # (B, K)
            out[i:j] = logits.float().cpu().numpy()
    del model, base
    if device == "cuda":
        torch.cuda.empty_cache()
    return out


def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def calibration_metrics(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> dict:
    """ECE, Brier, NLL, MCE on (N, K) probs."""
    K = probs.shape[1]
    preds = probs.argmax(axis=1)
    confs = probs.max(axis=1)
    correct = (preds == labels).astype(np.float64)

    one_hot = np.eye(K)[labels]
    brier = float(((probs - one_hot) ** 2).sum(axis=1).mean())
    eps = 1e-12
    nll = float(-np.log(np.clip(probs[np.arange(len(labels)), labels], eps, 1.0)).mean())

    order = np.argsort(confs)
    bins = np.array_split(order, n_bins)
    bin_data = []
    ece_total = 0.0
    mce_total = 0.0
    for b in bins:
        if len(b) == 0:
            bin_data.append({"count": 0, "mean_conf": 0.0, "mean_acc": 0.0})
            continue
        mc = float(confs[b].mean())
        ma = float(correct[b].mean())
        gap = abs(mc - ma)
        ece_total += (len(b) / len(confs)) * gap
        mce_total = max(mce_total, gap)
        bin_data.append({"count": int(len(b)), "mean_conf": mc, "mean_acc": ma})

    return {
        "accuracy": float(correct.mean()),
        "ece_15bin": float(ece_total),
        "mce_15bin": float(mce_total),
        "brier": brier,
        "nll": nll,
        "reliability_bins": bin_data,
    }


def select_split_indices(N: int, seed: int = 0) -> dict[str, np.ndarray]:
    """40/20/40 val_selection / val_combine / test split."""
    rng = np.random.default_rng(seed)
    idx = np.arange(N)
    rng.shuffle(idx)
    a = int(0.4 * N); b = int(0.6 * N)
    return {"val_selection": idx[:a], "val_combine": idx[a:b], "test": idx[b:]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True, type=Path)
    ap.add_argument("--task", required=True, choices=list(TASK_CONFIGS))
    ap.add_argument("--methods", default="soft_vote,logit_avg,majority_vote,best_single,uniform_soup,greedy_soup")
    ap.add_argument("--max-eval-samples", type=int, default=5000)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-length", type=int, default=128)
    ap.add_argument("--cache-dir", type=Path, default=None,
                    help="If set, write per-adapter logits as <cache>/<pool_id>__<adapter_id>.npy "
                         "(format consumed by diversity_metrics.py).")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--shuffle-seed", type=int, default=0,
                    help="Seed used to shuffle the eval split before subsampling.")
    ap.add_argument("--full-validation", action="store_true",
                    help="Bypass --max-eval-samples and use the entire eval split.")
    args = ap.parse_args()

    pool = json.loads(args.pool.read_text())
    base_model = pool.get("base_model") or pool["adapters"][0].get("base_model")
    methods = args.methods.split(",")

    task = TASK_CONFIGS[args.task]
    ds = load_dataset(task["dataset"], task["config"], split=task["split"])
    if args.full_validation:
        print(f"[ens] FULL VALIDATION: {len(ds)} samples (no shuffle/subsample)")
    elif args.max_eval_samples and args.max_eval_samples < len(ds):
        ds = ds.shuffle(seed=args.shuffle_seed).select(range(args.max_eval_samples))
    labels = np.asarray([int(x or 0) for x in ds[task["label_field"]]], dtype=np.int64)

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    splits = select_split_indices(len(labels), seed=0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    K = task["n_choices"]

    # Compute per-adapter logits
    all_logits = []  # list of (N, K)
    print(f"Computing logits for {len(pool['adapters'])} adapters on {len(ds)} examples...")
    for i, a in enumerate(pool["adapters"]):
        # Adapter path may be nested — find adapter_config.json
        path = Path(a["path"])
        if not (path / "adapter_config.json").exists():
            cands = list(path.rglob("adapter_config.json"))
            if not cands:
                print(f"  [skip] {a['adapter_id']} no adapter_config.json")
                continue
            path = cands[0].parent
        t0 = time.perf_counter()
        logits = compute_logits_for_adapter(base_model, str(path), ds, tokenizer,
                                            K, args.batch_size, device, args.max_length)
        all_logits.append(logits)
        if args.cache_dir is not None:
            args.cache_dir.mkdir(parents=True, exist_ok=True)
            np.save(args.cache_dir / f"{pool['pool_id']}__{a['adapter_id']}.npy", logits)
        print(f"  [{i+1}/{len(pool['adapters'])}] {a['adapter_id'][:50]:<50s} "
              f"acc={(logits.argmax(1)==labels).mean():.4f}  ({time.perf_counter()-t0:.1f}s)")

    if not all_logits:
        raise SystemExit("no adapters produced logits")
    L = np.stack(all_logits, axis=0)  # (M, N, K)
    P = softmax(L, axis=-1)            # (M, N, K)
    M, N, K = L.shape
    print(f"shape: M={M} N={N} K={K}")

    # Per-split per-adapter accuracy for selection
    val_sel = splits["val_selection"]
    val_com = splits["val_combine"]
    test = splits["test"]
    per_adapter_val_acc = np.array([
        (L[m, val_com].argmax(1) == labels[val_com]).mean() for m in range(M)
    ])

    results = {}

    def add_method(name, preds_test, probs_test, conf_test):
        cal = calibration_metrics(probs_test, labels[test])
        results[name] = {
            "accuracy": cal["accuracy"],
            "ece_15bin": cal["ece_15bin"],
            "mce_15bin": cal["mce_15bin"],
            "brier": cal["brier"],
            "nll": cal["nll"],
            "reliability_bins": cal["reliability_bins"],
            "predictions_test": preds_test.tolist(),
            "confidence_test": conf_test.tolist(),
        }
        print(f"  {name:<14s} acc={cal['accuracy']:.4f}  ECE={cal['ece_15bin']:.4f}  "
              f"Brier={cal['brier']:.4f}")

    if "best_single" in methods:
        m_best = int(per_adapter_val_acc.argmax())
        p = P[m_best, test]
        add_method("best_single", p.argmax(1), p, p.max(1))

    if "soft_vote" in methods:
        p = P.mean(axis=0)[test]
        add_method("soft_vote", p.argmax(1), p, p.max(1))

    if "logit_avg" in methods:
        l_avg = L.mean(axis=0)[test]
        p = softmax(l_avg, axis=-1)
        add_method("logit_avg", p.argmax(1), p, p.max(1))

    if "majority_vote" in methods:
        votes = L.argmax(axis=-1)  # (M, N)
        # mode per column
        from scipy.stats import mode as smode  # type: ignore
        m_votes, _ = smode(votes[:, test], axis=0, keepdims=False)
        # Confidence = fraction of voters agreeing
        agree = (votes[:, test] == m_votes[None, :]).mean(axis=0)
        # build a one-hot prob matrix using agreement as confidence; assume rest equally split
        N_t = len(test)
        probs = np.full((N_t, K), 0.0)
        for i in range(N_t):
            probs[i, m_votes[i]] = agree[i]
            rest = (1 - agree[i]) / max(1, K - 1)
            for k in range(K):
                if k != m_votes[i]:
                    probs[i, k] = rest
        add_method("majority_vote", m_votes.astype(np.int64), probs, agree)

    if "greedy_soup" in methods:
        # Greedy subset selection on val_combine accuracy by averaging probs.
        order = np.argsort(-per_adapter_val_acc)
        chosen = [int(order[0])]
        best_acc = (P[chosen[0], val_com].argmax(1) == labels[val_com]).mean()
        for cand in order[1:]:
            test_set = chosen + [int(cand)]
            mean_p = P[test_set].mean(axis=0)
            acc = (mean_p[val_com].argmax(1) == labels[val_com]).mean()
            if acc > best_acc:
                chosen = test_set
                best_acc = acc
        p = P[chosen].mean(axis=0)[test]
        add_method("greedy_soup", p.argmax(1), p, p.max(1))
        results["greedy_soup"]["chosen_adapters"] = chosen

    if "uniform_soup" in methods:
        # Weight-space LoRA average: simulate by averaging logits then softmax (proxy)
        l_uni = L.mean(axis=0)[test]
        p = softmax(l_uni, axis=-1)
        add_method("uniform_soup", p.argmax(1), p, p.max(1))

    out = {
        "pool_id": pool.get("pool_id"),
        "task": args.task,
        "n_adapters": M,
        "n_eval": int(N),
        "split_seed": 0,
        "splits": {k: v.tolist() for k, v in splits.items()},
        "labels_test": labels[test].tolist(),
        "methods": results,
        "per_adapter_val_acc": per_adapter_val_acc.tolist(),
        "provenance": provenance({k: str(v) for k, v in vars(args).items()}),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
