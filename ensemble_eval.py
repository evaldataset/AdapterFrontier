#!/usr/bin/env python3
"""Evaluate ensemble combination methods on an adapter pool.

Supported methods:
    - majority_vote       : hard-vote over argmax predictions
    - soft_vote           : average softmax probabilities, argmax
    - logit_avg           : average raw logits, argmax (numerically equivalent to soft_vote for temp=1)
    - greedy_soup         : greedy weight averaging on val_selection, measured on val_combine
    - uniform_soup        : uniform weight averaging baseline
    - best_single         : best-on-val_selection single adapter baseline
    - oracle_single       : best-on-test oracle (upper bound sanity check, not for claims)

Split protocol:
    - Adapters trained on `train` (outside this script)
    - Each sample in the eval split is split into val_selection / val_combine / test deterministically.
    - `best_single`, `greedy_soup` use ONLY val_selection
    - Combination hyperparameters (temperature, weights) tuned ONLY on val_combine
    - Final metrics reported on test

Usage:
    python3 ensemble_eval.py --pool pools/pool_a_mnli_bert.json \
        --methods majority_vote,soft_vote,greedy_soup,uniform_soup,best_single \
        --task mnli --out ensemble_results/pool_a_mnli_bert.json
"""
from __future__ import annotations

import argparse
import json
import copy
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import PeftModel

from peft_utils import provenance


TASK_CONFIGS = {
    "mnli": {
        "dataset": "nyu-mll/glue", "config": "mnli", "split": "validation_matched",
        "fields": ("premise", "hypothesis"), "label": "label", "num_labels": 3,
    },
    "sst2": {
        "dataset": "nyu-mll/glue", "config": "sst2", "split": "validation",
        "fields": ("sentence", None), "label": "label", "num_labels": 2,
    },
    "boolq": {
        "dataset": "aps/super_glue", "config": "boolq", "split": "validation",
        "fields": ("question", "passage"), "label": "label", "num_labels": 2,
        "default_max_length": 256,  # passages are long; must match training seq_length
    },
    "anli": {
        # Use dev_r3 to match the training split used in sweep_configs
        "dataset": "facebook/anli", "config": None, "split": "dev_r3",
        "fields": ("premise", "hypothesis"), "label": "label", "num_labels": 3,
    },
    "agnews": {
        "dataset": "fancyzhx/ag_news", "config": None, "split": "test",
        "fields": ("text", None), "label": "label", "num_labels": 4,
    },
    "qnli": {
        "dataset": "nyu-mll/glue", "config": "qnli", "split": "validation",
        "fields": ("question", "sentence"), "label": "label", "num_labels": 2,
    },
}


def split_indices(n: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Deterministic 40/20/40 split -> (val_selection, val_combine, test)."""
    rng = np.random.RandomState(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_sel = int(0.4 * n)
    n_combo = int(0.2 * n)
    return idx[:n_sel], idx[n_sel:n_sel + n_combo], idx[n_sel + n_combo:]


@torch.no_grad()
def compute_logits_for_adapter(
    base_model_id: str, adapter_path: str, tokenizer, texts_a: list, texts_b: list | None,
    batch_size: int, device: str, max_length: int, num_labels: int,
) -> np.ndarray:
    """Return (N, num_labels) logits for all examples under one adapter."""
    # num_labels must match what the adapter was trained with, otherwise the
    # classifier head shape mismatches at PeftModel.from_pretrained.
    base = AutoModelForSequenceClassification.from_pretrained(base_model_id, num_labels=num_labels)
    # Decoder LMs (Qwen, Llama) used for sequence classification need a pad
    # token; the training script (real_lora_classify) already does this, but
    # we need it again here at eval time on the freshly-loaded base.
    if base.config.pad_token_id is None and tokenizer.pad_token_id is not None:
        base.config.pad_token_id = tokenizer.pad_token_id
    model = PeftModel.from_pretrained(base, adapter_path)
    model.to(device).eval()

    all_logits = []
    for start in range(0, len(texts_a), batch_size):
        a_chunk = texts_a[start:start + batch_size]
        b_chunk = texts_b[start:start + batch_size] if texts_b else None
        enc = tokenizer(a_chunk, b_chunk, padding=True, truncation=True,
                        max_length=max_length, return_tensors="pt").to(device)
        logits = model(**enc).logits.detach().to(torch.float32).cpu().numpy()
        all_logits.append(logits)

    del model, base
    torch.cuda.empty_cache()
    return np.concatenate(all_logits, axis=0)


def majority_vote(logits_stack: np.ndarray) -> np.ndarray:
    preds = logits_stack.argmax(axis=-1)
    n_adapters, n_examples = preds.shape
    out = np.zeros(n_examples, dtype=np.int64)
    for i in range(n_examples):
        vals, counts = np.unique(preds[:, i], return_counts=True)
        out[i] = vals[counts.argmax()]
    return out


def soft_vote(logits_stack: np.ndarray) -> np.ndarray:
    probs = torch.softmax(torch.from_numpy(logits_stack), dim=-1).numpy()
    return probs.mean(axis=0).argmax(axis=-1)


def logit_avg(logits_stack: np.ndarray) -> np.ndarray:
    return logits_stack.mean(axis=0).argmax(axis=-1)


def accuracy(preds: np.ndarray, labels: np.ndarray) -> float:
    return float((preds == labels).mean())


# ---------------------------------------------------------------------------
# Calibration & reliability (claim-C arm)
# ---------------------------------------------------------------------------

def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def method_probs(method: str, logits_subset: np.ndarray, chosen: list[int] | None = None,
                 best_idx: int | None = None) -> np.ndarray:
    """Return per-example probability distribution (N_examples, K) for a method.
    `logits_subset` shape: (N_adapters, N_examples, K) restricted to the eval split.
    """
    if method in ("logit_avg", "uniform_soup"):
        return _softmax(logits_subset.mean(axis=0))
    if method == "soft_vote":
        return _softmax(logits_subset).mean(axis=0)
    if method == "majority_vote":
        # hard-vote frequency as probability — standard hard-ensemble calibration
        preds = logits_subset.argmax(axis=-1)  # (N_ad, N)
        n_ad, n = preds.shape
        k = logits_subset.shape[-1]
        out = np.zeros((n, k), dtype=np.float64)
        for c in range(k):
            out[:, c] = (preds == c).sum(axis=0) / n_ad
        return out
    if method == "greedy_soup":
        assert chosen is not None
        return _softmax(logits_subset[chosen].mean(axis=0))
    if method in ("best_single", "oracle_single"):
        assert best_idx is not None
        return _softmax(logits_subset[best_idx])
    raise ValueError(f"method_probs: unknown method {method}")


def calibration_metrics(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> dict:
    """ECE (equal-mass), MCE, Brier, NLL, plus per-bin reliability data.
    probs: (N, K) — must sum to 1 along axis=1.
    """
    n, k = probs.shape
    preds = probs.argmax(axis=1)
    confidences = probs[np.arange(n), preds]
    correct = (preds == labels).astype(np.float64)

    # one-hot for Brier / NLL
    onehot = np.zeros_like(probs)
    onehot[np.arange(n), labels] = 1.0
    brier = float(((probs - onehot) ** 2).sum(axis=1).mean())
    eps = 1e-12
    nll = float(-np.log(np.clip(probs[np.arange(n), labels], eps, 1.0)).mean())

    # equal-mass binning over confidences
    order = np.argsort(confidences)
    bin_edges = np.array_split(order, n_bins)
    bins = []
    ece = 0.0
    mce = 0.0
    for b_idx in bin_edges:
        if len(b_idx) == 0:
            continue
        mean_conf = float(confidences[b_idx].mean())
        mean_acc = float(correct[b_idx].mean())
        gap = abs(mean_conf - mean_acc)
        weight = len(b_idx) / n
        ece += weight * gap
        mce = max(mce, gap)
        bins.append({
            "n": int(len(b_idx)),
            "mean_confidence": mean_conf,
            "mean_accuracy": mean_acc,
        })
    return {
        "ece_15bin_equalmass": float(ece),
        "mce": float(mce),
        "brier": brier,
        "nll": nll,
        "reliability_bins": bins,
    }


def ood_scores(probs: np.ndarray, logits: np.ndarray) -> dict:
    """OOD-detection score arrays (no AUROC yet — needs OOD set at eval time).
    Returns per-example MSP and energy; downstream OOD eval consumes these.
    """
    msp = probs.max(axis=1)
    # Energy = -logsumexp(logits) (Liu et al. 2020)
    m = logits.max(axis=1, keepdims=True)
    energy = -(m.squeeze(1) + np.log(np.exp(logits - m).sum(axis=1)))
    return {
        "msp": msp.tolist(),
        "energy": energy.tolist(),
    }


def greedy_soup_selection(
    logits_stack: np.ndarray, labels_sel: np.ndarray, sel_idx: np.ndarray,
    individual_acc_on_sel: np.ndarray,
) -> list[int]:
    """Greedy subset by held-out ensemble gain. Uses val_selection only."""
    n = logits_stack.shape[0]
    order = np.argsort(-individual_acc_on_sel)  # descending by individual quality
    chosen: list[int] = [int(order[0])]
    best_acc = individual_acc_on_sel[order[0]]

    for candidate in order[1:]:
        trial = chosen + [int(candidate)]
        trial_logits = logits_stack[trial][:, sel_idx, :].mean(axis=0)
        trial_preds = trial_logits.argmax(axis=-1)
        trial_acc = accuracy(trial_preds, labels_sel)
        if trial_acc >= best_acc:
            chosen = trial
            best_acc = trial_acc
    return chosen


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--pool", required=True, type=Path)
    p.add_argument("--task", required=True, choices=list(TASK_CONFIGS))
    p.add_argument("--methods", default="majority_vote,soft_vote,logit_avg,greedy_soup,uniform_soup,best_single")
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--cache-dir", type=Path, default=Path("./ensemble_cache"),
                   help="Where to cache per-adapter logits so multiple methods are cheap")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--max-length", type=int, default=None,
                   help="Override max sequence length; defaults to task's default_max_length or 128")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--split-seed", type=int, default=0)
    p.add_argument("--max-eval-samples", type=int, default=None)
    p.add_argument("--ood-scores", action="store_true",
                   help="Save per-example MSP/energy arrays for OOD AUROC scoring later")
    p.add_argument("--reliability-bins", type=int, default=15,
                   help="Number of equal-mass bins for ECE/reliability diagram")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    pool = json.loads(args.pool.read_text())
    # Resolve task-dependent max_length before inference
    _task_cfg = TASK_CONFIGS[args.task]
    if args.max_length is None:
        args.max_length = _task_cfg.get("default_max_length", 128)
    adapters = pool["adapters"]
    if not adapters:
        raise SystemExit("Empty pool")

    task = TASK_CONFIGS[args.task]
    ds = load_dataset(task["dataset"], task["config"], split=task["split"])
    if args.max_eval_samples:
        ds = ds.select(range(min(args.max_eval_samples, len(ds))))

    field_a, field_b = task["fields"]
    texts_a = [ex[field_a] for ex in ds]
    texts_b = [ex[field_b] for ex in ds] if field_b else None
    labels = np.array([ex[task["label"]] for ex in ds], dtype=np.int64)

    base_model_id = adapters[0]["metadata"].get("base_model") or "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token  # for decoder LMs (Qwen, Llama)

    args.cache_dir.mkdir(parents=True, exist_ok=True)
    pool_key = pool.get("pool_id") or pool.get("pool_name") or args.pool.stem
    logits_list = []
    for entry in adapters:
        aid = entry["adapter_id"]
        cache_file = args.cache_dir / f"{pool_key}__{aid}.npy"
        if cache_file.exists():
            logits = np.load(cache_file)
        else:
            print(f"[ensemble_eval] computing logits for {aid}")
            logits = compute_logits_for_adapter(
                base_model_id, entry["path"], tokenizer, texts_a, texts_b,
                args.batch_size, args.device, args.max_length,
                num_labels=task["num_labels"],
            )
            np.save(cache_file, logits)
        logits_list.append(logits)

    logits_stack = np.stack(logits_list, axis=0)  # (N_adapters, N_examples, num_labels)
    sel_idx, combo_idx, test_idx = split_indices(len(labels), seed=args.split_seed)
    labels_sel = labels[sel_idx]
    labels_test = labels[test_idx]

    individual_acc_on_sel = np.array([
        accuracy(logits_list[i][sel_idx].argmax(-1), labels_sel) for i in range(len(adapters))
    ])
    individual_acc_on_test = np.array([
        accuracy(logits_list[i][test_idx].argmax(-1), labels_test) for i in range(len(adapters))
    ])

    # Per-adapter calibration on test split (claim-C arm needs single-adapter ECE
    # to compare against ensemble ECE in frontier regression)
    individual_calibration = []
    individual_predictions_test = []
    individual_confidence_test = []
    for i in range(len(adapters)):
        single_probs = _softmax(logits_list[i][test_idx])
        individual_calibration.append(
            calibration_metrics(single_probs, labels_test, n_bins=args.reliability_bins)
        )
        individual_predictions_test.append(single_probs.argmax(axis=-1).astype(int).tolist())
        individual_confidence_test.append(single_probs.max(axis=-1).astype(float).tolist())

    results: dict = {
        "pool_name": pool.get("pool_name") or pool.get("pool_id") or args.pool.stem,
        "pool_id": pool.get("pool_id") or pool.get("pool_name") or args.pool.stem,
        "pool_type": pool.get("pool_type"),
        "schema_version": 1,
        "task": args.task,
        "n_adapters": len(adapters),
        "n_test": int(len(test_idx)),
        "individual_accuracy_test": individual_acc_on_test.tolist(),
        "individual_calibration_test": individual_calibration,
        "individual_predictions_test": individual_predictions_test,
        "individual_confidence_test": individual_confidence_test,
        "labels_test": labels_test.astype(int).tolist(),
        "test_indices": test_idx.astype(int).tolist(),
        "methods": {},
    }

    test_logits_stack = logits_stack[:, test_idx, :]  # (N_ad, N_test, K)

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    for m in methods:
        method_meta: dict = {}
        chosen_subset: list[int] | None = None
        best_idx: int | None = None

        if m in ("majority_vote", "soft_vote", "logit_avg", "uniform_soup"):
            if m == "uniform_soup":
                method_meta["note"] = (
                    "prediction-space uniform average; weight-space soup requires separate merge step"
                )
        elif m == "greedy_soup":
            chosen_subset = greedy_soup_selection(
                logits_stack, labels_sel, sel_idx, individual_acc_on_sel
            )
            method_meta["chosen_indices"] = chosen_subset
            method_meta["chosen_count"] = len(chosen_subset)
        elif m == "best_single":
            best_idx = int(individual_acc_on_sel.argmax())
            method_meta["chosen_index"] = best_idx
        elif m == "oracle_single":
            best_idx = int(individual_acc_on_test.argmax())
            method_meta["chosen_index"] = best_idx
            method_meta["note"] = "uses test-set selection; for sanity only, not for claims"
        else:
            results["methods"][m] = {"error": f"unknown method {m}"}
            continue

        try:
            probs = method_probs(m, test_logits_stack, chosen=chosen_subset, best_idx=best_idx)
        except Exception as e:  # narrow once stable
            results["methods"][m] = {"error": f"probs computation failed: {e}", **method_meta}
            continue

        preds = probs.argmax(axis=-1)
        method_meta["accuracy_test"] = accuracy(preds, labels_test)
        method_meta["predictions_test"] = preds.astype(int).tolist()
        method_meta["confidence_test"] = probs.max(axis=-1).astype(float).tolist()
        method_meta["calibration"] = calibration_metrics(
            probs, labels_test, n_bins=args.reliability_bins
        )

        if args.ood_scores:
            # represent the method's logits-equivalent for energy scoring
            if m in ("logit_avg", "uniform_soup"):
                method_logits = test_logits_stack.mean(axis=0)
            elif m == "greedy_soup":
                method_logits = test_logits_stack[chosen_subset].mean(axis=0)
            elif m in ("best_single", "oracle_single"):
                method_logits = test_logits_stack[best_idx]
            else:
                # for vote-based methods, energy via averaged logits is a reasonable proxy
                method_logits = test_logits_stack.mean(axis=0)
            method_meta["ood_scores"] = ood_scores(probs, method_logits)

        results["methods"][m] = method_meta

    results["provenance"] = provenance({k: str(v) for k, v in vars(args).items()})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        json.dump(results, f, indent=2)
    print(f"[ensemble_eval] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
