#!/usr/bin/env python3
"""T1.3 OOD evaluation: take MNLI-trained adapters, eval on ANLI dev_r3.

Both MNLI and ANLI are 3-class entail/neutral/contradiction with the same
label encoding (0/1/2). Adapters trained on MNLI can be evaluated directly
on ANLI without label remapping.

For each MNLI Pool-A/B/C result, run inference on ANLI dev_r3 across all
adapters and save a parallel ensemble_results/<pool>_OOD_anli.json with the
same schema (so compute_match works on it).

Usage:
    python3 analysis/ood_eval.py --pools pools/pool_a_mnli_bert.json pools/pool_a_mnli_qwen25_05b.json
    python3 analysis/ood_eval.py --auto   # picks all *_mnli_* pools
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ensemble_eval import (
    TASK_CONFIGS, _softmax, method_probs,
    calibration_metrics, accuracy, greedy_soup_selection, split_indices,
    compute_logits_for_adapter,
)
from datasets import load_dataset
from transformers import AutoTokenizer
from peft_utils import provenance


def load_pool(path: Path):
    pool = json.loads(path.read_text())
    return pool, path.stem


def run_one_pool(pool, pool_id, ood_task: str, batch_size: int, max_length: int, device: str,
                 cache_dir: Path):
    adapters = pool["adapters"]
    if not adapters:
        raise SystemExit(f"empty pool {pool_id}")
    base = adapters[0].get("base_model") or adapters[0].get("metadata", {}).get("base_model")
    if not base:
        raise SystemExit(f"no base model for {pool_id}")
    task = TASK_CONFIGS[ood_task]
    ds_kwargs = {"path": task["dataset"], "split": task["split"], "trust_remote_code": True}
    if task["config"]:
        ds_kwargs["name"] = task["config"]
    ds = load_dataset(**ds_kwargs)
    field_a, field_b = task["fields"]
    texts_a = [ex[field_a] for ex in ds]
    texts_b = [ex[field_b] for ex in ds] if field_b else None
    labels = np.array([ex[task["label"]] for ex in ds], dtype=np.int64)

    tokenizer = AutoTokenizer.from_pretrained(base)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    cache_dir.mkdir(parents=True, exist_ok=True)
    logits_list = []
    for i, a in enumerate(adapters):
        cache = cache_dir / f"OOD_{ood_task}__{pool_id}__{a['adapter_id']}.npy"
        if cache.exists():
            L = np.load(cache)
        else:
            t0 = time.time()
            L = compute_logits_for_adapter(
                base, a["path"], tokenizer, texts_a, texts_b,
                batch_size, device, max_length, num_labels=task["num_labels"],
            )
            np.save(cache, L)
            print(f"  [{i+1}/{len(adapters)}] {a['adapter_id']} done in {time.time()-t0:.0f}s")
        logits_list.append(L)

    logits_stack = np.stack(logits_list, axis=0)
    sel_idx, combo_idx, test_idx = split_indices(len(labels), seed=0)
    labels_test = labels[test_idx]
    individual_acc_on_sel = np.array([
        accuracy(logits_list[i][sel_idx].argmax(-1), labels[sel_idx]) for i in range(len(adapters))
    ])
    individual_acc_on_test = np.array([
        accuracy(logits_list[i][test_idx].argmax(-1), labels_test) for i in range(len(adapters))
    ])

    individual_calibration = []
    individual_predictions_test = []
    individual_confidence_test = []
    for i in range(len(adapters)):
        single_probs = _softmax(logits_list[i][test_idx])
        individual_calibration.append(calibration_metrics(single_probs, labels_test, n_bins=15))
        individual_predictions_test.append(single_probs.argmax(axis=-1).astype(int).tolist())
        individual_confidence_test.append(single_probs.max(axis=-1).astype(float).tolist())

    test_logits_stack = logits_stack[:, test_idx, :]
    methods_out = {}
    for m in ("majority_vote", "soft_vote", "logit_avg", "uniform_soup", "greedy_soup", "best_single"):
        meta = {}
        chosen = None; best_idx = None
        if m == "greedy_soup":
            chosen = greedy_soup_selection(logits_stack, labels[sel_idx], sel_idx, individual_acc_on_sel)
            meta["chosen_indices"] = chosen
            meta["chosen_count"] = len(chosen)
        elif m == "best_single":
            best_idx = int(individual_acc_on_sel.argmax())
            meta["chosen_index"] = best_idx
        probs = method_probs(m, test_logits_stack, chosen=chosen, best_idx=best_idx)
        preds = probs.argmax(axis=-1)
        meta["accuracy_test"] = accuracy(preds, labels_test)
        meta["predictions_test"] = preds.astype(int).tolist()
        meta["confidence_test"] = probs.max(axis=-1).astype(float).tolist()
        meta["calibration"] = calibration_metrics(probs, labels_test, n_bins=15)
        methods_out[m] = meta

    out = {
        "pool_id": pool_id,
        "pool_id_OOD": f"{pool_id}__OOD_{ood_task}",
        "training_task": "mnli",
        "eval_task": ood_task,
        "schema_version": 1,
        "n_adapters": len(adapters),
        "n_test": int(len(test_idx)),
        "individual_accuracy_test": individual_acc_on_test.tolist(),
        "individual_calibration_test": individual_calibration,
        "individual_predictions_test": individual_predictions_test,
        "individual_confidence_test": individual_confidence_test,
        "labels_test": labels_test.astype(int).tolist(),
        "test_indices": test_idx.astype(int).tolist(),
        "methods": methods_out,
        "provenance": provenance({"ood_task": ood_task, "pool_id": pool_id}),
    }
    return out


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pools", nargs="*", type=Path, default=None)
    p.add_argument("--auto", action="store_true",
                   help="Auto-select all pool_*_mnli_*.json from pools/")
    p.add_argument("--ood-task", default="anli", choices=["anli"])
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--max-length", type=int, default=128)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--cache-dir", type=Path, default=Path("ensemble_cache_ood"))
    p.add_argument("--out-dir", type=Path, default=Path("ensemble_results"))
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.auto:
        pools = sorted(Path("pools").glob("pool_*_mnli_*.json"))
    else:
        pools = args.pools or []
    if not pools:
        raise SystemExit("no pools to process — pass --pools or --auto")
    print(f"OOD eval: {len(pools)} pools, eval_task={args.ood_task}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for p in pools:
        pool, pool_id = load_pool(p)
        out_path = args.out_dir / f"{pool_id}_OOD_{args.ood_task}.json"
        if out_path.exists():
            print(f"  SKIP {pool_id} (exists)")
            continue
        print(f"\n=== {pool_id} → OOD {args.ood_task} ===")
        try:
            res = run_one_pool(pool, pool_id, args.ood_task, args.batch_size,
                                args.max_length, args.device, args.cache_dir)
            out_path.write_text(json.dumps(res, indent=2))
            print(f"  wrote {out_path}")
        except Exception as e:
            print(f"  FAILED: {type(e).__name__}: {e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
