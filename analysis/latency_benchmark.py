#!/usr/bin/env python3
"""Inference latency benchmark for ensemble §9 — sequential PEFT inference,
no vLLM dependency.

For each (pool, ensemble_size N), measure per-batch wall-time on the
local A6000:
  - load N adapters into the same base model sequentially
  - for each batch, run all N forward passes + mean of softmax probs
  - record median per-batch latency over 10 trials × multiple batch sizes
Then pair this latency cost with the ECE/accuracy gain from §3 to produce
the latency-vs-ECE Pareto frontier.

Usage:
    python3 analysis/latency_benchmark.py \
      --pool pools/pool_a_mnli_bert.json --task mnli --out analysis/latency_pool_a_bert.json
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from peft import PeftModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from peft_utils import provenance


def load_model_with_adapter(base_model_id: str, adapter_path: str, num_labels: int, device: str):
    base = AutoModelForSequenceClassification.from_pretrained(base_model_id, num_labels=num_labels)
    model = PeftModel.from_pretrained(base, adapter_path).to(device).eval()
    return model


@torch.no_grad()
def time_ensemble(adapter_paths, base_model_id, num_labels, tokenizer, device,
                  batch_size, seq_len, max_length, n_trials=10):
    """Sequential N-adapter inference timing.
    Returns median per-batch wall-time in seconds.
    """
    # Synthetic input for stable timing (eliminate dataset variance)
    rng = torch.Generator(device='cpu').manual_seed(0)
    input_ids = torch.randint(0, tokenizer.vocab_size, (batch_size, seq_len), generator=rng).to(device)
    attention_mask = torch.ones_like(input_ids)

    # Pre-load all adapters into memory
    base = AutoModelForSequenceClassification.from_pretrained(base_model_id, num_labels=num_labels)
    if base.config.pad_token_id is None and tokenizer.pad_token_id is not None:
        base.config.pad_token_id = tokenizer.pad_token_id
    base = base.to(device).eval()

    # Sequential ensemble: load adapter, fwd, swap to next
    times = []
    for trial in range(n_trials + 2):  # 2 warmup
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        prob_sum = None
        for ap in adapter_paths:
            m = PeftModel.from_pretrained(base, ap).to(device).eval()
            logits = m(input_ids=input_ids, attention_mask=attention_mask).logits
            probs = torch.softmax(logits, dim=-1)
            prob_sum = probs if prob_sum is None else prob_sum + probs
            del m
        prob_sum = prob_sum / len(adapter_paths)
        if device == "cuda":
            torch.cuda.synchronize()
        t = time.perf_counter() - t0
        if trial >= 2:  # skip warmup
            times.append(t)
    del base
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    return {
        "median_s": float(np.median(times)),
        "p90_s": float(np.quantile(times, 0.9)),
        "p10_s": float(np.quantile(times, 0.1)),
        "n_trials": len(times),
    }


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pool", required=True, type=Path)
    p.add_argument("--task", required=True)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--ensemble-sizes", nargs="+", type=int, default=[1, 2, 4, 8, 16])
    p.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 8, 32])
    p.add_argument("--seq-length", type=int, default=128)
    p.add_argument("--n-trials", type=int, default=10)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--num-labels", type=int, default=3, help="3 for NLI, 2 for SST-2/BoolQ/QNLI, 4 for AG News")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    pool = json.loads(args.pool.read_text())
    pool_id = pool.get("pool_id") or pool.get("pool_name") or args.pool.stem
    adapters = pool["adapters"]
    if not adapters:
        raise SystemExit("empty pool")
    base_model_id = (pool.get("base_model") or adapters[0].get("base_model")
                     or adapters[0].get("metadata", {}).get("base_model"))
    if not base_model_id:
        raise SystemExit("base model unknown")

    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"[latency] pool={pool_id}, base={base_model_id}, n_adapters={len(adapters)}")
    print(f"[latency] ensemble sizes: {args.ensemble_sizes}, batch sizes: {args.batch_sizes}")

    all_paths = [a["path"] for a in adapters]
    grid = []
    for N in args.ensemble_sizes:
        if N > len(adapters):
            print(f"  skip N={N} (only {len(adapters)} adapters)")
            continue
        for bs in args.batch_sizes:
            t = time_ensemble(all_paths[:N], base_model_id, args.num_labels, tokenizer,
                               args.device, bs, args.seq_length, args.seq_length, args.n_trials)
            row = {
                "ensemble_size": N, "batch_size": bs,
                "median_s": t["median_s"], "p10_s": t["p10_s"], "p90_s": t["p90_s"],
                "throughput_examples_per_s": bs / t["median_s"],
            }
            grid.append(row)
            print(f"  N={N:2d}  bs={bs:3d}  median={t['median_s']:.3f}s  thr={row['throughput_examples_per_s']:.1f} ex/s")

    # Pair with ECE/acc data from existing ensemble_results
    ens_path = Path(f"ensemble_results/{pool_id}.json")
    pareto_pairs = []
    if ens_path.exists():
        ens = json.loads(ens_path.read_text())
        ind_acc_mean = float(np.mean(ens.get("individual_accuracy_test", [0])))
        for m_name in ("best_single", "soft_vote", "logit_avg", "greedy_soup"):
            mb = ens.get("methods", {}).get(m_name, {})
            if "accuracy_test" not in mb:
                continue
            n_eff = mb.get("chosen_count", 1) if m_name == "greedy_soup" else (1 if m_name == "best_single" else len(adapters))
            row = next((r for r in grid if r["ensemble_size"] == n_eff and r["batch_size"] == 32), None)
            pareto_pairs.append({
                "method": m_name,
                "n_eff_adapters": int(n_eff),
                "accuracy_test": mb["accuracy_test"],
                "ece_15bin": mb.get("calibration", {}).get("ece_15bin_equalmass"),
                "median_s_at_bs32": (row["median_s"] if row else None),
                "throughput_at_bs32": (row["throughput_examples_per_s"] if row else None),
            })

    out = {
        "pool_id": pool_id, "base_model": base_model_id,
        "device": args.device, "seq_length": args.seq_length,
        "n_trials_per_cell": args.n_trials,
        "grid": grid,
        "pareto_pairs": pareto_pairs,
        "provenance": provenance({k: str(v) for k, v in vars(args).items()}),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {args.out}")
    if pareto_pairs:
        print(f'\nPareto (method × n_eff × acc × ECE × latency):')
        for p in pareto_pairs:
            ms = f'{p["median_s_at_bs32"]:.3f}s' if p["median_s_at_bs32"] else 'n/a'
            print(f'  {p["method"]:<14s} N={p["n_eff_adapters"]:>2d}  acc={p["accuracy_test"]:.4f}  ECE={p["ece_15bin"]:.4f}  bs32_lat={ms}')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
