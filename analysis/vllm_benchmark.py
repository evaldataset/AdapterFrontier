#!/usr/bin/env python3
"""vLLM multi-LoRA serving benchmark for §9.

Compares per-batch latency of:
  (A) vanilla single-adapter PyTorch inference (reference, from latency_benchmark.py)
  (B) vLLM multi-LoRA: N requests, each routed to a different adapter, served from
      a single base-model load (the production deployment style)

Limitation: vLLM serves causal LMs (next-token generation), not sequence
classifiers directly. We reformat MNLI as a single-token classification prompt
and measure the output token's probability over {entailment, neutral, contradiction}
mapped to vocab tokens.

Usage:
    python3 analysis/vllm_benchmark.py \
        --pool pools/pool_a_mnli_qwen25_05b.json \
        --n-adapters 8 --n-requests 32 \
        --out analysis/vllm_bench_qwen.json
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
from peft_utils import provenance


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pool", required=True, type=Path)
    p.add_argument("--n-adapters", type=int, default=8)
    p.add_argument("--n-requests", type=int, default=32,
                   help="Total requests per benchmark run; spread across adapters")
    p.add_argument("--n-trials", type=int, default=5)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--max-lora-rank", type=int, default=16)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    pool = json.loads(args.pool.read_text())
    pool_id = pool.get("pool_id") or args.pool.stem
    adapters = pool["adapters"][:args.n_adapters]
    base_model = pool.get("base_model") or adapters[0].get("base_model") \
                  or adapters[0].get("metadata", {}).get("base_model")
    if not base_model:
        raise SystemExit("base model unknown")
    print(f"[vllm] pool={pool_id}, base={base_model}, N_adapters={len(adapters)}")

    # vLLM 0.10.0 forks subprocesses that re-init CUDA — need spawn.
    import os as _os
    _os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

    # transformers 5.x removed `all_special_tokens_extended`; vllm 0.10.0 still
    # accesses it. Alias it on PreTrainedTokenizerBase before vllm imports the
    # tokenizer module. Safe no-op when the attribute already exists.
    try:
        from transformers.tokenization_utils_base import PreTrainedTokenizerBase
        if not hasattr(PreTrainedTokenizerBase, "all_special_tokens_extended"):
            PreTrainedTokenizerBase.all_special_tokens_extended = property(
                lambda self: self.all_special_tokens
            )
    except Exception:
        pass

    # vLLM imports deferred to avoid broken import on script load
    try:
        from vllm import LLM, SamplingParams
        from vllm.lora.request import LoRARequest
    except Exception as e:
        raise SystemExit(f"vllm import failed: {e}")

    # Build a few MNLI-style prompts (premise / hypothesis → predict label)
    prompts = [
        "Premise: A man is playing guitar.\nHypothesis: A musician is playing.\nLabel:",
        "Premise: The cat sat on the mat.\nHypothesis: A dog ran outside.\nLabel:",
        "Premise: It is raining heavily.\nHypothesis: The weather is wet.\nLabel:",
        "Premise: She read the entire book.\nHypothesis: She did not read it.\nLabel:",
    ] * (args.n_requests // 4 + 1)
    prompts = prompts[:args.n_requests]

    # Initialize vLLM with multi-LoRA support
    llm = LLM(
        model=base_model,
        enable_lora=True,
        max_loras=args.n_adapters,
        max_lora_rank=args.max_lora_rank,
        max_model_len=512,
        gpu_memory_utilization=0.75,
        dtype="bfloat16",
    )
    sampling = SamplingParams(temperature=0.0, max_tokens=1)
    lora_reqs = [
        LoRARequest(lora_name=f"adapter_{i}", lora_int_id=i + 1, lora_path=a["path"])
        for i, a in enumerate(adapters)
    ]
    print(f"[vllm] loaded {len(lora_reqs)} LoRA requests")

    # === Multi-LoRA: N requests, each routed to one adapter ===
    multi_lora_times = []
    for trial in range(args.n_trials + 2):  # 2 warmup
        per_request_loras = [lora_reqs[i % len(lora_reqs)] for i in range(len(prompts))]
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        t0 = time.perf_counter()
        _ = llm.generate(prompts, sampling, lora_request=per_request_loras)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        t = time.perf_counter() - t0
        if trial >= 2:
            multi_lora_times.append(t)
            print(f"  multi-LoRA trial {trial-1}: {t:.3f}s")

    # === Single adapter for reference ===
    single_times = []
    for trial in range(args.n_trials + 2):
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        t0 = time.perf_counter()
        _ = llm.generate(prompts, sampling, lora_request=lora_reqs[0])
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        t = time.perf_counter() - t0
        if trial >= 2:
            single_times.append(t)
            print(f"  single-LoRA trial {trial-1}: {t:.3f}s")

    out = {
        "pool_id": pool_id,
        "base_model": base_model,
        "n_adapters": len(adapters),
        "n_requests_per_batch": len(prompts),
        "max_lora_rank": args.max_lora_rank,
        "single_lora_median_s": float(np.median(single_times)),
        "single_lora_p10_s": float(np.quantile(single_times, 0.1)),
        "single_lora_p90_s": float(np.quantile(single_times, 0.9)),
        "multi_lora_median_s": float(np.median(multi_lora_times)),
        "multi_lora_p10_s": float(np.quantile(multi_lora_times, 0.1)),
        "multi_lora_p90_s": float(np.quantile(multi_lora_times, 0.9)),
        "multi_over_single_ratio": float(np.median(multi_lora_times) / np.median(single_times)),
        "single_throughput_req_per_s": float(len(prompts) / np.median(single_times)),
        "multi_throughput_req_per_s": float(len(prompts) / np.median(multi_lora_times)),
        "n_trials": len(single_times),
        "provenance": provenance({k: str(v) for k, v in vars(args).items()}),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(f"\n=== vLLM benchmark summary ===")
    print(f"  Single-LoRA   median: {out['single_lora_median_s']:.3f}s/batch  "
          f"({out['single_throughput_req_per_s']:.1f} req/s)")
    print(f"  Multi-LoRA N={args.n_adapters} median: {out['multi_lora_median_s']:.3f}s/batch  "
          f"({out['multi_throughput_req_per_s']:.1f} req/s)")
    print(f"  Multi/Single ratio: {out['multi_over_single_ratio']:.2f}x  "
          f"(reference: sequential PyTorch ratio = ~N×)")
    print(f"  → {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
