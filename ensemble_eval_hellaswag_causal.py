#!/usr/bin/env python3
"""HellaSwag ensemble eval for decoder LMs via vLLM per-candidate scoring.

For each (ctx, endings[4], label) example and each adapter:
  - Score each candidate ending j by sum of per-token log-likelihood of
    ending tokens conditioned on ctx (lm-evaluation-harness convention).
  - Pick argmax j*.
  - correct = (j* == label).

Aggregation:
  - best_single: per-adapter accuracy
  - majority_vote: mode over per-adapter j* picks
  - soft_vote: per-example sum of softmax-normalized scores across adapters,
    then argmax.

Usage:
    python3 ensemble_eval_hellaswag_causal.py \\
        --pool pools/pool_a_hellaswag_qwen25_05b_causal.json \\
        --out ensemble_results/pool_a_hellaswag_qwen25_05b_causal.json \\
        --max-eval 1000
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
from datasets import load_dataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
from peft_utils import provenance


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--max-eval", type=int, default=1000)
    ap.add_argument("--max-lora-rank", type=int, default=16)
    ap.add_argument("--gpu-mem-util", type=float, default=0.30)
    ap.add_argument("--max-model-len", type=int, default=512)
    args = ap.parse_args()

    pool = json.loads(args.pool.read_text())
    base_model = pool.get("base_model") or pool["adapters"][0].get("base_model")
    adapters = pool["adapters"]
    print(f"[hs-causal-ens] {len(adapters)} adapters, base={base_model}")

    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    try:
        from transformers.tokenization_utils_base import PreTrainedTokenizerBase
        if not hasattr(PreTrainedTokenizerBase, "all_special_tokens_extended"):
            PreTrainedTokenizerBase.all_special_tokens_extended = property(
                lambda self: self.all_special_tokens)
    except Exception:
        pass
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    ds = load_dataset("hellaswag", split="validation")
    if args.max_eval and args.max_eval < len(ds):
        ds = ds.shuffle(seed=0).select(range(args.max_eval))
    Nq = len(ds)
    K = 4

    # Build (Nq * K) prompt-completion concatenations: ctx + " " + ending.
    # Track ctx length so we can mask it out during scoring.
    items = []  # list of dicts with 'concat' (str), 'ctx_len_chars' (int), 'eg_idx', 'cand_idx'
    labels = []
    for eg_idx, ex in enumerate(ds):
        ctx = (ex.get("ctx") or "").strip()
        endings = list(ex.get("endings") or [])
        endings = (endings + [""] * K)[:K]
        labels.append(int(ex.get("label", 0) or 0))
        for cand_idx, end in enumerate(endings):
            end = (str(end) if end else "").strip()
            concat = f"{ctx} {end}".strip()
            items.append({"concat": concat, "ctx": ctx, "eg_idx": eg_idx, "cand_idx": cand_idx})
    labels = np.array(labels, dtype=np.int64)
    print(f"[hs-causal-ens] {Nq} questions, {len(items)} (q*K) prompts")

    llm = LLM(
        model=base_model, enable_lora=True,
        max_loras=len(adapters), max_lora_rank=args.max_lora_rank,
        max_model_len=args.max_model_len, gpu_memory_utilization=args.gpu_mem_util,
        dtype="bfloat16",
    )
    # prompt_logprobs=1 returns logprobs at every prompt position (the chosen token's logprob).
    sampling = SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=1)

    def adapter_dir(p: str) -> str:
        path = Path(p)
        if (path / "adapter_config.json").exists():
            return str(path)
        cands = list(path.rglob("adapter_config.json"))
        if not cands:
            raise SystemExit(f"no adapter_config.json under {p}")
        return str(cands[0].parent)

    lora_reqs = [
        LoRARequest(lora_name=f"a{i}", lora_int_id=i + 1, lora_path=adapter_dir(a["path"]))
        for i, a in enumerate(adapters)
    ]

    # Tokenize ctx-only once per example to know how many ctx tokens to skip.
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(base_model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    ctx_token_lens = []
    for ex in ds:
        ctx = (ex.get("ctx") or "").strip()
        ctx_token_lens.append(len(tok(ctx, add_special_tokens=False)["input_ids"]))

    M = len(lora_reqs)
    # Per-adapter, per-(question,candidate) log-likelihood of ending tokens.
    per_adapter_scores = np.zeros((M, Nq, K), dtype=np.float32)

    prompts_all = [it["concat"] for it in items]

    for ai, lreq in enumerate(lora_reqs):
        t0 = time.perf_counter()
        outs = llm.generate(prompts_all, sampling, lora_request=lreq)
        for out, it in zip(outs, items):
            # out.prompt_logprobs: list of {token_id: Logprob} per prompt position.
            # First entry is None (no logprob for first token). We sum over
            # positions after the ctx to get the ending log-likelihood.
            plp = out.prompt_logprobs or []
            tok_ids = out.prompt_token_ids
            ctx_len = ctx_token_lens[it["eg_idx"]]
            score = 0.0
            n_ending = 0
            for pos in range(ctx_len, len(plp)):
                d = plp[pos]
                if not d:
                    continue
                lp_obj = d.get(tok_ids[pos])
                if lp_obj is None:
                    continue
                lp_val = getattr(lp_obj, "logprob", None)
                if lp_val is None and isinstance(lp_obj, (int, float)):
                    lp_val = float(lp_obj)
                if lp_val is not None:
                    score += float(lp_val)
                    n_ending += 1
            # length-normalized score (mean per ending token)
            per_adapter_scores[ai, it["eg_idx"], it["cand_idx"]] = score / max(1, n_ending)
        per_adapter_picks = per_adapter_scores[ai].argmax(axis=1)
        em = float((per_adapter_picks == labels).mean())
        print(f"  [{ai+1}/{M}] {adapters[ai]['adapter_id'][:50]:<50s} acc={em:.4f}  ({time.perf_counter()-t0:.1f}s)")

    # === Aggregation ===
    per_adapter_picks = per_adapter_scores.argmax(axis=2)  # (M, Nq)
    per_adapter_acc = [float((per_adapter_picks[ai] == labels).mean()) for ai in range(M)]

    # majority_vote over picks
    maj_picks = np.zeros(Nq, dtype=np.int64)
    vote_fracs = np.zeros(Nq, dtype=np.float64)
    for j in range(Nq):
        cnt = Counter(per_adapter_picks[:, j].tolist())
        top, count = cnt.most_common(1)[0]
        maj_picks[j] = top
        vote_fracs[j] = count / M
    maj_acc = float((maj_picks == labels).mean())

    # soft_vote: average per-candidate softmax across adapters, then argmax
    def softmax(x, axis=-1):
        x = x - x.max(axis=axis, keepdims=True)
        e = np.exp(x); return e / e.sum(axis=axis, keepdims=True)
    per_adapter_softmax = softmax(per_adapter_scores, axis=-1)  # (M, Nq, K)
    soft_probs = per_adapter_softmax.mean(axis=0)  # (Nq, K)
    soft_picks = soft_probs.argmax(axis=1)
    soft_acc = float((soft_picks == labels).mean())

    # logit_avg: average raw scores then argmax
    avg_scores = per_adapter_scores.mean(axis=0)  # (Nq, K)
    la_picks = avg_scores.argmax(axis=1)
    la_acc = float((la_picks == labels).mean())

    best_idx = int(np.argmax(per_adapter_acc))
    best_acc = float(per_adapter_acc[best_idx])

    # Calibration: 15-bin equal-mass on (max soft prob, correct) for soft_vote
    soft_conf = soft_probs.max(axis=1)
    soft_corr = (soft_picks == labels).astype(np.float64)
    order = np.argsort(soft_conf)
    bins = np.array_split(order, 15)
    ece, mce = 0.0, 0.0; bin_data = []
    for b in bins:
        if len(b) == 0:
            bin_data.append({"count": 0, "mean_conf": 0.0, "mean_acc": 0.0}); continue
        mc = float(soft_conf[b].mean()); ma = float(soft_corr[b].mean()); gap = abs(mc - ma)
        ece += (len(b) / Nq) * gap; mce = max(mce, gap)
        bin_data.append({"count": int(len(b)), "mean_conf": mc, "mean_acc": ma})
    brier = float(np.mean((soft_conf - soft_corr) ** 2))
    eps = 1e-12
    nll = float(-np.mean(np.log(np.clip(np.where(soft_corr == 1, soft_conf, 1 - soft_conf), eps, 1.0))))

    out = {
        "pool_id": pool.get("pool_id"),
        "task": "hellaswag_causal",
        "n_adapters": M,
        "n_eval": Nq,
        "per_adapter_acc": per_adapter_acc,
        "best_single_acc": best_acc,
        "best_single_idx": best_idx,
        "majority_vote_acc": maj_acc,
        "soft_vote_acc": soft_acc,
        "logit_avg_acc": la_acc,
        "soft_vote_calibration": {
            "ece_15bin": float(ece), "mce_15bin": float(mce),
            "brier": brier, "nll": nll, "reliability_bins": bin_data,
            "mean_confidence": float(soft_conf.mean()),
        },
        "vote_fractions": vote_fracs.tolist(),
        "labels": labels.tolist(),
        "provenance": provenance({k: str(v) for k, v in vars(args).items()}),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(f"\n=== HellaSwag causal ensemble summary ===")
    print(f"  best_single   acc = {best_acc:.4f}  (idx={best_idx})")
    print(f"  mean per-ad   acc = {np.mean(per_adapter_acc):.4f}")
    print(f"  majority_vote acc = {maj_acc:.4f}")
    print(f"  soft_vote     acc = {soft_acc:.4f}  ECE={ece:.4f}  Brier={brier:.4f}")
    print(f"  logit_avg     acc = {la_acc:.4f}")
    print(f"  → {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
