#!/usr/bin/env python3
"""GSM8K ensemble eval using vLLM multi-LoRA serving.

For each test question, generates an answer from each adapter in the pool,
extracts the integer answer (regex), then aggregates:

  - majority_vote: mode of integer answers (confidence = fraction agreeing)
  - best_single: best per-adapter exact-match (single-adapter baseline)
  - per_adapter_em: mean exact-match across all adapters

Reports accuracy and a vote-fraction-based calibration check.

Usage:
    python3 ensemble_eval_gsm8k.py \\
        --pool pools/pool_a_gsm8k_qwen25_05b.json \\
        --out ensemble_results/pool_a_gsm8k_qwen25_05b.json \\
        --max-eval 200
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
from datasets import load_dataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
from peft_utils import provenance


def extract_answer(text: str) -> str:
    m = re.search(r"####\s*([-+]?\d[\d,]*)", text)
    if m:
        return m.group(1).replace(",", "")
    nums = re.findall(r"[-+]?\d[\d,]*", text)
    return nums[-1].replace(",", "") if nums else ""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--max-eval", type=int, default=200)
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--max-lora-rank", type=int, default=16)
    ap.add_argument("--gpu-mem-util", type=float, default=0.30)
    args = ap.parse_args()

    pool = json.loads(args.pool.read_text())
    base_model = pool.get("base_model") or pool["adapters"][0].get("base_model")
    adapters = pool["adapters"]
    print(f"[gsm8k-ens] {len(adapters)} adapters, base={base_model}")

    # transformers 5.x compat patch (also active in vllm/transformers_utils/tokenizer.py)
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

    ds = load_dataset("openai/gsm8k", "main", split="test")
    if args.max_eval and args.max_eval < len(ds):
        ds = ds.shuffle(seed=0).select(range(args.max_eval))
    questions = [str(x).strip() for x in ds["question"]]
    golds = [extract_answer(str(x)) for x in ds["answer"]]
    prompts = [f"Question: {q}\nAnswer:" for q in questions]
    Nq = len(prompts)
    print(f"[gsm8k-ens] {Nq} test questions")

    llm = LLM(
        model=base_model, enable_lora=True,
        max_loras=len(adapters), max_lora_rank=args.max_lora_rank,
        max_model_len=512, gpu_memory_utilization=args.gpu_mem_util,
        dtype="bfloat16",
    )
    sampling = SamplingParams(temperature=0.0, max_tokens=args.max_new_tokens,
                              logprobs=1)

    # Resolve adapter paths (peft inner dir)
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

    # Per-adapter generation — extract integer answer + per-token logprob
    # confidence per (adapter, question). vLLM returns logprobs=1 → for each
    # generated token, the chosen token's logprob is in
    # output.outputs[0].logprobs[step][token_id].logprob.
    per_adapter_preds = []  # (M, Nq) of strings
    per_adapter_confs = []  # (M, Nq) of float in [0,1]
    per_adapter_em = []
    for i, lreq in enumerate(lora_reqs):
        t0 = time.perf_counter()
        outs = llm.generate(prompts, sampling, lora_request=lreq)
        preds = []
        confs = []
        for o in outs:
            gen = o.outputs[0]
            preds.append(extract_answer(gen.text))
            # confidence = exp(mean per-token logprob) — geometric mean of
            # per-token max-softmax probability across the full generation.
            lps = []
            if gen.logprobs is not None:
                for step_dict, tok_id in zip(gen.logprobs, gen.token_ids):
                    lp_obj = step_dict.get(tok_id) if step_dict else None
                    if lp_obj is not None:
                        lp_val = getattr(lp_obj, "logprob", None)
                        if lp_val is None and isinstance(lp_obj, (int, float)):
                            lp_val = float(lp_obj)
                        if lp_val is not None:
                            lps.append(float(lp_val))
            confs.append(float(np.exp(np.mean(lps))) if lps else 0.0)
        per_adapter_preds.append(preds)
        per_adapter_confs.append(confs)
        em = float(sum(p == g and g != "" for p, g in zip(preds, golds))) / Nq
        mean_conf = float(np.mean(confs)) if confs else 0.0
        per_adapter_em.append(em)
        print(f"  [{i+1}/{len(lora_reqs)}] {adapters[i]['adapter_id'][:50]:<50s} "
              f"EM={em:.4f}  conf={mean_conf:.3f}  ({time.perf_counter()-t0:.1f}s)")

    M = len(per_adapter_preds)
    preds_array = np.array(per_adapter_preds, dtype=object)  # (M, Nq)

    # Majority vote
    maj_preds = []
    vote_fractions = []
    for j in range(Nq):
        col = [preds_array[i, j] for i in range(M)]
        cnt = Counter(col)
        top, count = cnt.most_common(1)[0]
        maj_preds.append(top)
        vote_fractions.append(count / M)

    maj_em = float(sum(p == g and g != "" for p, g in zip(maj_preds, golds))) / Nq
    best_single_idx = int(np.argmax(per_adapter_em))
    best_single_em = float(per_adapter_em[best_single_idx])
    mean_em = float(np.mean(per_adapter_em))

    # Vote-fraction "calibration": for high-vote-fraction predictions (≥0.8),
    # what fraction is correct? bin into 5 bins and report ECE-like number.
    correct = np.array([1 if p == g and g != "" else 0 for p, g in zip(maj_preds, golds)])
    vfracs = np.array(vote_fractions)
    bins = np.linspace(1.0 / M, 1.0001, 6)
    bin_data = []
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (vfracs >= lo) & (vfracs < hi)
        n = int(mask.sum())
        if n == 0:
            bin_data.append({"lo": float(lo), "hi": float(hi), "count": 0, "mean_conf": 0.0, "mean_acc": 0.0})
            continue
        mc = float(vfracs[mask].mean())
        ma = float(correct[mask].mean())
        ece += (n / Nq) * abs(mc - ma)
        bin_data.append({"lo": float(lo), "hi": float(hi), "count": n, "mean_conf": mc, "mean_acc": ma})

    # === Per-token logprob calibration (rigorous ECE) ===
    # Per-adapter: confidence = exp(mean per-token logprob) over generation.
    # Per-example correct = (extracted answer == gold).
    confs_arr = np.array(per_adapter_confs, dtype=np.float64)  # (M, Nq)
    correct_per_adapter = np.zeros((M, Nq), dtype=np.int64)
    for ai in range(M):
        for j in range(Nq):
            correct_per_adapter[ai, j] = int(per_adapter_preds[ai][j] == golds[j] and golds[j] != "")

    def _ece_15bin(conf, corr, n_bins=15):
        order = np.argsort(conf)
        groups = np.array_split(order, n_bins)
        e, m_max = 0.0, 0.0
        bins_out = []
        for g in groups:
            if len(g) == 0:
                bins_out.append({"count": 0, "mean_conf": 0.0, "mean_acc": 0.0})
                continue
            mc = float(conf[g].mean()); ma = float(corr[g].mean()); gap = abs(mc - ma)
            e += (len(g) / len(conf)) * gap
            m_max = max(m_max, gap)
            bins_out.append({"count": int(len(g)), "mean_conf": mc, "mean_acc": ma})
        return e, m_max, bins_out

    eps = 1e-12
    per_adapter_calibration = []
    for ai in range(M):
        c = confs_arr[ai]; r = correct_per_adapter[ai]
        ece_a, mce_a, bins_a = _ece_15bin(c, r, 15)
        brier_a = float(np.mean((c - r) ** 2))
        nll_a = float(-np.mean(np.log(np.clip(np.where(r == 1, c, 1 - c), eps, 1.0))))
        per_adapter_calibration.append({
            "adapter_id": adapters[ai]["adapter_id"],
            "ece_15bin_logprob": float(ece_a),
            "mce_15bin_logprob": float(mce_a),
            "brier_logprob": brier_a,
            "nll_logprob": nll_a,
            "reliability_bins_logprob": bins_a,
        })

    # Best-single (by per-adapter EM) calibration
    best_single_conf = confs_arr[best_single_idx]
    best_single_corr = correct_per_adapter[best_single_idx]
    bs_ece, bs_mce, bs_bins = _ece_15bin(best_single_conf, best_single_corr, 15)
    bs_brier = float(np.mean((best_single_conf - best_single_corr) ** 2))
    bs_nll = float(-np.mean(np.log(np.clip(
        np.where(best_single_corr == 1, best_single_conf, 1 - best_single_conf),
        eps, 1.0))))

    # Ensemble calibration: confidence = mean of per-adapter confidences for the
    # *majority-voted* answer (so that high-agreement, high-confidence cases
    # both contribute to a high ensemble confidence). correct = majority correct.
    ens_conf = np.zeros(Nq, dtype=np.float64)
    for j in range(Nq):
        # mean confidence across adapters that voted for the majority answer
        agreeing = [confs_arr[ai, j] for ai in range(M) if per_adapter_preds[ai][j] == maj_preds[j]]
        ens_conf[j] = float(np.mean(agreeing)) if agreeing else 0.0
    ens_corr = correct.astype(np.int64)
    ens_ece, ens_mce, ens_bins = _ece_15bin(ens_conf, ens_corr, 15)
    ens_brier = float(np.mean((ens_conf - ens_corr) ** 2))
    ens_nll = float(-np.mean(np.log(np.clip(
        np.where(ens_corr == 1, ens_conf, 1 - ens_conf), eps, 1.0))))

    out = {
        "pool_id": pool.get("pool_id"),
        "task": "gsm8k",
        "n_adapters": M,
        "n_eval": Nq,
        "per_adapter_em": per_adapter_em,
        "best_single_em": best_single_em,
        "best_single_idx": best_single_idx,
        "majority_vote_em": maj_em,
        "mean_em": mean_em,
        "ece_5bin_votefrac": float(ece),
        "vote_fraction_bins": bin_data,
        "majority_predictions": maj_preds,
        "vote_fractions": vote_fractions,
        "golds": golds,
        # === per-token logprob calibration (rigorous; replaces vote-fraction proxy) ===
        "per_token_logprob_calibration": {
            "best_single": {
                "ece_15bin": float(bs_ece),
                "mce_15bin": float(bs_mce),
                "brier": bs_brier,
                "nll": bs_nll,
                "reliability_bins": bs_bins,
                "mean_confidence": float(best_single_conf.mean()),
            },
            "ensemble_majority": {
                "ece_15bin": float(ens_ece),
                "mce_15bin": float(ens_mce),
                "brier": ens_brier,
                "nll": ens_nll,
                "reliability_bins": ens_bins,
                "mean_confidence": float(ens_conf.mean()),
            },
            "per_adapter": per_adapter_calibration,
            "per_adapter_confidence": confs_arr.tolist(),
            "per_adapter_correct": correct_per_adapter.tolist(),
            "ensemble_confidence_test": ens_conf.tolist(),
        },
        "provenance": provenance({k: str(v) for k, v in vars(args).items()}),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))

    print(f"\n=== GSM8K ensemble summary ===")
    print(f"  best_single  EM = {best_single_em:.4f}  (seed_idx={best_single_idx})")
    print(f"  mean per-ad  EM = {mean_em:.4f}")
    print(f"  majority_vote EM = {maj_em:.4f}")
    print(f"  vote-frac ECE       = {ece:.4f}")
    print(f"  per-token ECE single = {bs_ece:.4f}  (Brier {bs_brier:.4f}, NLL {bs_nll:.4f})")
    print(f"  per-token ECE ensemble = {ens_ece:.4f}  (Brier {ens_brier:.4f}, NLL {ens_nll:.4f})")
    print(f"  → {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
