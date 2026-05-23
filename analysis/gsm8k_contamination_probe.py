#!/usr/bin/env python3
"""GSM8K contamination probe (R5 audit request).

Compare a Qwen-2.5-7B adapter's accuracy on:
  (a) original GSM8K test
  (b) digit-shifted GSM8K test (add +7 to every integer in question + answer)

If accuracy drops dramatically on (b), the model is likely relying on
memorized GSM8K items rather than genuine arithmetic reasoning.

Usage:
    CUDA_VISIBLE_DEVICES=1 python3 analysis/gsm8k_contamination_probe.py \\
        --adapter pool_a_gsm8k_qwen25_7b/seed_11/ \\
        --base-model Qwen/Qwen2.5-7B \\
        --n-samples 200 \\
        --out analysis/gsm8k_contamination_probe.json
"""
from __future__ import annotations
import argparse, json, re
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def shift_digits(text: str, shift: int = 7) -> str:
    """Replace every integer in the text with (integer + shift)."""
    def _shift(m):
        n = int(m.group(0))
        return str(n + shift)
    return re.sub(r"\d+", _shift, text)


def extract_answer(text: str) -> int | None:
    """Extract final integer answer from generation."""
    m = re.findall(r"-?\d+", text.replace(",", ""))
    return int(m[-1]) if m else None


def gold_answer(answer_field: str) -> int:
    """GSM8K gold answer format: '...#### 42'"""
    m = re.search(r"####\s*(-?\d+)", answer_field.replace(",", ""))
    return int(m.group(1)) if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True, type=Path)
    ap.add_argument("--base-model", required=True)
    ap.add_argument("--n-samples", type=int, default=200)
    ap.add_argument("--shift", type=int, default=7)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--max-new-tokens", type=int, default=200)
    args = ap.parse_args()

    print(f"[probe] loading {args.base_model} + adapter {args.adapter}")
    tok = AutoTokenizer.from_pretrained(args.base_model)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=torch.bfloat16, device_map="cuda")
    model = PeftModel.from_pretrained(base, args.adapter)
    model.eval()

    ds = load_dataset("openai/gsm8k", "main", split="test").shuffle(seed=0).select(range(args.n_samples))
    print(f"[probe] {len(ds)} test questions, shift={args.shift}")

    correct_orig = correct_shifted = 0
    examples = []
    for i, ex in enumerate(ds):
        q_orig = ex["question"]
        q_shift = shift_digits(q_orig, shift=args.shift)
        gold_orig = gold_answer(ex["answer"])
        gold_shift = gold_answer(shift_digits(ex["answer"], shift=args.shift))

        for q, gold, key in [(q_orig, gold_orig, "orig"), (q_shift, gold_shift, "shift")]:
            prompt = f"Question: {q}\nAnswer:"
            inp = tok(prompt, return_tensors="pt").to("cuda")
            with torch.no_grad():
                out = model.generate(**inp, max_new_tokens=args.max_new_tokens,
                                     do_sample=False, pad_token_id=tok.pad_token_id)
            gen = tok.decode(out[0][inp.input_ids.shape[1]:], skip_special_tokens=True)
            pred = extract_answer(gen)
            ok = (pred is not None and pred == gold)
            if key == "orig":
                correct_orig += int(ok)
                eg = {"i": i, "q_orig": q_orig[:80], "gold_orig": gold_orig, "pred_orig": pred, "ok_orig": ok}
            else:
                correct_shifted += int(ok)
                eg["q_shift"] = q_shift[:80]; eg["gold_shift"] = gold_shift
                eg["pred_shift"] = pred; eg["ok_shift"] = ok
                examples.append(eg)
        if (i+1) % 25 == 0:
            print(f"  {i+1}/{len(ds)}  orig_acc={correct_orig/(i+1):.3f}  shift_acc={correct_shifted/(i+1):.3f}")

    n = len(ds)
    res = {
        "adapter": str(args.adapter),
        "base_model": args.base_model,
        "n_samples": n,
        "shift": args.shift,
        "acc_orig": correct_orig / n,
        "acc_shift": correct_shifted / n,
        "delta_acc": (correct_orig - correct_shifted) / n,
        "examples_first10": examples[:10],
    }
    args.out.write_text(json.dumps(res, indent=2))
    print(f"\n=== Contamination probe ===")
    print(f"  Original GSM8K test (n={n}):  acc = {res['acc_orig']:.4f}")
    print(f"  Digit-shifted (+{args.shift}): acc = {res['acc_shift']:.4f}")
    print(f"  Δ = {res['delta_acc']:+.4f}")
    print(f"  Interpretation: large positive Δ suggests memorization; small Δ suggests genuine reasoning.")
    print(f"  Wrote {args.out}")


if __name__ == "__main__":
    raise SystemExit(main())
