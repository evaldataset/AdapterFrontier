#!/usr/bin/env python3
"""LoRA fine-tuning for HellaSwag with decoder LMs (Qwen, Llama, etc.).

AutoModelForMultipleChoice doesn't support causal-LM architectures, so we
recast HellaSwag as a completion task: prompt = ctx, target = endings[label].
Loss is computed only on the target tokens. At eval time the per-candidate
log-likelihood is the standard scoring rule (lm-evaluation-harness convention).

Usage:
    python3 real_lora_train_hellaswag_causal.py \\
        --model Qwen/Qwen2.5-0.5B \\
        --max-samples 39000 --max-eval-samples 1000 \\
        --num-epochs 3 --lr 2e-4 --batch-size 4 --seq-length 256 \\
        --lora-r 8 --lora-alpha 16 --lora-target q_proj v_proj \\
        --seed 42 --output-dir ./pool_a_hellaswag_qwen_causal/seed_42
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from peft import get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from peft_utils import build_peft_config, detect_lora_targets, provenance


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--output-dir", default=".")
    p.add_argument("--max-samples", type=int, default=10000)
    p.add_argument("--max-eval-samples", type=int, default=500)
    p.add_argument("--num-epochs", type=float, default=3.0)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--seq-length", type=int, default=256)
    p.add_argument("--lora-r", type=int, default=8)
    p.add_argument("--lora-alpha", type=int, default=16)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--lora-target", nargs="*", default=None)
    p.add_argument("--peft-method", default="lora",
                   choices=["lora", "dora", "pissa", "rslora"])
    p.add_argument("--pissa-niter", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--warmup-ratio", type=float, default=0.06)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--precision", choices=["fp16", "bf16", "fp32"], default="bf16")
    return p.parse_args()


def format_example(ex: dict) -> dict:
    """ctx + 4 endings → (prompt, target) of correct ending."""
    ctx = ex.get("ctx") or ""
    endings = list(ex.get("endings") or [])
    label = int(ex.get("label", 0) or 0)
    if not (0 <= label < len(endings)):
        label = 0
    target = endings[label] if endings else ""
    return {"prompt": ctx.strip(), "target": str(target).strip()}


def main() -> int:
    args = parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    print(f"[mc-causal] model={args.model}")

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    raw_train = load_dataset("hellaswag", split="train")
    raw_eval = load_dataset("hellaswag", split="validation")
    if args.max_samples and args.max_samples < len(raw_train):
        raw_train = raw_train.shuffle(seed=args.seed).select(range(args.max_samples))
    if args.max_eval_samples and args.max_eval_samples < len(raw_eval):
        raw_eval = raw_eval.shuffle(seed=args.seed).select(range(args.max_eval_samples))

    raw_train = raw_train.map(format_example)
    raw_eval = raw_eval.map(format_example)

    def tokenize(batch):
        # Concatenate prompt + " " + target. Loss masked on prompt tokens.
        full_input_ids, attn_masks, labels = [], [], []
        for p, t in zip(batch["prompt"], batch["target"]):
            p_ids = tok(p, add_special_tokens=False)["input_ids"]
            t_ids = tok(" " + t, add_special_tokens=False)["input_ids"]
            ids = (p_ids + t_ids)[: args.seq_length]
            mask = [1] * len(ids)
            label = [-100] * len(p_ids) + t_ids
            label = label[: args.seq_length]
            # pad-on-the-right to seq_length
            pad_len = args.seq_length - len(ids)
            ids = ids + [tok.pad_token_id] * pad_len
            mask = mask + [0] * pad_len
            label = label + [-100] * pad_len
            full_input_ids.append(ids); attn_masks.append(mask); labels.append(label)
        return {"input_ids": full_input_ids, "attention_mask": attn_masks, "labels": labels}

    train_ds = raw_train.map(tokenize, batched=True, remove_columns=raw_train.column_names)
    eval_ds = raw_eval.map(tokenize, batched=True, remove_columns=raw_eval.column_names)
    print(f"[mc-causal] train={len(train_ds)} eval={len(eval_ds)}")

    base = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16 if args.precision == "bf16" else (torch.float16 if args.precision == "fp16" else None),
    )
    targets = args.lora_target or detect_lora_targets(base)
    print(f"[mc-causal] LoRA targets: {targets}")
    cfg = build_peft_config(
        peft_method=args.peft_method, lora_r=args.lora_r, lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout, target_modules=targets,
        task_type="CAUSAL_LM", pissa_niter=args.pissa_niter,
    )
    model = get_peft_model(base, cfg)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"[mc-causal] trainable={n_train:,} / total={n_total:,} ({100*n_train/n_total:.3f}%)")

    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    targs = TrainingArguments(
        output_dir=str(out_dir / "_trainer"),
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        eval_strategy="epoch", save_strategy="no",
        logging_steps=50,
        bf16=args.precision == "bf16", fp16=args.precision == "fp16",
        report_to=[], seed=args.seed,
        remove_unused_columns=False,
    )
    trainer = Trainer(model=model, args=targs,
                      train_dataset=train_ds, eval_dataset=eval_ds,
                      processing_class=tok)
    t0 = time.perf_counter()
    trainer.train()
    train_s = time.perf_counter() - t0
    metrics = trainer.evaluate()

    model.save_pretrained(str(out_dir))
    tok.save_pretrained(str(out_dir))

    flat = {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))}
    flat["train_seconds"] = float(train_s)
    flat["trainable_params"] = int(n_train)
    summary = {
        "model": args.model, "task": "hellaswag_causal",
        "n_train": len(train_ds), "n_eval": len(eval_ds),
        "metrics": flat, "config": vars(args),
        "provenance": provenance({k: str(v) for k, v in vars(args).items()}),
    }
    (out_dir / "training_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[mc-causal] done. eval_loss={metrics.get('eval_loss',0):.4f}  "
          f"train_s={train_s:.1f}  → {out_dir}")
    print(json.dumps(flat, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
