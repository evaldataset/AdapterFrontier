#!/usr/bin/env python3
"""LoRA fine-tuning for multiple-choice tasks (HellaSwag-style).

Mirrors the API surface of real_lora_classify.py but uses
AutoModelForMultipleChoice. Each example is a context plus N candidate
endings; the model scores each (context, ending) pair and picks argmax.

Usage:
    python3 real_lora_multichoice.py \
        --model bert-base-uncased --dataset hellaswag \
        --lora-r 8 --num-epochs 3 --output-dir adapters/...
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from datasets import load_dataset
from peft import get_peft_model
from transformers import (
    AutoModelForMultipleChoice,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from peft_utils import build_peft_config, detect_lora_targets, provenance


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LoRA fine-tuning for multi-choice tasks")
    p.add_argument("--model", required=True)
    p.add_argument("--dataset", required=True, help="HF dataset ID (e.g. hellaswag)")
    p.add_argument("--dataset-config", default=None)
    p.add_argument("--train-split", default="train")
    p.add_argument("--eval-split", default="validation")
    p.add_argument("--output-dir", default=".")
    p.add_argument("--context-field", default="ctx",
                   help="Field holding the context text (default ctx for HellaSwag)")
    p.add_argument("--endings-field", default="endings",
                   help="Field holding the list of candidate endings")
    p.add_argument("--label-field", default="label")
    p.add_argument("--n-choices", type=int, default=4)
    p.add_argument("--max-samples", type=int, default=10000)
    p.add_argument("--max-eval-samples", type=int, default=1000)
    p.add_argument("--num-epochs", type=float, default=3.0)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--seq-length", type=int, default=192)
    p.add_argument("--lora-r", type=int, default=8)
    p.add_argument("--lora-alpha", type=int, default=16)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--lora-target", nargs="*", default=None)
    p.add_argument("--peft-method", default="lora",
                   choices=["lora", "dora", "pissa", "rslora"])
    p.add_argument("--pissa-niter", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--warmup-ratio", type=float, default=0.05)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--precision", choices=["fp16", "bf16", "fp32"], default="bf16")
    return p.parse_args()


@dataclass
class MultiChoiceCollator:
    """Pads a batch of (n_choices, seq_len) tensors per example."""
    tokenizer: Any
    n_choices: int = 4

    def __call__(self, features: list[dict]) -> dict:
        labels = torch.tensor([f["label"] for f in features], dtype=torch.long)
        # Each feature has input_ids/attention_mask of shape (n_choices, seq_len)
        # — flatten into (B*n_choices, seq_len), pad, then reshape back.
        B = len(features)
        flat: list[dict] = []
        for f in features:
            for i in range(self.n_choices):
                flat.append({
                    "input_ids": f["input_ids"][i],
                    "attention_mask": f["attention_mask"][i],
                })
        padded = self.tokenizer.pad(flat, return_tensors="pt")
        out = {
            "input_ids": padded["input_ids"].view(B, self.n_choices, -1),
            "attention_mask": padded["attention_mask"].view(B, self.n_choices, -1),
            "labels": labels,
        }
        if "token_type_ids" in padded:
            out["token_type_ids"] = padded["token_type_ids"].view(B, self.n_choices, -1)
        return out


def make_preprocess(tokenizer, args):
    def _pp(batch):
        ctxs = batch[args.context_field]
        endings_lists = batch[args.endings_field]
        labels = batch[args.label_field]
        first: list[str] = []
        second: list[str] = []
        for ctx, endings in zip(ctxs, endings_lists):
            ctx = ctx if isinstance(ctx, str) else ""
            ends = list(endings) if endings else [""] * args.n_choices
            ends = (ends + [""] * args.n_choices)[: args.n_choices]
            for end in ends:
                first.append(ctx)
                second.append(end if isinstance(end, str) else "")
        toks = tokenizer(
            first, second,
            truncation=True, max_length=args.seq_length, padding=False,
        )
        N = len(ctxs)
        out = {"label": [int(x) for x in labels]}
        for k, v in toks.items():
            out[k] = [v[i * args.n_choices : (i + 1) * args.n_choices] for i in range(N)]
        return out
    return _pp


def compute_metrics(eval_pred):
    preds, labels = eval_pred
    if isinstance(preds, tuple):
        preds = preds[0]
    preds = np.asarray(preds)
    labels = np.asarray(labels)
    pred_ids = preds.argmax(axis=-1)
    return {"accuracy": float((pred_ids == labels).mean())}


def main() -> int:
    args = parse_args()
    # Determinism: pin RNGs and cuDNN heuristics for bit-exact training reproducibility.
    import os as _os
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    _os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    print(f"[mc] model={args.model} dataset={args.dataset} cfg={args.dataset_config}")
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    raw_train = load_dataset(args.dataset, args.dataset_config, split=args.train_split)
    raw_eval = load_dataset(args.dataset, args.dataset_config, split=args.eval_split)
    if args.max_samples and args.max_samples < len(raw_train):
        raw_train = raw_train.shuffle(seed=args.seed).select(range(args.max_samples))
    if args.max_eval_samples and args.max_eval_samples < len(raw_eval):
        raw_eval = raw_eval.shuffle(seed=args.seed).select(range(args.max_eval_samples))

    # HellaSwag has labels stored as strings — coerce.
    def _coerce_label(ex):
        ex["label"] = int(ex.get(args.label_field, 0) or 0)
        return ex
    raw_train = raw_train.map(_coerce_label)
    raw_eval = raw_eval.map(_coerce_label)

    pp = make_preprocess(tok, args)
    train_ds = raw_train.map(pp, batched=True, remove_columns=raw_train.column_names)
    eval_ds = raw_eval.map(pp, batched=True, remove_columns=raw_eval.column_names)

    print(f"[mc] train={len(train_ds)} eval={len(eval_ds)}")

    base = AutoModelForMultipleChoice.from_pretrained(args.model)
    targets = args.lora_target or detect_lora_targets(base)
    print(f"[mc] LoRA targets: {targets}")
    cfg = build_peft_config(
        peft_method=args.peft_method,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=targets,
        task_type="SEQ_CLS",  # PEFT has no MULTIPLE_CHOICE; SEQ_CLS hooks the same backbone weights.
        pissa_niter=args.pissa_niter,
    )
    model = get_peft_model(base, cfg)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"[mc] trainable={n_train:,} / total={n_total:,} ({100*n_train/n_total:.3f}%)")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fp16 = args.precision == "fp16"
    bf16 = args.precision == "bf16"
    targs = TrainingArguments(
        output_dir=str(out_dir / "_trainer"),
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        eval_strategy="epoch",
        save_strategy="no",
        logging_steps=50,
        fp16=fp16, bf16=bf16,
        report_to=[],
        seed=args.seed,
        remove_unused_columns=False,
    )
    collator = MultiChoiceCollator(tokenizer=tok, n_choices=args.n_choices)
    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tok,
        data_collator=collator,
        compute_metrics=compute_metrics,
    )
    t0 = time.perf_counter()
    trainer.train()
    train_s = time.perf_counter() - t0
    metrics = trainer.evaluate()

    model.save_pretrained(str(out_dir))
    tok.save_pretrained(str(out_dir))

    summary = {
        "model": args.model,
        "dataset": args.dataset,
        "dataset_config": args.dataset_config,
        "n_train": len(train_ds),
        "n_eval": len(eval_ds),
        "train_seconds": train_s,
        "metrics": {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))},
        "trainable_params": int(n_train),
        "config": vars(args),
        "provenance": provenance({k: str(v) for k, v in vars(args).items()}),
    }
    (out_dir / "training_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[mc] done. eval_accuracy={metrics.get('eval_accuracy', 0.0):.4f}  "
          f"train_s={train_s:.1f}  → {out_dir}")
    # sweep_runner parses the LAST JSON object in stdout for the optimize key.
    flat_metrics = {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))}
    flat_metrics["train_seconds"] = float(train_s)
    flat_metrics["trainable_params"] = int(n_train)
    print(json.dumps(flat_metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
