#!/usr/bin/env python3
"""Generic LoRA fine-tuning for Hugging Face sequence classification tasks.

Designed to match the style of `real_lora_train.py` while supporting
single-sentence and sentence-pair classification datasets.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from peft_utils import provenance


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generic LoRA fine-tuning for sequence classification"
    )
    p.add_argument("--model", required=True, help="HF model ID")
    p.add_argument("--dataset", required=True, help="HF dataset ID")
    p.add_argument("--dataset-config", default=None)
    p.add_argument("--train-split", default="train")
    p.add_argument("--eval-split", default="validation")
    p.add_argument("--output-dir", default=".")
    p.add_argument("--text-field-a", default=None)
    p.add_argument("--text-field-b", default=None)
    p.add_argument("--label-field", default="label")
    p.add_argument("--metric", default="accuracy", choices=["accuracy", "f1"])
    p.add_argument("--max-samples", type=int, default=10000)
    p.add_argument("--max-eval-samples", type=int, default=1000)
    p.add_argument("--num-epochs", type=float, default=1.0)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--seq-length", type=int, default=256)
    p.add_argument("--lora-r", type=int, default=8)
    p.add_argument("--lora-alpha", type=int, default=16)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--lora-target", nargs="*", default=None)
    p.add_argument(
        "--peft-method",
        default="lora",
        choices=["lora", "dora", "pissa", "rslora"],
        help="PEFT variant to apply",
    )
    p.add_argument(
        "--pissa-niter",
        type=int,
        default=0,
        help="If > 0, use pissa_niter_<N> initialization",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--warmup-ratio", type=float, default=0.05)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--no-fp16", action="store_true",
                   help="Legacy: disable fp16. Equivalent to --precision fp32.")
    p.add_argument("--precision", choices=["fp16", "bf16", "fp32"], default=None,
                   help="Mixed-precision training. Overrides --no-fp16 if both are passed. "
                        "DeBERTa-v3 + LoRA + fp16 hits a grad-scaler bug — use bf16 instead.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    # Determinism: pin RNGs and cuDNN heuristics for bit-exact training reproducibility.
    import os as _os
    torch.manual_seed(args.seed)
    try:
        import numpy as _np
        _np.random.seed(args.seed)
    except Exception:
        pass
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    _os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"Loading dataset: {args.dataset}"
        + (f" ({args.dataset_config})" if args.dataset_config else "")
    )
    ds_kwargs: dict[str, object] = {
        "path": args.dataset,
        "split": args.train_split,
    }
    if args.dataset_config:
        ds_kwargs["name"] = args.dataset_config

    raw_train = load_dataset(**ds_kwargs)
    train_ds = raw_train.select(range(min(args.max_samples, len(raw_train))))

    try:
        eval_kwargs = {**ds_kwargs, "split": args.eval_split}
        raw_eval = load_dataset(**eval_kwargs)
        eval_ds = raw_eval.select(range(min(args.max_eval_samples, len(raw_eval))))
    except (ValueError, FileNotFoundError, KeyError) as exc:
        print(f"WARNING: eval split '{args.eval_split}' not found ({exc}); holding out from train data")
        split = raw_train.train_test_split(test_size=min(args.max_eval_samples, len(raw_train) // 5), seed=args.seed)
        train_ds = split["train"].select(range(min(args.max_samples, len(split["train"]))))
        eval_ds = split["test"]

    text_field_a, text_field_b = _resolve_text_fields(args, train_ds)
    label_field = args.label_field
    label_list = _resolve_label_list(train_ds, eval_ds, label_field)
    num_labels = len(label_list)

    print(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        num_labels=num_labels,
    )
    if model.config.pad_token_id is None and tokenizer.pad_token_id is not None:
        model.config.pad_token_id = tokenizer.pad_token_id

    trainable_params_pre = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(
        f"  Params: {total_params / 1e6:.1f}M total, {trainable_params_pre / 1e6:.1f}M trainable"
    )

    print(
        f"Applying {args.peft_method.upper()}: "
        f"r={args.lora_r}, alpha={args.lora_alpha}, dropout={args.lora_dropout}"
    )
    target_modules = args.lora_target
    if target_modules is None:
        target_modules = _detect_lora_targets(model)
    print(f"  Target modules: {target_modules}")

    peft_config = _build_peft_config(args, target_modules)
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    print(f"  Train: {len(train_ds)} rows, Eval: {len(eval_ds)} rows")

    def tokenize(batch: dict[str, list[object]]) -> dict[str, object]:
        text_a = [_normalize_text(v) for v in batch[text_field_a]]
        if text_field_b is None:
            result = tokenizer(
                text_a,
                truncation=True,
                max_length=args.seq_length,
            )
        else:
            text_b = [_normalize_text(v) for v in batch[text_field_b]]
            result = tokenizer(
                text_a,
                text_b,
                truncation=True,
                max_length=args.seq_length,
            )
        result["labels"] = [int(v) for v in batch[label_field]]
        return result

    print("Tokenizing...")
    train_ds = train_ds.map(
        tokenize, batched=True, remove_columns=train_ds.column_names
    )
    eval_ds = eval_ds.map(tokenize, batched=True, remove_columns=eval_ds.column_names)

    # Precision resolution: --precision takes priority; --no-fp16 is legacy fallback.
    if args.precision is not None:
        precision = args.precision
    else:
        precision = "fp32" if args.no_fp16 else "fp16"
    if not torch.cuda.is_available():
        precision = "fp32"
    use_fp16 = (precision == "fp16")
    use_bf16 = (precision == "bf16")
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        num_train_epochs=args.num_epochs,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        logging_steps=25,
        save_strategy="no",
        eval_strategy="epoch",
        report_to=[],
        fp16=use_fp16,
        bf16=use_bf16,
        remove_unused_columns=False,
        dataloader_pin_memory=False,
        seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=lambda pred: _compute_metrics(pred, args.metric),
    )

    t0 = time.time()
    print("Training...")
    train_result = trainer.train()
    train_wall = time.time() - t0

    print("Evaluating...")
    eval_result = trainer.evaluate()
    trainer.save_model()

    train_loss = train_result.metrics.get("train_loss", float("nan"))
    primary_metric_key = f"eval_{args.metric}"
    primary_metric = eval_result.get(primary_metric_key, float("nan"))
    eval_loss = eval_result.get("eval_loss", float("nan"))
    train_samples_per_sec = len(train_ds) / train_wall if train_wall > 0 else 0

    metrics = {
        "model": args.model,
        "dataset": args.dataset,
        "dataset_config": args.dataset_config,
        "train_split": args.train_split,
        "eval_split": args.eval_split,
        "text_field_a": text_field_a,
        "text_field_b": text_field_b,
        "label_field": label_field,
        "metric": args.metric,
        "num_labels": num_labels,
        "peft_method": args.peft_method,
        "total_params_m": round(total_params / 1e6, 1),
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "lr": args.lr,
        "num_epochs": args.num_epochs,
        "batch_size": args.batch_size,
        "seq_length": args.seq_length,
        "train_samples": len(train_ds),
        "eval_samples": len(eval_ds),
        "train_loss": round(train_loss, 6),
        "eval_loss": round(eval_loss, 6),
        primary_metric_key: round(float(primary_metric), 6),
        "train_runtime_s": round(train_wall, 2),
        "train_samples_per_sec": round(train_samples_per_sec, 1),
        "seed": args.seed,
        "cuda_available": torch.cuda.is_available(),
        "device_name": torch.cuda.get_device_name(0)
        if torch.cuda.is_available()
        else "cpu",
        "fp16": use_fp16,
        "precision": precision,
    }
    metrics["provenance"] = provenance({k: str(v) for k, v in vars(args).items()})
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))
    return 0


def _normalize_text(value: object) -> str:
    if isinstance(value, str):
        return value
    if value is None:
        return ""
    return str(value)


def _build_peft_config(
    args: argparse.Namespace, target_modules: list[str]
) -> LoraConfig:
    init_lora_weights: bool | str = True
    use_dora = False
    use_rslora = False
    if args.peft_method == "dora":
        use_dora = True
    elif args.peft_method == "pissa":
        init_lora_weights = (
            f"pissa_niter_{args.pissa_niter}" if args.pissa_niter > 0 else "pissa"
        )
    elif args.peft_method == "rslora":
        use_rslora = True

    return LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        bias="none",
        task_type="SEQ_CLS",
        init_lora_weights=init_lora_weights,
        use_dora=use_dora,
        use_rslora=use_rslora,
    )


def _resolve_text_fields(args: argparse.Namespace, ds) -> tuple[str, str | None]:
    if args.text_field_a:
        return args.text_field_a, args.text_field_b

    columns = ds.column_names
    pair_candidates = [
        ("premise", "hypothesis"),
        ("sentence1", "sentence2"),
        ("question", "sentence"),
        ("question1", "question2"),
    ]
    for left, right in pair_candidates:
        if left in columns and right in columns:
            return left, right

    single_candidates = ["text", "sentence", "content"]
    for col in single_candidates:
        if col in columns:
            return col, None

    string_cols = []
    sample = ds[0]
    for col in columns:
        if isinstance(sample[col], str):
            string_cols.append(col)
    if len(string_cols) >= 2:
        return string_cols[0], string_cols[1]
    if len(string_cols) == 1:
        return string_cols[0], None

    raise ValueError(
        "Could not infer text fields; please pass --text-field-a/--text-field-b"
    )


def _resolve_label_list(train_ds, eval_ds, label_field: str) -> list[int]:
    labels = set()
    for ds in (train_ds, eval_ds):
        for value in ds[label_field]:
            labels.add(int(value))
    return sorted(labels)


def _compute_metrics(pred, metric_name: str) -> dict[str, float]:
    logits, labels = pred
    preds = np.argmax(logits, axis=-1)
    accuracy = float((preds == labels).mean())
    metrics = {"accuracy": accuracy}
    if metric_name == "f1":
        metrics["f1"] = _macro_f1(labels, preds)
    return metrics


def _macro_f1(labels: np.ndarray, preds: np.ndarray) -> float:
    labels = np.asarray(labels)
    preds = np.asarray(preds)
    classes = sorted(set(labels.tolist()) | set(preds.tolist()))
    f1_scores = []
    for cls in classes:
        tp = int(((preds == cls) & (labels == cls)).sum())
        fp = int(((preds == cls) & (labels != cls)).sum())
        fn = int(((preds != cls) & (labels == cls)).sum())
        precision = tp / (tp + fp) if tp + fp > 0 else 0.0
        recall = tp / (tp + fn) if tp + fn > 0 else 0.0
        if precision + recall == 0:
            f1_scores.append(0.0)
        else:
            f1_scores.append(2 * precision * recall / (precision + recall))
    return float(sum(f1_scores) / len(f1_scores)) if f1_scores else 0.0


def _detect_lora_targets(model) -> list[str]:
    known_prefixes = {
        "BertForSequenceClassification": ["query", "value"],
        "RobertaForSequenceClassification": ["query", "value"],
        "DebertaV2ForSequenceClassification": ["query_proj", "value_proj"],
        "DebertaForSequenceClassification": ["in_proj"],
        "LlamaForSequenceClassification": ["q_proj", "v_proj"],
        "GemmaForSequenceClassification": ["q_proj", "v_proj"],
        "MistralForSequenceClassification": ["q_proj", "v_proj"],
    }
    for prefix, targets in known_prefixes.items():
        if model.__class__.__name__.startswith(prefix):
            return targets

    valid = {
        "query",
        "value",
        "query_proj",
        "value_proj",
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "in_proj",
    }
    found = set()
    for name, _mod in model.named_modules():
        leaf = name.split(".")[-1]
        if leaf in valid:
            found.add(leaf)
    if found:
        preferred_order = [
            "query",
            "value",
            "query_proj",
            "value_proj",
            "q_proj",
            "v_proj",
            "o_proj",
            "in_proj",
            "k_proj",
        ]
        ordered = [name for name in preferred_order if name in found]
        return ordered if ordered else sorted(found)

    return ["q_proj", "v_proj"]


if __name__ == "__main__":
    raise SystemExit(main())
