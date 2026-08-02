#!/usr/bin/env python3
"""Real LoRA fine-tuning on HuggingFace model + dataset.

Supports any causal LM from HuggingFace Hub with LoRA via PEFT.
Designed for distributed hyperparameter sweeps across the cluster.

Usage:
    python3 real_lora_train.py --model gpt2 --dataset wikitext \\
        --dataset-config wikitext-2-raw-v1 --lora-r 8 --lr 0.0003

    python3 real_lora_train.py --model distilbert/distilgpt2 \\
        --dataset wikitext --dataset-config wikitext-2-raw-v1 \\
        --lora-r 16 --lr 0.001 --max-samples 5000 --num-epochs 2
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)

from peft_utils import provenance


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Real LoRA fine-tuning on HF model + dataset"
    )
    p.add_argument("--model", required=True, help="HF model ID")
    p.add_argument("--dataset", required=True, help="HF dataset ID")
    p.add_argument("--dataset-config", default=None)
    p.add_argument("--dataset-split", default="train")
    p.add_argument("--eval-split", default="validation")
    p.add_argument("--eval-dataset", default=None,
                   help="Separate eval dataset ID (default: same as --dataset)")
    p.add_argument("--eval-dataset-config", default=None)
    p.add_argument("--output-dir", default=".")
    p.add_argument("--max-samples", type=int, default=10000)
    p.add_argument("--max-eval-samples", type=int, default=500)
    p.add_argument("--num-epochs", type=float, default=1.0)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--seq-length", type=int, default=128)
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
    p.add_argument("--prompt-field", default=None)
    p.add_argument("--target-field", default=None)
    p.add_argument(
        "--eval-metric",
        default="loss",
        choices=["loss", "gsm8k_exact_match"],
        help="Primary evaluation metric",
    )
    p.add_argument("--generation-max-new-tokens", type=int, default=64)
    p.add_argument("--load-in-4bit", action="store_true")
    p.add_argument("--load-in-8bit", action="store_true")
    p.add_argument(
        "--bnb-compute-dtype",
        default="float16",
        choices=["float16", "bfloat16", "float32"],
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--warmup-ratio", type=float, default=0.05)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--no-fp16", action="store_true")
    p.add_argument("--gradient-checkpointing", action="store_true",
                   help="Enable gradient checkpointing to reduce VRAM")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    # Determinism: pin RNGs and cuDNN heuristics for bit-exact training reproducibility.
    # Reduces GPU non-determinism (cuDNN autotune) at the cost of mild throughput.
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

    print(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            **_model_load_kwargs(args),
        )
    except Exception as exc:
        if args.load_in_4bit or args.load_in_8bit:
            raise RuntimeError(
                "Quantized model loading failed. This usually means the runtime is missing a "
                "working bitsandbytes/Triton backend or required system headers (for example "
                "Python.h from python3-dev)."
            ) from exc
        raise
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

    print(
        f"Loading dataset: {args.dataset}"
        + (f" ({args.dataset_config})" if args.dataset_config else "")
    )
    ds_kwargs: dict = {
        "path": args.dataset,
        "split": args.dataset_split,
    }
    if args.dataset_config:
        ds_kwargs["name"] = args.dataset_config

    raw_ds = load_dataset(**ds_kwargs)
    prompt_field, target_field, text_column = _resolve_training_fields(args, raw_ds)

    train_ds = raw_ds.select(range(min(args.max_samples, len(raw_ds))))
    raw_eval = None
    try:
        if args.eval_dataset:
            # Separate eval dataset (e.g., train on MetaMathQA, eval on GSM8K)
            eval_ds_kwargs = {"path": args.eval_dataset, "split": args.eval_split}
            if args.eval_dataset_config:
                eval_ds_kwargs["name"] = args.eval_dataset_config
            raw_eval = load_dataset(**eval_ds_kwargs)
        else:
            eval_ds_kwargs = {**ds_kwargs, "split": args.eval_split}
            raw_eval = load_dataset(**eval_ds_kwargs)
        eval_ds = raw_eval.select(range(min(args.max_eval_samples, len(raw_eval))))
    except (ValueError, FileNotFoundError, KeyError) as exc:
        print(f"WARNING: eval split not found ({exc}); holding out from train data")
        split = raw_ds.train_test_split(test_size=min(args.max_eval_samples, len(raw_ds) // 5), seed=args.seed)
        raw_ds = split["train"]
        train_ds = raw_ds.select(range(min(args.max_samples, len(raw_ds))))
        eval_ds = split["test"]
        raw_eval = eval_ds

    print(f"  Train: {len(train_ds)} rows, Eval: {len(eval_ds)} rows")

    def tokenize(batch):
        cleaned = _prepare_causal_texts(batch, text_column, prompt_field, target_field)
        result = tokenizer(
            cleaned,
            truncation=True,
            max_length=args.seq_length,
            padding="max_length",
        )
        result["labels"] = result["input_ids"].copy()
        return result

    print("Tokenizing...")
    train_ds = train_ds.map(
        tokenize, batched=True, remove_columns=train_ds.column_names
    )
    # If eval dataset is different from train dataset, auto-detect its fields
    if args.eval_dataset:
        eval_cols = raw_eval.column_names
        # Auto-detect prompt/target for common eval datasets
        eval_prompt_f = None
        eval_target_f = None
        for pf in ["question", "query", "prompt", "instruction"]:
            if pf in eval_cols:
                eval_prompt_f = pf
                break
        for tf in ["answer", "response", "output", "target"]:
            if tf in eval_cols:
                eval_target_f = tf
                break
        eval_text_col = "text" if "text" in eval_cols else None

        def tokenize_eval(batch):
            cleaned = _prepare_causal_texts(batch, eval_text_col, eval_prompt_f, eval_target_f)
            result = tokenizer(cleaned, truncation=True, max_length=args.seq_length, padding="max_length")
            result["labels"] = result["input_ids"].copy()
            return result
        eval_ds = eval_ds.map(tokenize_eval, batched=True, remove_columns=eval_ds.column_names)
    else:
        eval_ds = eval_ds.map(tokenize, batched=True, remove_columns=eval_ds.column_names)

    use_fp16 = torch.cuda.is_available() and not args.no_fp16
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
        remove_unused_columns=False,
        dataloader_pin_memory=False,
        gradient_checkpointing=args.gradient_checkpointing,
        seed=args.seed,
    )

    if args.gradient_checkpointing:
        model.enable_input_require_grads()

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
    )

    t0 = time.time()
    print("Training...")
    train_result = trainer.train()
    train_wall = time.time() - t0

    print("Evaluating...")
    eval_result = trainer.evaluate()

    extra_metrics: dict[str, float] = {}
    if args.eval_metric == "gsm8k_exact_match":
        # If eval dataset is separate, auto-detect its prompt/target fields
        eval_pf = prompt_field
        eval_tf = target_field
        if args.eval_dataset and raw_eval is not None:
            ecols = raw_eval.column_names
            for pf in ["question", "query", "prompt"]:
                if pf in ecols:
                    eval_pf = pf
                    break
            for tf in ["answer", "response", "output"]:
                if tf in ecols:
                    eval_tf = tf
                    break
        extra_metrics = _evaluate_gsm8k_exact_match(
            model,
            tokenizer,
            raw_eval,
            args,
            eval_pf,
            eval_tf,
        )

    trainer.save_model()

    train_loss = train_result.metrics.get("train_loss", float("nan"))
    eval_loss = eval_result.get("eval_loss", float("nan"))
    train_samples_per_sec = args.max_samples / train_wall if train_wall > 0 else 0

    metrics = {
        "model": args.model,
        "dataset": args.dataset,
        "dataset_config": args.dataset_config,
        "eval_split": args.eval_split,
        "prompt_field": prompt_field,
        "target_field": target_field,
        "peft_method": args.peft_method,
        "eval_metric": args.eval_metric,
        "load_in_4bit": args.load_in_4bit,
        "load_in_8bit": args.load_in_8bit,
        "bnb_compute_dtype": args.bnb_compute_dtype,
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
        "train_runtime_s": round(train_wall, 2),
        "train_samples_per_sec": round(train_samples_per_sec, 1),
        "seed": args.seed,
        "cuda_available": torch.cuda.is_available(),
        "device_name": torch.cuda.get_device_name(0)
        if torch.cuda.is_available()
        else "cpu",
        "fp16": use_fp16,
    }
    metrics.update({k: round(v, 6) for k, v in extra_metrics.items()})
    metrics["provenance"] = provenance({k: str(v) for k, v in vars(args).items()})
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))
    return 0


def _detect_lora_targets(model) -> list[str]:
    known_prefixes = {
        "GPT2": ["c_attn", "c_proj"],
        "GPTNeoX": ["query_key_value", "dense"],
        "LlamaForCausalLM": ["q_proj", "v_proj"],
        "MistralForCausalLM": ["q_proj", "v_proj"],
        "OPTForCausalLM": ["q_proj", "v_proj"],
    }
    for prefix, targets in known_prefixes.items():
        if model.__class__.__name__.startswith(prefix):
            return targets

    valid = {
        "c_attn",
        "c_proj",
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "query_key_value",
        "dense",
        "qkv_proj",
    }
    found = set()
    for name, mod in model.named_modules():
        leaf = name.split(".")[-1]
        if leaf in valid:
            found.add(leaf)
    if found:
        return sorted(found)

    return ["q_proj", "v_proj"]


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
        task_type="CAUSAL_LM",
        init_lora_weights=init_lora_weights,
        use_dora=use_dora,
        use_rslora=use_rslora,
    )


def _model_load_kwargs(args: argparse.Namespace) -> dict:
    if args.load_in_4bit and args.load_in_8bit:
        raise ValueError("Choose at most one of --load-in-4bit or --load-in-8bit")

    kwargs: dict = {}
    if args.load_in_4bit or args.load_in_8bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=args.load_in_4bit,
            load_in_8bit=args.load_in_8bit,
            bnb_4bit_compute_dtype=_resolve_torch_dtype(args.bnb_compute_dtype),
        )
        kwargs["device_map"] = "auto"
        kwargs["max_memory"] = {0: "10GiB", "cpu": "24GiB"}
    return kwargs


def _resolve_torch_dtype(name: str) -> torch.dtype:
    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    return mapping[name]


def _resolve_training_fields(
    args: argparse.Namespace, ds
) -> tuple[str | None, str | None, str | None]:
    if args.prompt_field and args.target_field:
        return args.prompt_field, args.target_field, None

    columns = set(ds.column_names)
    if {"question", "answer"}.issubset(columns):
        return "question", "answer", None
    if {"prompt", "completion"}.issubset(columns):
        return "prompt", "completion", None
    return None, None, _detect_text_column(ds)


def _prepare_causal_texts(
    batch: dict,
    text_column: str | None,
    prompt_field: str | None,
    target_field: str | None,
) -> list[str]:
    if prompt_field and target_field:
        prompts = batch[prompt_field]
        targets = batch[target_field]
        return [
            _format_prompt_target(prompt, target)
            for prompt, target in zip(prompts, targets)
        ]

    texts = batch[text_column] if text_column else []
    return [t if t and isinstance(t, str) else "" for t in texts]


def _format_prompt_target(prompt: object, target: object) -> str:
    prompt_text = str(prompt).strip() if prompt is not None else ""
    target_text = str(target).strip() if target is not None else ""
    return f"Question: {prompt_text}\nAnswer: {target_text}"


def _evaluate_gsm8k_exact_match(
    model, tokenizer, raw_eval_ds, args, prompt_field, target_field
):
    if not prompt_field or not target_field:
        return {}

    model.eval()
    device = next(model.parameters()).device
    exact = 0
    total = 0
    for example in raw_eval_ds.select(
        range(min(args.max_eval_samples, len(raw_eval_ds)))
    ):
        prompt = str(example[prompt_field]).strip()
        gold = str(example[target_field]).strip()
        input_text = f"Question: {prompt}\nAnswer:"
        inputs = tokenizer(
            input_text, return_tensors="pt", truncation=True, max_length=args.seq_length
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            generated = model.generate(
                **inputs,
                max_new_tokens=args.generation_max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        output = tokenizer.decode(
            generated[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )
        pred_answer = _extract_gsm8k_answer(output)
        gold_answer = _extract_gsm8k_answer(gold)
        exact += int(pred_answer == gold_answer and gold_answer != "")
        total += 1

    if total == 0:
        return {}
    return {"eval_gsm8k_exact_match": 100.0 * exact / total}


def _extract_gsm8k_answer(text: str) -> str:
    # GSM8K answers are integers, optionally with commas (e.g., "1,234")
    marker = re.search(r"####\s*([-+]?\d[\d,]*)", text)
    if marker:
        return marker.group(1).replace(",", "")
    numbers = re.findall(r"[-+]?\d[\d,]*", text)
    if not numbers:
        return ""
    return numbers[-1].replace(",", "")


def _detect_text_column(ds) -> str:
    if "text" in ds.column_names:
        return "text"
    for col in ds.column_names:
        if (
            "text" in col.lower()
            or "content" in col.lower()
            or "sentence" in col.lower()
        ):
            return col
    for col in ds.column_names:
        sample = ds[0][col]
        if isinstance(sample, str) and len(sample) > 50:
            return col
    return ds.column_names[0]


if __name__ == "__main__":
    raise SystemExit(main())
