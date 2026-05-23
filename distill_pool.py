#!/usr/bin/env python3
"""Distill an adapter pool into a single student adapter (Phase 4 T2.2).

Two-step process:
  1. Compute teacher training-set logits by running each pool member on
     the training data once (cached as analysis/distill_cache/<pool>_train_logits.npy
     of shape (N_adapters, N_train, K) for the eval-time adapter set).
  2. Train a single LoRA adapter using a combined loss:
        L = alpha * KL(softmax(student/T), softmax(teacher_avg/T)) * T^2
          + (1 - alpha) * CE(student_logits, hard_labels)
     Default T=4, alpha=0.7 (Hinton 2015 distillation).

Outputs the distilled adapter to <out-dir> plus distill_metrics.json with
accuracy + ECE on the test split (same 40/20/40 split as ensemble_eval).

Usage:
    python3 distill_pool.py --pool pools/pool_a_mnli_bert.json --task mnli \
        --out-dir distilled_adapters/pool_a_mnli_bert_distilled
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, PeftModel
from transformers import (
    AutoModelForSequenceClassification, AutoTokenizer,
    DataCollatorWithPadding, Trainer, TrainingArguments,
)

from peft_utils import provenance


TASK_CONFIGS = {
    "mnli": {
        "dataset": "nyu-mll/glue", "config": "mnli",
        "train_split": "train", "eval_split": "validation_matched",
        "fields": ("premise", "hypothesis"), "label": "label", "num_labels": 3,
        "max_length": 128, "max_train": 40000,
    },
    "anli": {
        "dataset": "facebook/anli", "config": None,
        "train_split": "train_r3", "eval_split": "dev_r3",
        "fields": ("premise", "hypothesis"), "label": "label", "num_labels": 3,
        "max_length": 128, "max_train": 40000,
    },
    "boolq": {
        "dataset": "aps/super_glue", "config": "boolq",
        "train_split": "train", "eval_split": "validation",
        "fields": ("question", "passage"), "label": "label", "num_labels": 2,
        "max_length": 256, "max_train": 9427,
    },
}


def split_indices(n: int, seed: int = 0):
    rng = np.random.RandomState(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_sel = int(0.4 * n)
    n_combo = int(0.2 * n)
    return idx[:n_sel], idx[n_sel:n_sel + n_combo], idx[n_sel + n_combo:]


@torch.no_grad()
def compute_adapter_logits(base_model_id, adapter_path, tokenizer, texts_a, texts_b,
                           batch_size, device, max_length, num_labels):
    base = AutoModelForSequenceClassification.from_pretrained(base_model_id, num_labels=num_labels)
    if base.config.pad_token_id is None and tokenizer.pad_token_id is not None:
        base.config.pad_token_id = tokenizer.pad_token_id
    model = PeftModel.from_pretrained(base, adapter_path).to(device).eval()
    out = []
    for start in range(0, len(texts_a), batch_size):
        a = texts_a[start:start + batch_size]
        b = texts_b[start:start + batch_size] if texts_b else None
        enc = tokenizer(a, b, padding=True, truncation=True, max_length=max_length,
                        return_tensors="pt").to(device)
        out.append(model(**enc).logits.detach().to(torch.float32).cpu().numpy())
    del model, base
    torch.cuda.empty_cache()
    return np.concatenate(out, axis=0)


def get_teacher_train_logits(pool_id, adapters, base_model_id, tokenizer, texts_a, texts_b,
                              batch_size, device, max_length, num_labels):
    cache_dir = Path("analysis/distill_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{pool_id}_teacher_train_logits.npy"
    if cache_path.exists():
        loaded = np.load(cache_path)
        if loaded.shape == (len(adapters), len(texts_a), num_labels):
            print(f"[distill] reuse cached teacher logits {cache_path} shape={loaded.shape}")
            return loaded
    print(f"[distill] computing teacher train logits for {len(adapters)} adapters x {len(texts_a)} examples")
    stack = []
    for i, a in enumerate(adapters):
        t0 = time.time()
        L = compute_adapter_logits(base_model_id, a["path"], tokenizer, texts_a, texts_b,
                                    batch_size, device, max_length, num_labels)
        print(f"  [{i+1}/{len(adapters)}] {a['adapter_id']} done in {time.time()-t0:.0f}s, shape={L.shape}")
        stack.append(L)
    arr = np.stack(stack, axis=0).astype(np.float32)
    np.save(cache_path, arr)
    return arr


class DistillTrainer(Trainer):
    """KL distillation + CE on hard labels."""
    def __init__(self, *args, teacher_probs=None, alpha=0.7, temperature=4.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher_probs = teacher_probs  # shape (N_train, num_labels), already softmaxed at temp T
        self.alpha = alpha
        self.temperature = temperature

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        ex_idx = inputs.pop("ex_idx").long()
        outputs = model(**inputs)
        student_logits = outputs.logits  # (B, K)
        T = self.temperature
        soft_targets = torch.tensor(self.teacher_probs[ex_idx.cpu().numpy()],
                                    dtype=student_logits.dtype, device=student_logits.device)
        student_log_softmax_T = F.log_softmax(student_logits / T, dim=-1)
        kl = F.kl_div(student_log_softmax_T, soft_targets, reduction="batchmean") * (T * T)
        ce = F.cross_entropy(student_logits, labels)
        loss = self.alpha * kl + (1.0 - self.alpha) * ce
        return (loss, outputs) if return_outputs else loss


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pool", required=True, type=Path)
    p.add_argument("--task", required=True, choices=list(TASK_CONFIGS))
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--alpha", type=float, default=0.7, help="KL weight; (1-alpha) on CE hard label")
    p.add_argument("--temperature", type=float, default=4.0)
    p.add_argument("--lora-r", type=int, default=8)
    p.add_argument("--lora-alpha", type=int, default=16)
    p.add_argument("--lora-target", nargs="+", default=None,
                   help="If unset, inferred from base model class")
    p.add_argument("--num-epochs", type=int, default=2)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--inference-batch-size", type=int, default=64,
                   help="Batch size for teacher logit pre-compute")
    p.add_argument("--precision", choices=["fp16", "bf16", "fp32"], default="bf16")
    return p.parse_args()


def auto_lora_target(model):
    cls = model.__class__.__name__
    if cls.startswith("BertForSequenceClassification") or cls.startswith("RobertaForSequenceClassification"):
        return ["query", "value"]
    if cls.startswith("DebertaV2ForSequenceClassification"):
        return ["query_proj", "value_proj"]
    if cls.startswith(("Qwen2ForSequenceClassification", "LlamaForSequenceClassification")):
        return ["q_proj", "v_proj"]
    return ["q_proj", "v_proj"]


def main() -> int:
    args = parse_args()
    pool = json.loads(args.pool.read_text())
    adapters = pool["adapters"]
    if not adapters:
        raise SystemExit("Empty pool")
    pool_id = pool.get("pool_id") or pool.get("pool_name") or args.pool.stem
    base_model_id = (
        pool.get("base_model")
        or adapters[0].get("base_model")
        or adapters[0].get("metadata", {}).get("base_model")
    )
    if not base_model_id:
        raise SystemExit("Could not infer base model from pool manifest")

    task = TASK_CONFIGS[args.task]
    print(f"[distill] pool={pool_id}, base={base_model_id}, task={args.task}, "
          f"N={len(adapters)}, alpha={args.alpha}, T={args.temperature}")

    # === Load training data ===
    ds_kwargs = {"path": task["dataset"], "split": task["train_split"], "trust_remote_code": True}
    if task["config"]: ds_kwargs["name"] = task["config"]
    train_ds = load_dataset(**ds_kwargs)
    if task["max_train"] and len(train_ds) > task["max_train"]:
        train_ds = train_ds.select(range(task["max_train"]))
    field_a, field_b = task["fields"]
    texts_a = [ex[field_a] for ex in train_ds]
    texts_b = [ex[field_b] for ex in train_ds] if field_b else None
    train_labels = np.asarray([ex[task["label"]] for ex in train_ds], dtype=np.int64)

    # === Tokenizer & teacher ===
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    teacher_logits = get_teacher_train_logits(
        pool_id, adapters, base_model_id, tokenizer, texts_a, texts_b,
        args.inference_batch_size, args.device, task["max_length"], task["num_labels"],
    )  # (N_ad, N_train, K)
    teacher_probs = torch.softmax(
        torch.from_numpy(teacher_logits.mean(axis=0)) / args.temperature, dim=-1
    ).numpy().astype(np.float32)
    print(f"[distill] teacher_probs shape={teacher_probs.shape}, "
          f"mean max prob={teacher_probs.max(axis=1).mean():.3f}")

    # === Build student LoRA adapter ===
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    student_base = AutoModelForSequenceClassification.from_pretrained(
        base_model_id, num_labels=task["num_labels"])
    if student_base.config.pad_token_id is None and tokenizer.pad_token_id is not None:
        student_base.config.pad_token_id = tokenizer.pad_token_id
    target = args.lora_target or auto_lora_target(student_base)
    lora_cfg = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.0,
        target_modules=target, bias="none", task_type="SEQ_CLS",
    )
    student = get_peft_model(student_base, lora_cfg)
    student.print_trainable_parameters()

    # === Tokenize training data and attach example index ===
    def tok_batch(batch, idxs):
        a = batch[field_a]; b = batch[field_b] if field_b else None
        enc = tokenizer(a, b, padding=False, truncation=True, max_length=task["max_length"])
        enc["labels"] = batch[task["label"]]
        enc["ex_idx"] = idxs
        return enc

    train_ds_tok = train_ds.map(
        tok_batch, batched=True, with_indices=True,
        remove_columns=train_ds.column_names,
    )

    use_fp16 = args.precision == "fp16" and torch.cuda.is_available()
    use_bf16 = args.precision == "bf16" and torch.cuda.is_available()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    targs = TrainingArguments(
        output_dir=str(args.out_dir),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        num_train_epochs=args.num_epochs,
        warmup_ratio=0.06, weight_decay=0.01,
        logging_steps=50, save_strategy="no", eval_strategy="no",
        report_to=[], fp16=use_fp16, bf16=use_bf16,
        remove_unused_columns=False, dataloader_pin_memory=False, seed=args.seed,
    )
    trainer = DistillTrainer(
        model=student, args=targs,
        train_dataset=train_ds_tok,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        teacher_probs=teacher_probs, alpha=args.alpha, temperature=args.temperature,
    )
    t0 = time.time()
    train_out = trainer.train()
    train_wall = time.time() - t0
    student.save_pretrained(args.out_dir)
    tokenizer.save_pretrained(args.out_dir)

    # === Eval on test split ===
    eval_ds_kwargs = {"path": task["dataset"], "split": task["eval_split"], "trust_remote_code": True}
    if task["config"]: eval_ds_kwargs["name"] = task["config"]
    eval_ds = load_dataset(**eval_ds_kwargs)
    eval_a = [ex[field_a] for ex in eval_ds]
    eval_b = [ex[field_b] for ex in eval_ds] if field_b else None
    eval_labels = np.asarray([ex[task["label"]] for ex in eval_ds], dtype=np.int64)
    sel_idx, combo_idx, test_idx = split_indices(len(eval_labels), seed=0)
    student_logits = compute_adapter_logits(
        base_model_id, str(args.out_dir), tokenizer, eval_a, eval_b,
        args.inference_batch_size, args.device, task["max_length"], task["num_labels"],
    )
    test_logits = student_logits[test_idx]
    test_labels = eval_labels[test_idx]
    probs = torch.softmax(torch.from_numpy(test_logits), dim=-1).numpy()
    preds = probs.argmax(axis=-1)
    acc = float((preds == test_labels).mean())

    # ECE 15-bin equal-mass
    confidences = probs[np.arange(len(probs)), preds]
    correct = (preds == test_labels).astype(np.float64)
    order = np.argsort(confidences)
    bins = np.array_split(order, 15)
    ece = 0.0
    for b in bins:
        if len(b) == 0: continue
        ece += (len(b) / len(probs)) * abs(confidences[b].mean() - correct[b].mean())

    metrics = {
        "pool": pool_id, "task": args.task,
        "n_test": int(len(test_labels)),
        "accuracy_test": acc,
        "ece_15bin_equalmass_test": float(ece),
        "alpha": args.alpha, "temperature": args.temperature,
        "lora_r": args.lora_r, "lora_alpha": args.lora_alpha,
        "train_runtime_s": round(train_wall, 1),
        "n_train": int(len(train_ds)),
        "provenance": provenance({k: str(v) for k, v in vars(args).items()}),
    }
    (args.out_dir / "distill_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"[distill] test_acc={acc:.4f}  test_ece={ece:.4f}  → {args.out_dir/'distill_metrics.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
