#!/usr/bin/env python3
"""Linear Mode Connectivity (LMC) evaluation for adapter pools.

For each pool, sample N pair adapters, interpolate LoRA weights at
alpha ∈ {0, 0.25, 0.5, 0.75, 1.0}, and evaluate each interpolated
weight on the test split. Reports per-pair barrier height
  barrier = max(loss_along_path) - max(loss_at_endpoints)

If barrier ≈ 0, the pair is mode-connected (same loss basin).
If barrier > 0, there's a hump — distinct basins.

Hypothesis: pools where naive ensembling wins (decoders, low individual
variance) are highly mode-connected. Pools where greedy_soup is necessary
(Pool-B HP-diverse, DeBERTa) are NOT mode-connected.

Usage:
    python3 analysis/lmc_eval.py --pool pools/pool_a_mnli_bert.json \
        --task mnli --n-pairs 5 --out analysis/lmc_pool_a_mnli_bert.json
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from weight_space_merge import load_weights, save_weights
from peft_utils import provenance


TASK_CONFIGS = {
    "mnli":   {"dataset": "nyu-mll/glue", "config": "mnli", "split": "validation_matched",
               "fields": ("premise", "hypothesis"), "label": "label", "num_labels": 3, "max_len": 128},
    "anli":   {"dataset": "facebook/anli", "config": None, "split": "dev_r3",
               "fields": ("premise", "hypothesis"), "label": "label", "num_labels": 3, "max_len": 128},
    "boolq":  {"dataset": "aps/super_glue", "config": "boolq", "split": "validation",
               "fields": ("question", "passage"), "label": "label", "num_labels": 2, "max_len": 256},
    "agnews": {"dataset": "fancyzhx/ag_news", "config": None, "split": "test",
               "fields": ("text", None), "label": "label", "num_labels": 4, "max_len": 128},
    "qnli":   {"dataset": "nyu-mll/glue", "config": "qnli", "split": "validation",
               "fields": ("question", "sentence"), "label": "label", "num_labels": 2, "max_len": 128},
}


def split_indices(n: int, seed: int = 0):
    rng = np.random.RandomState(seed)
    idx = np.arange(n); rng.shuffle(idx)
    n_sel = int(0.4 * n); n_combo = int(0.2 * n)
    return idx[:n_sel], idx[n_sel:n_sel + n_combo], idx[n_sel + n_combo:]


def interp_weights(w_a: dict, w_b: dict, alpha: float) -> dict:
    """alpha * w_a + (1-alpha) * w_b on shared keys."""
    keys = sorted(set(w_a.keys()) & set(w_b.keys()))
    return {k: (alpha * w_a[k].float() + (1 - alpha) * w_b[k].float()) for k in keys}


@torch.no_grad()
def eval_adapter(base_model_id, adapter_dir, tokenizer, texts_a, texts_b,
                 labels, batch_size, device, max_length, num_labels):
    base = AutoModelForSequenceClassification.from_pretrained(base_model_id, num_labels=num_labels)
    if base.config.pad_token_id is None and tokenizer.pad_token_id is not None:
        base.config.pad_token_id = tokenizer.pad_token_id
    model = PeftModel.from_pretrained(base, adapter_dir).to(device).eval()
    losses, correct, n = 0.0, 0, 0
    for start in range(0, len(texts_a), batch_size):
        a = texts_a[start:start + batch_size]
        b = texts_b[start:start + batch_size] if texts_b else None
        enc = tokenizer(a, b, padding=True, truncation=True, max_length=max_length,
                        return_tensors="pt").to(device)
        L = labels[start:start + batch_size]
        logits = model(**enc).logits.float()
        loss = F.cross_entropy(logits, torch.tensor(L, device=device), reduction="sum")
        losses += float(loss.item())
        preds = logits.argmax(dim=-1).cpu().numpy()
        correct += int((preds == L).sum())
        n += len(L)
    del model, base
    torch.cuda.empty_cache()
    return {"loss": losses / max(n, 1), "acc": correct / max(n, 1), "n": n}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pool", required=True, type=Path)
    p.add_argument("--task", required=True, choices=list(TASK_CONFIGS))
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--n-pairs", type=int, default=5)
    p.add_argument("--alphas", nargs="+", type=float, default=[0.0, 0.25, 0.5, 0.75, 1.0])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--max-eval", type=int, default=None,
                   help="Cap eval set size for speed; default uses test split (40% of full eval)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    pool = json.loads(args.pool.read_text())
    adapters = pool["adapters"]
    pool_id = pool.get("pool_id") or pool.get("pool_name") or args.pool.stem
    base_model_id = (pool.get("base_model") or adapters[0].get("base_model")
                     or adapters[0].get("metadata", {}).get("base_model"))
    if not base_model_id:
        raise SystemExit("base_model unknown")
    task = TASK_CONFIGS[args.task]

    # === Load eval set, take test split ===
    ds_kwargs = {"path": task["dataset"], "split": task["split"], "trust_remote_code": True}
    if task["config"]:
        ds_kwargs["name"] = task["config"]
    ds = load_dataset(**ds_kwargs)
    field_a, field_b = task["fields"]
    texts_a_all = [ex[field_a] for ex in ds]
    texts_b_all = [ex[field_b] for ex in ds] if field_b else None
    labels_all = np.array([ex[task["label"]] for ex in ds], dtype=np.int64)
    _, _, test_idx = split_indices(len(labels_all), seed=0)
    if args.max_eval and args.max_eval < len(test_idx):
        # subsample for speed
        rng = np.random.RandomState(0)
        test_idx = rng.choice(test_idx, size=args.max_eval, replace=False)
    texts_a = [texts_a_all[i] for i in test_idx]
    texts_b = [texts_b_all[i] for i in test_idx] if texts_b_all else None
    labels = labels_all[test_idx]
    print(f"[lmc] pool={pool_id}, task={args.task}, base={base_model_id}, "
          f"N_adapters={len(adapters)}, n_test_eval={len(labels)}")

    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # === Pre-load all adapter weights, find compatible pairs ===
    loaded = []
    for a in adapters:
        w = load_weights(Path(a["path"]))
        if w is None:
            continue
        loaded.append((a, w))
    print(f"[lmc] loaded {len(loaded)} adapter weight sets")
    if len(loaded) < 2:
        raise SystemExit("need >= 2 adapters")

    # Reference adapter dir for adapter_config.json (needed by PeftModel.from_pretrained)
    ref_dir = Path(loaded[0][0]["path"])

    # === Sample N pairs ===
    rng = np.random.RandomState(args.seed)
    pair_idxs = []
    while len(pair_idxs) < args.n_pairs:
        i, j = rng.choice(len(loaded), size=2, replace=False)
        i, j = int(min(i, j)), int(max(i, j))
        if (i, j) in pair_idxs: continue
        pair_idxs.append((i, j))
    print(f"[lmc] pairs: {pair_idxs}")

    # === For each pair, eval interpolated weights at each alpha ===
    out_pairs = []
    tmp_root = Path(tempfile.mkdtemp(prefix="lmc_"))
    try:
        for pi, (i, j) in enumerate(pair_idxs):
            a_i = loaded[i][0]; a_j = loaded[j][0]
            w_i = loaded[i][1]; w_j = loaded[j][1]
            print(f"\n=== pair {pi+1}/{len(pair_idxs)}: {a_i['adapter_id']}  ↔  {a_j['adapter_id']} ===")
            path_results = []
            for alpha in args.alphas:
                interp = interp_weights(w_i, w_j, alpha)
                tmp_dir = tmp_root / f"pair{pi}_a{alpha}"
                tmp_dir.mkdir(parents=True, exist_ok=True)
                save_weights(interp, tmp_dir)
                # copy adapter_config.json from reference adapter
                cfg = ref_dir / "adapter_config.json"
                if cfg.exists():
                    shutil.copy2(cfg, tmp_dir / "adapter_config.json")
                t0 = time.time()
                m = eval_adapter(base_model_id, str(tmp_dir), tokenizer, texts_a, texts_b,
                                  labels, args.batch_size, args.device, task["max_len"], task["num_labels"])
                m["alpha"] = alpha
                m["wall_s"] = round(time.time() - t0, 1)
                path_results.append(m)
                print(f"  alpha={alpha:.2f}  loss={m['loss']:.4f}  acc={m['acc']:.4f}  "
                      f"({m['wall_s']}s)")
            losses = [r["loss"] for r in path_results]
            endpoints_max = max(losses[0], losses[-1])
            barrier = max(losses) - endpoints_max
            out_pairs.append({
                "adapter_a": a_i["adapter_id"], "adapter_b": a_j["adapter_id"],
                "path": path_results,
                "barrier": float(barrier),
                "endpoints_max": float(endpoints_max),
            })
            print(f"  → barrier={barrier:.4f}")
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)

    barriers = [p["barrier"] for p in out_pairs]
    out = {
        "pool_id": pool_id,
        "task": args.task,
        "base_model": base_model_id,
        "n_pairs": len(out_pairs),
        "alphas": args.alphas,
        "n_eval": int(len(labels)),
        "pairs": out_pairs,
        "barrier_mean": float(np.mean(barriers)),
        "barrier_std": float(np.std(barriers)),
        "barrier_max": float(max(barriers)),
        "barrier_min": float(min(barriers)),
        "provenance": provenance({k: str(v) for k, v in vars(args).items()}),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(f"\n[lmc] barrier mean={np.mean(barriers):.4f} ± {np.std(barriers):.4f}  "
          f"(max={max(barriers):.4f})  → {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
