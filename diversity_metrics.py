#!/usr/bin/env python3
"""Compute pairwise and pool-level diversity metrics for an adapter pool.

Metrics produced:
    Prediction-space (use cached ensemble_eval logits):
        - pairwise disagreement rate (fraction of examples where argmax differs)
        - logit Pearson correlation (averaged over labels)
        - error-set Jaccard (intersection over union of misclassified examples)
        - pool entropy (mean example-wise predictive entropy of soft-voted ensemble)
    Parameter-space (requires loading adapter weights):
        - pairwise LoRA-A/B cosine distance
        - pairwise delta-W Frobenius distance
    Metadata:
        - hyperparameter entropy across pool (rank/alpha/lr/method diversity)

Usage:
    python3 diversity_metrics.py --pool pools/pool_a_mnli_bert.json \
        --logits-cache ensemble_cache/ --out analysis/pool_a_diversity.json
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np

from peft_utils import provenance


def pairwise_disagreement(preds: np.ndarray) -> float:
    """preds: (N_adapters, N_examples). Returns mean pairwise disagreement."""
    n_adapters = preds.shape[0]
    total = 0.0
    pairs = 0
    for i, j in itertools.combinations(range(n_adapters), 2):
        total += float((preds[i] != preds[j]).mean())
        pairs += 1
    return total / max(1, pairs)


def pairwise_logit_correlation(logits_stack: np.ndarray) -> float:
    """logits_stack: (N_adapters, N_examples, num_labels). Mean Pearson over pairs and labels."""
    n = logits_stack.shape[0]
    L = logits_stack.shape[-1]
    corrs = []
    for i, j in itertools.combinations(range(n), 2):
        per_label = []
        for lbl in range(L):
            a = logits_stack[i, :, lbl]
            b = logits_stack[j, :, lbl]
            if a.std() < 1e-8 or b.std() < 1e-8:
                continue
            per_label.append(float(np.corrcoef(a, b)[0, 1]))
        if per_label:
            corrs.append(np.mean(per_label))
    return float(np.mean(corrs)) if corrs else float("nan")


def pairwise_error_jaccard(preds: np.ndarray, labels: np.ndarray) -> float:
    n = preds.shape[0]
    error_sets = [set(np.where(preds[i] != labels)[0].tolist()) for i in range(n)]
    vals = []
    for i, j in itertools.combinations(range(n), 2):
        inter = len(error_sets[i] & error_sets[j])
        union = len(error_sets[i] | error_sets[j])
        if union > 0:
            vals.append(inter / union)
    return float(np.mean(vals)) if vals else float("nan")


def ensemble_predictive_entropy(logits_stack: np.ndarray) -> float:
    """Mean per-example entropy (nats) of the soft-voted ensemble."""
    probs = np.exp(logits_stack - logits_stack.max(axis=-1, keepdims=True))
    probs /= probs.sum(axis=-1, keepdims=True)
    mean_probs = probs.mean(axis=0)
    ent = -np.sum(mean_probs * np.log(mean_probs + 1e-12), axis=-1)
    return float(ent.mean())


def metadata_entropy(adapters: list[dict], key: str) -> float:
    vals = [str(a["metadata"].get(key)) for a in adapters if a["metadata"].get(key) is not None]
    if not vals:
        return 0.0
    counts = Counter(vals)
    total = sum(counts.values())
    return -sum((c / total) * math.log(c / total) for c in counts.values())


def try_load_adapter_weights(adapter_dir: Path) -> dict | None:
    """Best-effort load of lora_A/lora_B weights for cosine/frobenius metrics."""
    for fname in ("adapter_model.safetensors", "adapter_model.bin"):
        p = adapter_dir / fname
        if not p.exists():
            continue
        try:
            if fname.endswith(".safetensors"):
                from safetensors.torch import load_file
                return {k: v for k, v in load_file(str(p)).items()}
            else:
                import torch
                return {k: v for k, v in torch.load(str(p), map_location="cpu").items()}
        except Exception as e:
            print(f"[diversity] failed to load {p}: {e}")
            return None
    return None


def pairwise_weight_distance(adapters: list[dict], max_pairs: int = 200) -> dict:
    """Compute mean pairwise LoRA-A and LoRA-B cosine distance across adapters.

    If adapters have different ranks (incompatible), skip those pairs.
    """
    import torch
    import torch.nn.functional as F

    weights = []
    for a in adapters:
        w = try_load_adapter_weights(Path(a["path"]))
        weights.append(w)

    idx_pairs = list(itertools.combinations(range(len(adapters)), 2))
    if len(idx_pairs) > max_pairs:
        rng = np.random.RandomState(0)
        sampled = rng.choice(len(idx_pairs), size=max_pairs, replace=False)
        idx_pairs = [idx_pairs[i] for i in sampled]

    cos_dists_a, cos_dists_b, fro_dists = [], [], []
    compat_pairs = 0

    for i, j in idx_pairs:
        wi, wj = weights[i], weights[j]
        if wi is None or wj is None:
            continue
        common_keys = set(wi) & set(wj)
        if not common_keys:
            continue
        ok = True
        for k in common_keys:
            if wi[k].shape != wj[k].shape:
                ok = False
                break
        if not ok:
            continue
        compat_pairs += 1
        a_keys = [k for k in common_keys if "lora_A" in k]
        b_keys = [k for k in common_keys if "lora_B" in k]
        if a_keys:
            va = torch.cat([wi[k].flatten() for k in a_keys])
            vb = torch.cat([wj[k].flatten() for k in a_keys])
            cos_dists_a.append(float(1.0 - F.cosine_similarity(va.unsqueeze(0), vb.unsqueeze(0)).item()))
        if b_keys:
            va = torch.cat([wi[k].flatten() for k in b_keys])
            vb = torch.cat([wj[k].flatten() for k in b_keys])
            cos_dists_b.append(float(1.0 - F.cosine_similarity(va.unsqueeze(0), vb.unsqueeze(0)).item()))
        # Frobenius on the whole parameter vector
        va = torch.cat([wi[k].flatten() for k in common_keys])
        vb = torch.cat([wj[k].flatten() for k in common_keys])
        fro_dists.append(float(torch.norm(va - vb).item()))

    return {
        "compatible_pairs": compat_pairs,
        "mean_cosine_distance_lora_A": float(np.mean(cos_dists_a)) if cos_dists_a else None,
        "mean_cosine_distance_lora_B": float(np.mean(cos_dists_b)) if cos_dists_b else None,
        "mean_frobenius_distance": float(np.mean(fro_dists)) if fro_dists else None,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--pool", required=True, type=Path)
    p.add_argument("--logits-cache", type=Path, default=None,
                   help="Directory with cached (pool_name__adapter_id).npy files from ensemble_eval")
    p.add_argument("--labels-path", type=Path, default=None,
                   help="Optional .npy of label array aligned to cached logits")
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--skip-weights", action="store_true",
                   help="Skip parameter-space metrics (faster)")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    pool = json.loads(args.pool.read_text())
    adapters = pool["adapters"]
    pool_name = pool["pool_name"]

    out: dict = {
        "pool_name": pool_name,
        "n_adapters": len(adapters),
        "hyperparameter_entropy": {
            key: metadata_entropy(adapters, key)
            for key in ("seed", "peft_method", "lora_r", "lora_alpha", "lr")
        },
    }

    if args.logits_cache:
        logits_list = []
        for a in adapters:
            cache_file = args.logits_cache / f"{pool_name}__{a['adapter_id']}.npy"
            if cache_file.exists():
                logits_list.append(np.load(cache_file))
        if logits_list:
            logits_stack = np.stack(logits_list, axis=0)
            preds = logits_stack.argmax(axis=-1)
            out["prediction_space"] = {
                "disagreement_rate": pairwise_disagreement(preds),
                "logit_correlation": pairwise_logit_correlation(logits_stack),
                "ensemble_entropy": ensemble_predictive_entropy(logits_stack),
            }
            if args.labels_path and args.labels_path.exists():
                labels = np.load(args.labels_path)
                out["prediction_space"]["error_jaccard"] = pairwise_error_jaccard(preds, labels)
        else:
            out["prediction_space"] = {"error": "no cached logits found"}

    if not args.skip_weights:
        out["parameter_space"] = pairwise_weight_distance(adapters)

    out["provenance"] = provenance({k: str(v) for k, v in vars(args).items()})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        json.dump(out, f, indent=2)
    print(f"[diversity_metrics] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
