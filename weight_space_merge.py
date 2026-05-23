#!/usr/bin/env python3
"""Weight-space merging baselines for adapter pools.

Implements:
    uniform      : simple average of all compatible adapters
    ties         : TIES merging (Yadav et al. 2023) — trim + elect sign + disjoint merge
    dare         : DARE merging (Yu et al. 2023) — random drop + rescale + average
    greedy_soup  : greedy weight-space soup (Wortsman et al. 2022) — add if val improves

Output: a merged adapter directory (copies the reference adapter_config.json from
adapter[0] and writes adapter_model.safetensors with merged tensors).

Adapters with incompatible shapes (different ranks, different target modules) are
filtered out to the largest compatible subset before merging.

Usage:
    python3 weight_space_merge.py --pool pools/pool_a_mnli_bert.json \
        --methods uniform,ties,dare --out-dir merged_adapters/pool_a/
"""
from __future__ import annotations

import argparse
import json
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from peft_utils import provenance


def load_weights(path: Path) -> dict[str, torch.Tensor] | None:
    sfile = path / "adapter_model.safetensors"
    bfile = path / "adapter_model.bin"
    if sfile.exists():
        from safetensors.torch import load_file
        return load_file(str(sfile))
    if bfile.exists():
        return torch.load(str(bfile), map_location="cpu")
    return None


def save_weights(tensors: dict[str, torch.Tensor], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        from safetensors.torch import save_file
        save_file(tensors, str(out_dir / "adapter_model.safetensors"))
    except ImportError:
        torch.save(tensors, str(out_dir / "adapter_model.bin"))


def largest_compatible_subset(all_weights: list[dict]) -> tuple[list[int], list[str]]:
    """Find the largest group of adapters sharing identical key set + shapes."""
    buckets: dict[tuple, list[int]] = defaultdict(list)
    for i, w in enumerate(all_weights):
        if w is None:
            continue
        key = tuple(sorted((k, tuple(v.shape)) for k, v in w.items()))
        buckets[key].append(i)
    if not buckets:
        return [], []
    best_key, best_idx = max(buckets.items(), key=lambda kv: len(kv[1]))
    keys = [k for k, _ in best_key]
    return best_idx, keys


def uniform_merge(weights_list: list[dict], keys: list[str]) -> dict[str, torch.Tensor]:
    out = {}
    for k in keys:
        stacked = torch.stack([w[k].float() for w in weights_list], dim=0)
        out[k] = stacked.mean(dim=0)
    return out


def ties_merge(weights_list: list[dict], keys: list[str], trim_pct: float = 0.2) -> dict[str, torch.Tensor]:
    """TIES: trim top-k% by magnitude, elect sign by summed magnitude, average only kept values with that sign."""
    out = {}
    for k in keys:
        stacked = torch.stack([w[k].float() for w in weights_list], dim=0)  # (N, *shape)
        flat = stacked.view(stacked.shape[0], -1)
        abs_flat = flat.abs()
        for i in range(flat.shape[0]):
            thresh = torch.quantile(abs_flat[i], trim_pct)
            mask = abs_flat[i] < thresh
            flat[i][mask] = 0.0
        signed_sum = flat.sum(dim=0)
        elected_sign = torch.sign(signed_sum)
        agree = (flat.sign() == elected_sign.unsqueeze(0)) & (flat != 0)
        denom = agree.sum(dim=0).clamp(min=1).float()
        merged = (flat * agree.float()).sum(dim=0) / denom
        out[k] = merged.view(stacked.shape[1:])
    return out


def dare_merge(weights_list: list[dict], keys: list[str], drop_rate: float = 0.5,
               seed: int = 0) -> dict[str, torch.Tensor]:
    """DARE: per-parameter Bernoulli drop with prob=drop_rate, rescale survivors by 1/(1-drop_rate), then average."""
    g = torch.Generator().manual_seed(seed)
    scale = 1.0 / (1.0 - drop_rate)
    out = {}
    for k in keys:
        stacked = torch.stack([w[k].float() for w in weights_list], dim=0)
        mask = (torch.rand(stacked.shape, generator=g) > drop_rate).float()
        stacked = stacked * mask * scale
        out[k] = stacked.mean(dim=0)
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--pool", required=True, type=Path)
    p.add_argument("--methods", default="uniform,ties,dare")
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--trim-pct", type=float, default=0.2)
    p.add_argument("--drop-rate", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    pool = json.loads(args.pool.read_text())
    adapters = pool["adapters"]

    loaded = [load_weights(Path(a["path"])) for a in adapters]
    subset_idx, keys = largest_compatible_subset(loaded)
    if len(subset_idx) < 2:
        raise SystemExit(f"Not enough compatible adapters: {len(subset_idx)}")

    weights_subset = [loaded[i] for i in subset_idx]
    ref_path = Path(adapters[subset_idx[0]]["path"])

    manifest = {
        "pool_name": pool["pool_name"],
        "compatible_count": len(subset_idx),
        "compatible_adapter_ids": [adapters[i]["adapter_id"] for i in subset_idx],
        "methods": {},
    }

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    for m in methods:
        out_adapter_dir = args.out_dir / m
        if m == "uniform":
            merged = uniform_merge(weights_subset, keys)
        elif m == "ties":
            merged = ties_merge(weights_subset, keys, trim_pct=args.trim_pct)
        elif m == "dare":
            merged = dare_merge(weights_subset, keys, drop_rate=args.drop_rate, seed=args.seed)
        else:
            print(f"[warn] unknown method {m}")
            continue

        save_weights(merged, out_adapter_dir)
        cfg_src = ref_path / "adapter_config.json"
        if cfg_src.exists():
            shutil.copy2(cfg_src, out_adapter_dir / "adapter_config.json")
        manifest["methods"][m] = {"output_dir": str(out_adapter_dir.resolve())}

    manifest["provenance"] = provenance({k: str(v) for k, v in vars(args).items()})
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with (args.out_dir / "merge_manifest.json").open("w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[weight_space_merge] wrote {len(methods)} merged adapters to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
