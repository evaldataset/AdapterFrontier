#!/usr/bin/env python3
"""Scan a completed sweep directory and build a pool manifest JSON.

A pool manifest is a flat JSON list where each entry is an adapter:
    {
      "adapter_id": "seed_11",
      "path": "/abs/path/to/adapter_dir",
      "metadata": {"seed": 11, "peft_method": "lora", "lora_r": 8, ...},
      "metrics": {"eval_accuracy": 0.847, "eval_loss": 0.42, ...}
    }

Usage:
    python3 build_adapter_pool.py --sweep-dir sweep_runs/pool_a_mnli_bert --out pools/pool_a_mnli_bert.json
"""
from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path

from peft_utils import provenance

POOL_SCHEMA_VERSION = 1
HYPERPARAM_KEYS = (
    "lora_r", "lora_alpha", "lora_dropout", "lr", "seed",
    "peft_method", "target_modules", "batch_size", "num_epochs",
    "max_samples", "warmup_ratio", "weight_decay",
)


def find_adapter_dirs(sweep_dir: Path) -> list[Path]:
    """Locate directories containing adapter_config.json (peft adapters)."""
    return sorted(p.parent for p in sweep_dir.rglob("adapter_config.json"))


def load_metrics(adapter_dir: Path) -> dict:
    """Try multiple known metric file names."""
    for name in ("final_metrics.json", "eval_results.json", "metrics.json", "all_results.json"):
        candidate = adapter_dir / name
        if candidate.exists():
            with candidate.open() as f:
                return json.load(f)
    return {}


def load_adapter_config(adapter_dir: Path) -> dict:
    cfg_path = adapter_dir / "adapter_config.json"
    if not cfg_path.exists():
        return {}
    with cfg_path.open() as f:
        return json.load(f)


def parse_dir_name(name: str) -> dict:
    """Extract metadata from patterns like 'seed_11' or 'r8_a16_lr3e-4' or 'lora_seed_11'."""
    meta: dict = {}
    m = re.search(r"seed[_=](\d+)", name)
    if m:
        meta["seed"] = int(m.group(1))
    m = re.search(r"r(\d+)_a(\d+)", name)
    if m:
        meta["lora_r"] = int(m.group(1))
        meta["lora_alpha"] = int(m.group(2))
    m = re.search(r"lr([0-9eE\.\-+]+)", name)
    if m:
        try:
            meta["lr"] = float(m.group(1))
        except ValueError:
            pass
    for method in ("lora", "dora", "rslora", "pissa", "adalora"):
        if name.lower().startswith(method + "_") or f"/{method}_" in name or f"_{method}_" in name:
            meta["peft_method"] = method
            break
    return meta


def build_manifest(sweep_dir: Path) -> list[dict]:
    entries: list[dict] = []
    for adir in find_adapter_dirs(sweep_dir):
        rel_id = adir.relative_to(sweep_dir).as_posix().replace("/", "__") or adir.name
        cfg = load_adapter_config(adir)
        metrics = load_metrics(adir)
        meta = parse_dir_name(adir.name)
        if "peft_method" not in meta:
            meta["peft_method"] = cfg.get("peft_type", "").lower() or "lora"
        if "lora_r" not in meta and "r" in cfg:
            meta["lora_r"] = cfg["r"]
        if "lora_alpha" not in meta and "lora_alpha" in cfg:
            meta["lora_alpha"] = cfg["lora_alpha"]
        meta["base_model"] = cfg.get("base_model_name_or_path", "")
        meta["target_modules"] = cfg.get("target_modules", [])

        # v1 schema: split hyperparams into a normalized sub-block so downstream
        # consumers can rely on fixed locations. Anything else stays in metadata.
        hyperparams = {k: meta[k] for k in HYPERPARAM_KEYS if k in meta}

        entries.append({
            "adapter_id": rel_id,
            "path": str(adir.resolve()),
            "peft_method": meta.get("peft_method"),
            "base_model": meta.get("base_model"),
            "hyperparams": hyperparams,
            "metadata": meta,  # keep full metadata blob for backward compat
            "metrics": {k: v for k, v in metrics.items() if isinstance(v, (int, float, str))},
            "provenance_from_metrics": metrics.get("provenance"),  # carry through if present
        })
    return entries


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--sweep-dir", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path, help="Pool manifest JSON output path")
    p.add_argument("--min-adapters", type=int, default=2)
    p.add_argument("--pool-type", default="unknown",
                   help="e.g. seed_only, hp_diverse, method_mixed, baseline")
    p.add_argument("--task", default=None, help="Primary task id (e.g. mnli, sst2)")
    p.add_argument("--pool-id", default=None,
                   help="Override pool_id (defaults to sweep_dir name)")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if not args.sweep_dir.exists():
        raise SystemExit(f"Sweep directory not found: {args.sweep_dir}")

    entries = build_manifest(args.sweep_dir)
    if len(entries) < args.min_adapters:
        raise SystemExit(f"Only {len(entries)} adapters found; need >= {args.min_adapters}")

    base_models = {e["base_model"] for e in entries if e.get("base_model")}
    base_model = next(iter(base_models)) if len(base_models) == 1 else None

    manifest = {
        "schema_version": POOL_SCHEMA_VERSION,
        "pool_id": args.pool_id or args.sweep_dir.name,
        "pool_name": args.pool_id or args.sweep_dir.name,  # alias for backward compat
        "pool_type": args.pool_type,
        "task": args.task,
        "base_model": base_model,
        "n_adapters": len(entries),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "adapters": entries,
        "provenance": provenance({k: str(v) for k, v in vars(args).items()}),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    # Convert adapter paths to relative-from-manifest so the released JSON does
    # not embed absolute paths that break on download or leak the author setup.
    manifest_dir = args.out.parent.resolve()
    for e in manifest["adapters"]:
        try:
            e["path"] = str(Path(e["path"]).resolve().relative_to(manifest_dir.parent))
        except (ValueError, OSError):
            pass  # path not under manifest_dir.parent; keep as-is
    with args.out.open("w") as f:
        json.dump(manifest, f, indent=2)

    print(f"[build_adapter_pool] Wrote {len(entries)} adapters to {args.out} (pool_id={manifest['pool_id']}, pool_type={args.pool_type})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
