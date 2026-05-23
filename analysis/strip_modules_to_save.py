#!/usr/bin/env python3
"""Create vLLM-compatible adapter copies by stripping modules_to_save.

vLLM multi-LoRA serving doesn't support modules_to_save (classification heads),
so for §9 inference benchmarking we strip them. The resulting adapters keep
only the LoRA-A/B matrices over q_proj/v_proj — the actual cost driver for
multi-tenant serving.

Usage:
    python3 analysis/strip_modules_to_save.py \\
        --pool pools/pool_a_mnli_qwen25_05b.json \\
        --out-dir collected_adapters/pool_a_mnli_qwen25_05b_vllm/ \\
        --n-adapters 8
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import safetensors.torch as st


def strip_one(src: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((src / "adapter_config.json").read_text())
    cfg["modules_to_save"] = None
    (dst / "adapter_config.json").write_text(json.dumps(cfg, indent=2))

    src_st = src / "adapter_model.safetensors"
    tensors = st.load_file(str(src_st))
    # Drop classifier/score keys; keep only LoRA-{A,B} params.
    keep = {k: v for k, v in tensors.items()
            if "lora_" in k and "modules_to_save" not in k}
    st.save_file(keep, str(dst / "adapter_model.safetensors"))

    # Copy tokenizer + readme so vllm load doesn't complain.
    for opt in ("README.md", "tokenizer.json", "tokenizer_config.json",
                "chat_template.jinja"):
        if (src / opt).exists():
            shutil.copy(src / opt, dst / opt)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--n-adapters", type=int, default=8)
    args = ap.parse_args()

    pool = json.loads(args.pool.read_text())
    adapters = pool["adapters"][:args.n_adapters]
    args.out_dir.mkdir(parents=True, exist_ok=True)

    new_adapters = []
    for a in adapters:
        src = Path(a["path"])
        dst = args.out_dir / src.name
        strip_one(src, dst)
        new_adapters.append({**a, "path": str(dst.resolve())})
        print(f"  stripped: {src.name}")

    new_pool = {**pool, "adapters": new_adapters,
                "pool_id": pool["pool_id"] + "_vllm",
                "_note": "modules_to_save stripped for vLLM serving compatibility"}
    pool_out = args.out_dir.parent / (args.out_dir.name + "_pool.json")
    pool_out.parent.mkdir(parents=True, exist_ok=True)
    pool_out.write_text(json.dumps(new_pool, indent=2))
    print(f"\nwrote {len(new_adapters)} stripped adapters → {args.out_dir}")
    print(f"new pool manifest → {pool_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
