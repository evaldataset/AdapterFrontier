"""Shared PEFT configuration utilities used by real_lora_train.py, real_lora_classify.py, and train_and_eval_ood.py."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from peft import LoraConfig


def build_peft_config(
    peft_method: str,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    target_modules: list[str],
    task_type: str = "CAUSAL_LM",
    pissa_niter: int = 0,
) -> LoraConfig:
    """Build a LoraConfig for the given PEFT method."""
    init_lora_weights: bool | str = True
    use_dora = False
    use_rslora = False
    if peft_method == "dora":
        use_dora = True
    elif peft_method == "pissa":
        init_lora_weights = (
            f"pissa_niter_{pissa_niter}" if pissa_niter > 0 else "pissa"
        )
    elif peft_method == "rslora":
        use_rslora = True

    return LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        bias="none",
        task_type=task_type,
        init_lora_weights=init_lora_weights,
        use_dora=use_dora,
        use_rslora=use_rslora,
    )


def detect_lora_targets(model) -> list[str]:
    """Auto-detect LoRA target module names from a model."""
    known_prefixes = {
        "BertForSequenceClassification": ["query", "value"],
        "RobertaForSequenceClassification": ["query", "value"],
        "DebertaV2ForSequenceClassification": ["query_proj", "value_proj"],
        "DebertaForSequenceClassification": ["in_proj"],
        "LlamaForSequenceClassification": ["q_proj", "v_proj"],
        "GemmaForSequenceClassification": ["q_proj", "v_proj"],
        "MistralForSequenceClassification": ["q_proj", "v_proj"],
        "Qwen2ForSequenceClassification": ["q_proj", "v_proj"],
        "Qwen3ForSequenceClassification": ["q_proj", "v_proj"],
        "Phi3ForSequenceClassification": ["qkv_proj"],
        "GPTNeoXForSequenceClassification": ["query_key_value"],
    }
    cls_name = model.__class__.__name__
    for prefix, targets in known_prefixes.items():
        if cls_name.startswith(prefix):
            return targets

    valid = {
        "query", "value", "query_proj", "value_proj",
        "q_proj", "k_proj", "v_proj", "o_proj", "in_proj",
    }
    found = set()
    for name, _ in model.named_modules():
        leaf = name.split(".")[-1]
        if leaf in valid:
            found.add(leaf)
    if found:
        preferred_order = [
            "query", "value", "query_proj", "value_proj",
            "q_proj", "v_proj", "o_proj", "in_proj", "k_proj",
        ]
        ordered = [n for n in preferred_order if n in found]
        return ordered if ordered else sorted(found)

    return ["q_proj", "v_proj"]


# ---------------------------------------------------------------------------
# Reproducibility / provenance
# ---------------------------------------------------------------------------

PROVENANCE_SCHEMA_VERSION = 1


def _git_info(repo_dir: Path) -> tuple[str, bool]:
    """Return (sha, dirty). Returns ('unknown', False) outside a git repo."""
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo_dir, stderr=subprocess.DEVNULL
        ).decode().strip()
        status = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=repo_dir, stderr=subprocess.DEVNULL
        ).decode().strip()
        return sha, bool(status)
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown", False


def _lib_version(name: str) -> str:
    try:
        mod = __import__(name)
        return getattr(mod, "__version__", "unknown")
    except ImportError:
        return "not-installed"


def config_hash(config: dict) -> str:
    """Stable sha256 of a config dict (sorted keys, separators normalized)."""
    payload = json.dumps(config, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode()).hexdigest()


def provenance(config: dict | None = None, repo_dir: Path | None = None) -> dict:
    """Return a provenance block to embed in every result JSON.

    Pass the run's normalized config (e.g. vars(args)) to get a config_hash.
    All Phase 0 result writers (real_lora_classify, real_lora_train,
    ensemble_eval, diversity_metrics, weight_space_merge) must call this
    and write the dict under the top-level key "provenance".
    """
    repo_dir = repo_dir or Path(__file__).resolve().parent
    sha, dirty = _git_info(repo_dir)
    return {
        "schema_version": PROVENANCE_SCHEMA_VERSION,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "git_sha": sha,
        "git_dirty": dirty,
        "config_hash": config_hash(config) if config is not None else None,
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "torch_version": _lib_version("torch"),
        "transformers_version": _lib_version("transformers"),
        "peft_version": _lib_version("peft"),
        "datasets_version": _lib_version("datasets"),
        "numpy_version": _lib_version("numpy"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
