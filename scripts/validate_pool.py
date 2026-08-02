#!/usr/bin/env python3
"""Validate a pool manifest against the v1 schema.

Exit codes:
    0 — valid
    1 — schema violations
    2 — file/IO error

Usage:
    python3 scripts/validate_pool.py pools/pool_a_mnli_bert.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REQUIRED_TOP = (
    "schema_version", "pool_id", "pool_type", "n_adapters",
    "created_at_utc", "adapters", "provenance",
)
REQUIRED_ADAPTER = ("adapter_id", "path", "metadata", "metrics")
SUPPORTED_SCHEMA = (1,)


def validate(manifest: dict, manifest_path: Path) -> list[str]:
    errors: list[str] = []

    for k in REQUIRED_TOP:
        if k not in manifest:
            errors.append(f"missing top-level key: {k}")

    sv = manifest.get("schema_version")
    if sv not in SUPPORTED_SCHEMA:
        errors.append(f"unsupported schema_version={sv} (supported: {SUPPORTED_SCHEMA})")

    adapters = manifest.get("adapters", [])
    if not isinstance(adapters, list):
        errors.append("'adapters' must be a list")
        return errors

    if manifest.get("n_adapters") != len(adapters):
        errors.append(
            f"n_adapters={manifest.get('n_adapters')} but len(adapters)={len(adapters)}"
        )

    seen_ids: set[str] = set()
    for i, a in enumerate(adapters):
        prefix = f"adapters[{i}]"
        if not isinstance(a, dict):
            errors.append(f"{prefix} not a dict")
            continue
        for k in REQUIRED_ADAPTER:
            if k not in a:
                errors.append(f"{prefix} missing key: {k}")

        aid = a.get("adapter_id")
        if aid in seen_ids:
            errors.append(f"{prefix} duplicate adapter_id: {aid}")
        seen_ids.add(aid)

        p = a.get("path")
        if p:
            ap = Path(p)
            if not ap.exists():
                errors.append(f"{prefix} path does not exist: {p}")
            elif not (ap / "adapter_config.json").exists():
                errors.append(f"{prefix} no adapter_config.json under path: {p}")

    return errors


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("manifest", type=Path)
    ap.add_argument("--strict-paths", action="store_true",
                    help="Treat missing adapter directories as a hard failure (default already does this)")
    args = ap.parse_args()

    try:
        manifest = json.loads(args.manifest.read_text())
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"FAIL: cannot read {args.manifest}: {e}", file=sys.stderr)
        return 2

    errors = validate(manifest, args.manifest)
    if errors:
        print(f"FAIL: {len(errors)} schema violation(s) in {args.manifest}", file=sys.stderr)
        for e in errors:
            print(f"  - {e}", file=sys.stderr)
        return 1
    print(
        f"OK: {args.manifest} — pool_id={manifest['pool_id']}, "
        f"pool_type={manifest['pool_type']}, n_adapters={manifest['n_adapters']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
