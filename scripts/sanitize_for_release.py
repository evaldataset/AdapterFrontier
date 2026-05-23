#!/usr/bin/env python3
"""Sanitize release artifacts: mask absolute paths and hostnames in JSONs.

Run before uploading to anonymous.4open.science.

Replacements:
    /projects/Adapter/   -> ./
    /home/suan/          -> ~/
    oem-System-Product-Name -> anon-host
    (any author email)   -> anon@anon.invalid

Reports the count of affected files and stops on any failure.

Usage:
    python3 scripts/sanitize_for_release.py --dry-run   # report only
    python3 scripts/sanitize_for_release.py             # apply
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

PATTERNS = [
    (re.compile(r"/projects/Adapter/?"), "./"),
    (re.compile(r"/home/suan/?"), "~/"),
    (re.compile(r"oem-System-Product-Name"), "anon-host"),
    (re.compile(r"\bsuanlab@gmail\.com\b"), "anon@anon.invalid"),
    (re.compile(r"\bsuan\b(?!\w)"), "anon"),
]

SKIP_DIRS = {".git", "__pycache__", "node_modules", ".venv", "venv"}


def sanitize_text(s: str) -> tuple[str, int]:
    n = 0
    for pat, repl in PATTERNS:
        s, k = pat.subn(repl, s)
        n += k
    return s, n


def sanitize_file(path: Path, dry: bool) -> int:
    try:
        raw = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return 0
    new, n = sanitize_text(raw)
    if n > 0 and not dry:
        if path.suffix == ".json":
            try:
                json.loads(new)
            except json.JSONDecodeError:
                return -1
        path.write_text(new, encoding="utf-8")
    return n


def walk(root: Path):
    for p in root.rglob("*"):
        if any(part in SKIP_DIRS for part in p.parts):
            continue
        if not p.is_file():
            continue
        if p.suffix in {".json", ".md", ".py", ".sh", ".tex", ".cfg", ".yaml", ".yml", ".txt", ".csv"}:
            yield p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".", help="Repo root")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    by_ext: dict[str, list[Path]] = {}
    total_subs = 0
    err = 0
    for p in walk(root):
        n = sanitize_file(p, dry=args.dry_run)
        if n == -1:
            err += 1
            print(f"  ERROR (would break JSON): {p}")
            continue
        if n > 0:
            by_ext.setdefault(p.suffix, []).append(p)
            total_subs += n

    print(f"\n=== Sanitization {'PREVIEW' if args.dry_run else 'APPLIED'} ===")
    for ext in sorted(by_ext):
        print(f"  {ext}: {len(by_ext[ext])} files")
    print(f"Total substitutions: {total_subs}")
    if err:
        print(f"Errors (would break JSON): {err}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
