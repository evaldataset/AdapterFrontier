#!/usr/bin/env python3
"""Audit fix for Blocker 2: regenerate corpus-wide BH-FDR labels on every
per-cell compute-match JSON.

Background
----------
The paper (paper/emnlp.tex:194-203, 214, 651-655) states that the verdict
label `adjudication` carried in each `analysis/*_cm_*.json` reflects BOTH
the paired-bootstrap CI AND a corpus-wide BH-FDR (q=0.05) correction.
The May 2026 release of compute_match.py stored only the CI-based label
and a singleton-Holm field; the q-value column claimed by the paper was
missing from the per-cell JSONs. Reviewers comparing artifact to claim
would catch this.

What this script does
---------------------
1. Scans every `analysis/**/*_cm_*.json` file.
2. For each cell that has a `sign_flip_p`, computes a corpus-wide BH q-value
   (Benjamini-Hochberg step-up) and a corpus-wide Holm p (table-level Holm).
3. Writes back into each cell:
     - bh_q_value        — corpus-wide BH q
     - bh_pass           — bool, q <= --q
     - bh_q_threshold    — the q used (default 0.05)
     - bh_batch_size     — number of cells in the BH batch
     - bh_batch_hash     — sha256 of the sorted list of cell paths (audit trail)
     - holm_adjusted_p_batch — corpus-wide Holm adjusted p
     - adjudication_pre_fdr   — the original CI-only verdict
     - adjudication_post_fdr  — pre_fdr if bh_pass else "unsupported"
     - adjudication           — set to adjudication_post_fdr (paper-aligned)
4. Writes `analysis/corpus_bh_fdr_summary.json` with the corpus-wide cutoff,
   counts pre/post FDR, and the file list.

This rewrites JSON in place. Run from a clean repo and inspect the diff.
Cells without a `sign_flip_p` are skipped and recorded in the summary.

Usage:
    python3 scripts/apply_corpus_bh_fdr.py            # default q=0.05
    python3 scripts/apply_corpus_bh_fdr.py --q 0.05 --dry-run
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def bh_qvalues(pvals: list[float]) -> tuple[list[float], float]:
    """Benjamini-Hochberg step-up. Returns (q_values_in_input_order, max_p_passing_at_q).
    The second return is the largest p satisfying p <= k*q/m using q=1 (i.e.
    just the BH-style ordered cutoff p). The bh_pass decision uses q_value
    against the user-chosen q threshold, not this raw cutoff.
    """
    m = len(pvals)
    if m == 0:
        return [], float("nan")
    order = sorted(range(m), key=lambda i: pvals[i])
    # step-up: q_i = min_{j>=i} ( p_(j) * m / j )
    q_sorted = [0.0] * m
    running_min = 1.0
    for rank_minus_one in range(m - 1, -1, -1):
        i_one_based = rank_minus_one + 1
        p = pvals[order[rank_minus_one]]
        q = min(1.0, p * m / i_one_based)
        if q < running_min:
            running_min = q
        q_sorted[rank_minus_one] = running_min
    q_in_order = [0.0] * m
    for rank_idx, original_idx in enumerate(order):
        q_in_order[original_idx] = q_sorted[rank_idx]
    return q_in_order, float("nan")


def bh_cutoff_p(pvals: list[float], q: float) -> float:
    """Largest p_(k) satisfying p_(k) <= k*q/m. 0 if none."""
    m = len(pvals)
    if m == 0:
        return 0.0
    sp = sorted(pvals)
    cutoff = 0.0
    for k, p in enumerate(sp, 1):
        if p <= k * q / m:
            cutoff = p
    return cutoff


def holm_qvalues(pvals: list[float]) -> list[float]:
    """Holm step-down adjusted p-values, in input order."""
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adj = [0.0] * m
    running_max = 0.0
    for rank_idx, original_idx in enumerate(order):
        a = min(1.0, (m - rank_idx) * pvals[original_idx])
        running_max = max(running_max, a)
        adj[original_idx] = running_max
    return adj


def adj_post_fdr(adj_pre: str, bh_pass: bool) -> str:
    if not bh_pass:
        return "unsupported"
    return adj_pre


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--analysis-dir", default=ROOT / "analysis", type=Path)
    ap.add_argument("--glob", default="**/*_cm_*.json",
                    help="glob (relative to --analysis-dir) for per-cell JSONs")
    ap.add_argument("--q", type=float, default=0.05,
                    help="BH q threshold (default 0.05)")
    ap.add_argument("--summary-out", default=ROOT / "analysis/corpus_bh_fdr_summary.json",
                    type=Path)
    ap.add_argument("--dry-run", action="store_true",
                    help="Compute and report, but do not modify JSONs.")
    args = ap.parse_args()

    files = sorted(args.analysis_dir.glob(args.glob))
    if not files:
        print(f"No files matched {args.analysis_dir}/{args.glob}", file=sys.stderr)
        return 1
    print(f"[bh-fdr] found {len(files)} candidate cm files")

    cells = []
    skipped = []
    for fp in files:
        try:
            d = json.loads(fp.read_text())
        except Exception as e:
            skipped.append({"path": str(fp), "reason": f"json-load-error: {e}"})
            continue
        p = d.get("sign_flip_p")
        if not isinstance(p, (int, float)):
            skipped.append({"path": str(fp), "reason": "missing-sign_flip_p"})
            continue
        ci_lo = d.get("ci_low")
        ci_hi = d.get("ci_high")
        if ci_lo is None or ci_hi is None:
            skipped.append({"path": str(fp), "reason": "missing-ci"})
            continue
        cells.append({
            "path": fp, "p": float(p), "ci_lo": float(ci_lo), "ci_hi": float(ci_hi),
            "data": d,
        })

    print(f"[bh-fdr] usable cells: {len(cells)}  skipped: {len(skipped)}")
    if not cells:
        print("[bh-fdr] no usable cells; aborting", file=sys.stderr)
        return 1

    pvals = [c["p"] for c in cells]
    q_vals, _ = bh_qvalues(pvals)
    cutoff = bh_cutoff_p(pvals, args.q)
    holm_vals = holm_qvalues(pvals)

    # Content hash (TOP_PLAN §3.4): commit to WHAT was corrected, not just
    # which files. Each cell contributes (relpath, metric, delta, p, ci_lo,
    # ci_hi, method, baseline_kind); renaming files no longer preserves the
    # hash, and silently editing any statistic invalidates it.
    def cell_content_line(c):
        d = c["data"]
        delta = d.get("accuracy_diff", d.get("ece_diff"))
        method = (d.get("ensemble") or {}).get("method")
        bkind = (d.get("baseline") or {}).get("kind")
        rel = str(Path(c["path"]).resolve().relative_to(ROOT))
        return (f"{rel}|{d.get('metric')}|{delta:.10g}|{c['p']:.10g}"
                f"|{c['ci_lo']:.10g}|{c['ci_hi']:.10g}|{method}|{bkind}")

    batch_hash = hashlib.sha256(
        ("\n".join(sorted(cell_content_line(c) for c in cells))).encode()
    ).hexdigest()[:16]

    counts_pre = {"supported": 0, "reversed": 0, "unsupported": 0, "other": 0}
    counts_post = {"supported": 0, "reversed": 0, "unsupported": 0, "other": 0}
    for c, q, h in zip(cells, q_vals, holm_vals):
        ci_lo, ci_hi = c["ci_lo"], c["ci_hi"]
        pre = "supported" if ci_lo > 0 else ("reversed" if ci_hi < 0 else "unsupported")
        bh_pass = q <= args.q
        post = adj_post_fdr(pre, bh_pass)
        counts_pre[pre if pre in counts_pre else "other"] += 1
        counts_post[post if post in counts_post else "other"] += 1
        c["q"] = q
        c["bh_pass"] = bh_pass
        c["holm_batch"] = h
        c["pre"] = pre
        c["post"] = post

    total = len(cells)
    pct_pre_sup = 100 * counts_pre["supported"] / total
    pct_pre_rev = 100 * counts_pre["reversed"] / total
    pct_post_sup = 100 * counts_post["supported"] / total
    pct_post_rev = 100 * counts_post["reversed"] / total

    print(f"[bh-fdr] BH cutoff p (q={args.q}): {cutoff:.6f}")
    print(f"[bh-fdr] pre-FDR  supported {counts_pre['supported']:4d} ({pct_pre_sup:.1f}%)  "
          f"reversed {counts_pre['reversed']:4d} ({pct_pre_rev:.1f}%)  "
          f"unsupported {counts_pre['unsupported']:4d}")
    print(f"[bh-fdr] post-FDR supported {counts_post['supported']:4d} ({pct_post_sup:.1f}%)  "
          f"reversed {counts_post['reversed']:4d} ({pct_post_rev:.1f}%)  "
          f"unsupported {counts_post['unsupported']:4d}")
    print(f"[bh-fdr] batch hash: {batch_hash}")

    if args.dry_run:
        print("[bh-fdr] --dry-run set; no files modified")
    else:
        n_written = 0
        for c in cells:
            d = c["data"]
            d["bh_q_value"] = float(c["q"])
            d["bh_pass"] = bool(c["bh_pass"])
            d["bh_q_threshold"] = float(args.q)
            d["bh_batch_size"] = int(total)
            d["bh_batch_hash"] = batch_hash
            d["holm_adjusted_p_batch"] = float(c["holm_batch"])
            d["adjudication_pre_fdr"] = c["pre"]
            d["adjudication_post_fdr"] = c["post"]
            d["adjudication"] = c["post"]  # paper-aligned label = post-FDR
            d.setdefault("bh_fdr_note",
                         "Corpus-wide BH-FDR applied by scripts/apply_corpus_bh_fdr.py "
                         "(Audit response, Blocker 2 fix). bh_batch_hash pins the cell set "
                         "used in the correction.")
            c["path"].write_text(json.dumps(d, indent=2))
            n_written += 1
        print(f"[bh-fdr] wrote BH fields back into {n_written} cells")

    summary = {
        "q_threshold": args.q,
        "n_total_files_scanned": len(files),
        "n_cells_in_batch": total,
        "n_cells_skipped": len(skipped),
        "bh_cutoff_p_at_q": cutoff,
        "bh_batch_hash": batch_hash,
        "counts_pre_fdr": counts_pre,
        "counts_post_fdr": counts_post,
        "pct_pre_fdr": {
            "supported": round(pct_pre_sup, 2),
            "reversed": round(pct_pre_rev, 2),
            "unsupported": round(100 * counts_pre["unsupported"] / total, 2),
        },
        "pct_post_fdr": {
            "supported": round(pct_post_sup, 2),
            "reversed": round(pct_post_rev, 2),
            "unsupported": round(100 * counts_post["unsupported"] / total, 2),
        },
        "skipped_examples": skipped[:20],
    }
    args.summary_out.parent.mkdir(parents=True, exist_ok=True)
    args.summary_out.write_text(json.dumps(summary, indent=2))
    print(f"[bh-fdr] summary written to {args.summary_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
