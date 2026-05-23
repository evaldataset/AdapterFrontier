#!/usr/bin/env python3
"""Compute-matched comparison between an ensemble and a single-adapter baseline.

This module is the **single source of truth** for ensemble-vs-baseline claims.
Per CLAUDE.md "Statistical Protocol", every cell reported in the paper goes
through here. Output JSON is canonical and downstream-stable.

Baseline kinds:
    best_of_n     — best individual adapter from the ensemble's own pool
                    (selected on val_selection internally by ensemble_eval)
    n_rank        — single adapter trained with N× the base rank
    n_steps       — single adapter trained with N× the training steps
    n_data        — single adapter trained with N× the training data

For best_of_n, the baseline data is read from the ensemble result file itself
(method "best_single"). For the others, pass --baseline pointing at a separate
ensemble_eval run on a singleton pool.

Usage:
    # ensemble vs best-of-N single
    python3 analysis/compute_match.py \
        --ensemble ensemble_results/pool_a_mnli_bert.json \
        --method greedy_soup --baseline-kind best_of_n \
        --out analysis/cm_pool_a_greedy_vs_best.json

    # ensemble vs N-times-rank single
    python3 analysis/compute_match.py \
        --ensemble ensemble_results/pool_a_mnli_bert.json --method greedy_soup \
        --baseline ensemble_results/baseline_n_rank_mnli_bert.json --baseline-method best_single \
        --baseline-kind n_rank \
        --out analysis/cm_pool_a_greedy_vs_n_rank.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

import numpy as np

# Reuse the project's existing paired-bootstrap implementation if present;
# otherwise fall back to a local implementation.
try:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from project_a_pairwise_summary import (  # type: ignore
        paired_bootstrap_diff_ci,
        sign_flip_pvalue,
    )
    _HAS_PROJECT_STATS = True
except Exception:
    _HAS_PROJECT_STATS = False

from peft_utils import provenance  # noqa: E402

SCHEMA_VERSION = 1
BASELINE_KINDS = ("best_of_n", "n_rank", "n_steps", "n_data")
METRICS = ("accuracy", "ece")  # brier needs full prob distribution; deferred


# ---------------------------------------------------------------------------
# Statistics fallback (used only when project_a_pairwise_summary is absent
# or not compatible). Bootstrap and sign-flip implementations are intentionally
# explicit so the math is reviewable inline.
# ---------------------------------------------------------------------------

def _paired_bootstrap_diff_ci(
    correct_a: np.ndarray, correct_b: np.ndarray,
    n_boot: int = 5000, alpha: float = 0.05, seed: int = 0,
) -> tuple[float, float, float]:
    """Percentile bootstrap CI on mean(correct_a - correct_b)."""
    assert correct_a.shape == correct_b.shape
    rng = np.random.RandomState(seed)
    n = len(correct_a)
    diffs = correct_a - correct_b
    obs = float(diffs.mean())
    boots = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        idx = rng.randint(0, n, size=n)
        boots[b] = float(diffs[idx].mean())
    lo = float(np.quantile(boots, alpha / 2))
    hi = float(np.quantile(boots, 1 - alpha / 2))
    return obs, lo, hi


def _sign_flip_pvalue(
    correct_a: np.ndarray, correct_b: np.ndarray,
    n_perm: int = 10000, seed: int = 0,
) -> float:
    """Two-sided sign-flip permutation p-value on paired correct-vector diffs."""
    diffs = (correct_a - correct_b).astype(np.float64)
    obs = abs(diffs.mean())
    rng = np.random.RandomState(seed)
    n = len(diffs)
    extreme = 0
    for _ in range(n_perm):
        signs = rng.choice([-1.0, 1.0], size=n)
        if abs((diffs * signs).mean()) >= obs:
            extreme += 1
    return (extreme + 1) / (n_perm + 1)


# ---------------------------------------------------------------------------
# ECE on equal-mass bins (claim-C arm)
# ---------------------------------------------------------------------------

def _ece_equal_mass(confidences: np.ndarray, correct: np.ndarray, n_bins: int = 15) -> float:
    """Equal-mass binning ECE for a single (confidence, correctness) sample."""
    n = len(confidences)
    if n == 0:
        return float("nan")
    order = np.argsort(confidences)
    bins = np.array_split(order, min(n_bins, n))
    ece = 0.0
    for b in bins:
        if len(b) == 0:
            continue
        gap = abs(confidences[b].mean() - correct[b].mean())
        ece += (len(b) / n) * gap
    return float(ece)


def _paired_ece_diff_ci(
    conf_a: np.ndarray, correct_a: np.ndarray,
    conf_b: np.ndarray, correct_b: np.ndarray,
    n_boot: int = 5000, alpha: float = 0.05, seed: int = 0, n_bins: int = 15,
) -> tuple[float, float, float]:
    """Bootstrap CI on (ECE_baseline - ECE_ensemble). Positive = ensemble more calibrated.

    'a' = ensemble, 'b' = baseline. We resample indices and recompute ECE
    for both on the same resampled set, then take ECE_b - ECE_a.
    """
    assert len(conf_a) == len(conf_b) == len(correct_a) == len(correct_b)
    rng = np.random.RandomState(seed)
    n = len(conf_a)
    obs = _ece_equal_mass(conf_b, correct_b, n_bins) - _ece_equal_mass(conf_a, correct_a, n_bins)
    boots = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        idx = rng.randint(0, n, size=n)
        e_a = _ece_equal_mass(conf_a[idx], correct_a[idx], n_bins)
        e_b = _ece_equal_mass(conf_b[idx], correct_b[idx], n_bins)
        boots[i] = e_b - e_a
    lo = float(np.quantile(boots, alpha / 2))
    hi = float(np.quantile(boots, 1 - alpha / 2))
    return float(obs), lo, hi


def _sign_flip_pvalue_ece(
    conf_a: np.ndarray, correct_a: np.ndarray,
    conf_b: np.ndarray, correct_b: np.ndarray,
    n_perm: int = 5000, seed: int = 0, n_bins: int = 15,
) -> float:
    """Permutation test for ECE diff: swap (conf_a/correct_a) vs (conf_b/correct_b)
    pointwise across examples. Tests H0: ensemble & baseline come from same
    confidence-correctness joint distribution.
    """
    obs = abs(_ece_equal_mass(conf_b, correct_b, n_bins) - _ece_equal_mass(conf_a, correct_a, n_bins))
    rng = np.random.RandomState(seed)
    n = len(conf_a)
    extreme = 0
    for _ in range(n_perm):
        flip = rng.rand(n) > 0.5
        ca = np.where(flip, conf_b, conf_a)
        cb = np.where(flip, conf_a, conf_b)
        ka = np.where(flip, correct_b, correct_a)
        kb = np.where(flip, correct_a, correct_b)
        d = abs(_ece_equal_mass(cb, kb, n_bins) - _ece_equal_mass(ca, ka, n_bins))
        if d >= obs:
            extreme += 1
    return (extreme + 1) / (n_perm + 1)


def paired_stats(correct_a: np.ndarray, correct_b: np.ndarray, seed: int) -> dict:
    if _HAS_PROJECT_STATS:
        try:
            obs, lo, hi = paired_bootstrap_diff_ci(correct_a, correct_b, n_boot=5000, seed=seed)
            p = sign_flip_pvalue(correct_a, correct_b, seed=seed)
            return {"observed_diff": float(obs), "ci_low": float(lo), "ci_high": float(hi),
                    "sign_flip_p": float(p), "stats_backend": "project_a_pairwise_summary"}
        except TypeError:
            pass  # signature drift -> fall back
    obs, lo, hi = _paired_bootstrap_diff_ci(correct_a, correct_b, seed=seed)
    p = _sign_flip_pvalue(correct_a, correct_b, seed=seed)
    return {"observed_diff": obs, "ci_low": lo, "ci_high": hi,
            "sign_flip_p": p, "stats_backend": "compute_match.local"}


# ---------------------------------------------------------------------------
# Holm-Bonferroni correction across a batch of cells
# ---------------------------------------------------------------------------

def holm_correct(pvals: list[float]) -> list[float]:
    """Holm step-down correction. Returns adjusted p-values in original order."""
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m, dtype=np.float64)
    running_max = 0.0
    for rank, idx in enumerate(order):
        a = (m - rank) * pvals[idx]
        a = min(a, 1.0)
        running_max = max(running_max, a)
        adj[idx] = running_max
    return adj.tolist()


# ---------------------------------------------------------------------------
# Result loading & alignment
# ---------------------------------------------------------------------------

def _load_method_preds(result_path: Path, method: str, *, need_confidence: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, dict]:
    data = json.loads(result_path.read_text())
    if method not in data.get("methods", {}):
        raise SystemExit(f"{result_path}: method {method!r} not present in result file")
    method_block = data["methods"][method]
    if "predictions_test" not in method_block:
        raise SystemExit(f"{result_path} method {method!r} has no predictions_test (re-run ensemble_eval)")
    if "labels_test" not in data:
        raise SystemExit(f"{result_path}: missing labels_test (re-run ensemble_eval)")
    conf = None
    if need_confidence:
        if "confidence_test" not in method_block:
            raise SystemExit(
                f"{result_path} method {method!r} has no confidence_test "
                "(re-run ensemble_eval after Phase 1.5 patch to enable --metric ece)"
            )
        conf = np.asarray(method_block["confidence_test"], dtype=np.float64)
    return (
        np.asarray(method_block["predictions_test"], dtype=np.int64),
        np.asarray(data["labels_test"], dtype=np.int64),
        conf,
        data,
    )


def _align(ens_labels, base_labels) -> None:
    """Both files must have the same labels_test (i.e. same eval split)."""
    if ens_labels.shape != base_labels.shape or not np.array_equal(ens_labels, base_labels):
        raise SystemExit(
            "Ensemble and baseline labels_test do not match — they were not evaluated "
            "on the same split. Re-run ensemble_eval with the same --task, --split-seed, "
            "and --max-eval-samples on both sides."
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--ensemble", required=True, type=Path,
                   help="Ensemble result JSON from ensemble_eval.py")
    p.add_argument("--method", required=True,
                   help="Which ensemble method (e.g. greedy_soup, soft_vote, logit_avg)")
    p.add_argument("--baseline-kind", required=True, choices=BASELINE_KINDS)
    p.add_argument("--baseline", type=Path, default=None,
                   help="Baseline result JSON (required for n_rank/n_steps/n_data; "
                        "ignored for best_of_n which uses the ensemble file's best_single)")
    p.add_argument("--baseline-method", default="best_single",
                   help="Method name inside the baseline file to compare against (default best_single)")
    p.add_argument("--metric", default="accuracy", choices=METRICS,
                   help="Which metric to compare (accuracy diff or ECE diff). "
                        "ece requires confidence_test in result JSON (Phase 1.5+)")
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--n-boot", type=int, default=5000)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    need_conf = args.metric == "ece"

    ens_preds, ens_labels, ens_conf, ens_meta = _load_method_preds(
        args.ensemble, args.method, need_confidence=need_conf,
    )

    if args.baseline_kind == "best_of_n":
        base_preds, base_labels, base_conf, base_meta = _load_method_preds(
            args.ensemble, "best_single", need_confidence=need_conf,
        )
        baseline_method_name = "best_single"
    else:
        if args.baseline is None:
            raise SystemExit(f"--baseline is required when --baseline-kind={args.baseline_kind}")
        base_preds, base_labels, base_conf, base_meta = _load_method_preds(
            args.baseline, args.baseline_method, need_confidence=need_conf,
        )
        baseline_method_name = args.baseline_method

    _align(ens_labels, base_labels)
    correct_a = (ens_preds == ens_labels).astype(np.float64)
    correct_b = (base_preds == base_labels).astype(np.float64)

    if args.metric == "accuracy":
        stats = paired_stats(correct_a, correct_b, seed=args.seed)
        diff_key = "accuracy_diff"
        diff_val = stats["observed_diff"]
        ci_lo, ci_hi = stats["ci_low"], stats["ci_high"]
        pval = stats["sign_flip_p"]
        backend = stats["stats_backend"]
    else:  # ece
        diff_val, ci_lo, ci_hi = _paired_ece_diff_ci(
            ens_conf, correct_a, base_conf, correct_b,
            n_boot=args.n_boot, seed=args.seed,
        )
        pval = _sign_flip_pvalue_ece(
            ens_conf, correct_a, base_conf, correct_b, seed=args.seed,
        )
        diff_key = "ece_diff"  # positive = ensemble more calibrated (lower ECE)
        backend = "compute_match.local_ece"

    holm_adjusted = holm_correct([pval])[0]

    n_adapters = ens_meta.get("n_adapters")
    out = {
        "schema_version": SCHEMA_VERSION,
        "metric": args.metric,
        "ensemble": {
            "result_path": str(args.ensemble.resolve()),
            "pool_id": ens_meta.get("pool_id") or ens_meta.get("pool_name"),
            "pool_type": ens_meta.get("pool_type"),
            "method": args.method,
            "n_adapters": n_adapters,
            "accuracy_test": ens_meta["methods"][args.method].get("accuracy_test"),
            "ece_test": ens_meta["methods"][args.method].get("calibration", {}).get("ece_15bin_equalmass"),
        },
        "baseline": {
            "kind": args.baseline_kind,
            "result_path": str(args.baseline.resolve()) if args.baseline else str(args.ensemble.resolve()),
            "pool_id": base_meta.get("pool_id") or base_meta.get("pool_name"),
            "method": baseline_method_name,
            "accuracy_test": base_meta["methods"][baseline_method_name].get("accuracy_test"),
            "ece_test": base_meta["methods"][baseline_method_name].get("calibration", {}).get("ece_15bin_equalmass"),
        },
        "n_test": int(len(correct_a)),
        diff_key: diff_val,
        "ci_low": ci_lo,
        "ci_high": ci_hi,
        "sign_flip_p": pval,
        "holm_adjusted_p_singleton": holm_adjusted,
        "stats_backend": backend,
        "n_boot": args.n_boot,
        "seed": args.seed,
        "adjudication": _adjudicate(ci_lo, ci_hi),
        "provenance": provenance({k: str(v) for k, v in vars(args).items()}),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(
        f"[compute_match/{args.metric}] {args.method} vs {args.baseline_kind}: "
        f"diff={diff_val:+.4f} CI=[{ci_lo:+.4f},{ci_hi:+.4f}] "
        f"p={pval:.4f} -> {out['adjudication']}"
    )
    return 0


def _adjudicate(ci_low: float, ci_high: float) -> str:
    """CLAUDE.md adjudication rule applied to the CI."""
    if ci_low > 0:
        return "supported"
    if ci_high < 0:
        return "reversed"
    return "unsupported"


def holm_correct_batch(report_paths: Iterable[Path], out_path: Path) -> None:
    """Utility: re-apply Holm correction across a *batch* of compute_match outputs.
    Each cell already has its own singleton-Holm; this overrides with the multi-cell
    correction across the batch (matching paper-table semantics).
    """
    cells = []
    pvals = []
    for rp in report_paths:
        d = json.loads(Path(rp).read_text())
        cells.append(d)
        pvals.append(d["sign_flip_p"])
    adj = holm_correct(pvals)
    for cell, a in zip(cells, adj):
        cell["holm_adjusted_p_batch"] = a
        cell["batch_size"] = len(pvals)
    Path(out_path).write_text(json.dumps(cells, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
