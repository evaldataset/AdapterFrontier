#!/usr/bin/env python3
"""Summarize paired Project A sweep results from two sweep_runner CSV files."""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Summarize paired Project A sweep CSV outputs"
    )
    p.add_argument("--baseline-csv", required=True)
    p.add_argument("--candidate-csv", required=True)
    p.add_argument("--metric", required=True)
    p.add_argument("--baseline-name", required=True)
    p.add_argument("--candidate-name", required=True)
    p.add_argument("--out-json", default=None)
    p.add_argument("--out-md", default=None)
    p.add_argument("--bootstrap-samples", type=int, default=5000)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    baseline_rows, baseline_failures = _load_rows(args.baseline_csv, args.metric)
    candidate_rows, candidate_failures = _load_rows(args.candidate_csv, args.metric)

    shared_seeds = sorted(set(baseline_rows) & set(candidate_rows))
    if not shared_seeds:
        raise SystemExit("No shared successful seeds found between the two CSV files")

    deltas = [
        candidate_rows[s]["metric"] - baseline_rows[s]["metric"] for s in shared_seeds
    ]
    mean_delta = sum(deltas) / len(deltas)
    ci_low, ci_high = _bootstrap_ci(deltas, args.bootstrap_samples, args.seed)
    sign_flip_p = _sign_flip_pvalue(deltas)
    wins = sum(1 for d in deltas if d > 0)
    losses = sum(1 for d in deltas if d < 0)
    ties = sum(1 for d in deltas if d == 0)

    summary = {
        "baseline_name": args.baseline_name,
        "candidate_name": args.candidate_name,
        "metric": args.metric,
        "shared_seeds": shared_seeds,
        "baseline_mean": sum(v["metric"] for v in baseline_rows.values())
        / len(baseline_rows),
        "candidate_mean_shared": sum(candidate_rows[s]["metric"] for s in shared_seeds)
        / len(shared_seeds),
        "mean_delta": mean_delta,
        "bootstrap_ci": {"low": ci_low, "high": ci_high},
        "sign_flip_p": sign_flip_p,
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "baseline_failures": baseline_failures,
        "candidate_failures": candidate_failures,
        "per_seed": [
            {
                "seed": seed,
                "baseline": baseline_rows[seed]["metric"],
                "candidate": candidate_rows[seed]["metric"],
                "delta": candidate_rows[seed]["metric"] - baseline_rows[seed]["metric"],
            }
            for seed in shared_seeds
        ],
    }

    if args.out_json:
        Path(args.out_json).write_text(json.dumps(summary, indent=2))
    if args.out_md:
        Path(args.out_md).write_text(_render_markdown(summary))

    print(json.dumps(summary, indent=2))
    return 0


def _load_rows(
    path_str: str, metric_key: str
) -> tuple[dict[int, dict[str, float]], list[dict[str, str]]]:
    rows: dict[int, dict[str, float]] = {}
    failures: list[dict[str, str]] = []
    with Path(path_str).open() as f:
        for row in csv.DictReader(f):
            seed = int(row["seed"])
            metric_value = row.get(metric_key, "")
            if row.get("status") != "OK" or not metric_value:
                failures.append(
                    {
                        "seed": str(seed),
                        "host_id": row.get("host_id", ""),
                        "status": row.get("status", ""),
                        "note": row.get("note", ""),
                    }
                )
                continue
            rows[seed] = {"metric": float(metric_value)}
    return rows, failures


def _bootstrap_ci(deltas: list[float], samples: int, seed: int) -> tuple[float, float]:
    """Bias-corrected (BC) bootstrap 95% CI."""
    import math
    rng = random.Random(seed)
    n = len(deltas)
    observed_mean = sum(deltas) / n
    means = []
    for _ in range(samples):
        draw = [rng.choice(deltas) for _ in deltas]
        means.append(sum(draw) / len(draw))
    means.sort()

    # Bias correction: z0 = Phi^{-1}(proportion of bootstrap means < observed)
    below = sum(1 for m in means if m < observed_mean) / len(means)
    below = max(0.001, min(0.999, below))  # clamp to avoid ±inf

    # Approximate inverse normal CDF (Beasley-Springer-Moro)
    def _norm_inv(p: float) -> float:
        # Rational approximation for Phi^{-1}
        if p <= 0:
            return -6.0
        if p >= 1:
            return 6.0
        t = math.sqrt(-2.0 * math.log(min(p, 1 - p)))
        c0, c1, c2 = 2.515517, 0.802853, 0.010328
        d1, d2, d3 = 1.432788, 0.189269, 0.001308
        result = t - (c0 + c1 * t + c2 * t * t) / (1 + d1 * t + d2 * t * t + d3 * t * t * t)
        return result if p > 0.5 else -result

    z0 = _norm_inv(below)
    z_alpha_lo = _norm_inv(0.025)
    z_alpha_hi = _norm_inv(0.975)

    # BC-adjusted percentiles
    p_lo = below  # fallback: use percentile if z0 ≈ 0
    p_hi = below
    # p_lo = Phi(2*z0 + z_alpha_lo), p_hi = Phi(2*z0 + z_alpha_hi)
    def _norm_cdf(x: float) -> float:
        return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

    p_lo = _norm_cdf(2 * z0 + z_alpha_lo)
    p_hi = _norm_cdf(2 * z0 + z_alpha_hi)

    idx_lo = max(0, min(len(means) - 1, int(p_lo * len(means))))
    idx_hi = max(0, min(len(means) - 1, int(p_hi * len(means))))
    return means[idx_lo], means[idx_hi]


def _sign_flip_pvalue(deltas: list[float], one_sided: bool = False) -> float:
    """Sign-flip permutation p-value.

    If one_sided=True, tests H0: mean(deltas) <= 0 (i.e., one-sided for
    pre-specified positive direction). Use for confirmatory cells where the
    original paper claims method > LoRA.
    """
    n = len(deltas)
    observed = sum(deltas) / n
    if one_sided:
        # One-sided: count how often permuted mean >= observed
        obs_stat = observed
    else:
        obs_stat = abs(observed)

    total = 1 << n
    if n > 20:
        rng = random.Random(42)
        mc_n = 100_000
        count = 0
        for _ in range(mc_n):
            perm_mean = sum(d * rng.choice([-1, 1]) for d in deltas) / n
            if one_sided:
                if perm_mean >= obs_stat:
                    count += 1
            else:
                if abs(perm_mean) >= obs_stat:
                    count += 1
        return count / mc_n
    count = 0
    for mask in range(total):
        vals = []
        for idx, delta in enumerate(deltas):
            vals.append(delta if (mask >> idx) & 1 else -delta)
        perm_mean = sum(vals) / len(vals)
        if one_sided:
            if perm_mean >= obs_stat:
                count += 1
        else:
            if abs(perm_mean) >= obs_stat:
                count += 1
    return count / total


def _render_markdown(summary: dict[str, Any]) -> str:
    baseline_name = str(summary["baseline_name"])
    candidate_name = str(summary["candidate_name"])
    metric = str(summary["metric"])
    shared_seed_values = list(summary["shared_seeds"])
    shared_seeds = ", ".join(str(seed) for seed in shared_seed_values)
    ci = dict(summary["bootstrap_ci"])
    per_seed = list(summary["per_seed"])

    lines = [
        f"# {candidate_name} vs {baseline_name}",
        "",
        f"- metric: `{metric}`",
        f"- shared seeds: `{shared_seeds}`",
        f"- mean delta (`{candidate_name} - {baseline_name}`): `{summary['mean_delta']:+.6f}`",
        f"- bootstrap CI: `[{ci['low']:+.6f}, {ci['high']:+.6f}]`",
        f"- sign-flip p-value: `{summary['sign_flip_p']}`",
        f"- wins / losses / ties: `{summary['wins']} / {summary['losses']} / {summary['ties']}`",
        "",
        "## Per-Seed",
        "",
        f"| seed | {baseline_name} | {candidate_name} | delta |",
        "|---|---:|---:|---:|",
    ]
    for row in per_seed:
        lines.append(
            f"| {row['seed']} | {row['baseline']:.6f} | {row['candidate']:.6f} | {row['delta']:+.6f} |"
        )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
