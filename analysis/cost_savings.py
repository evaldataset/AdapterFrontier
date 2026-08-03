#!/usr/bin/env python3
"""Table `tab:cost` had no generator and one row did not reconstruct.

The cost-savings table contrasts blind ensembling (always `soft_vote`) with a
per-cell oracle pick over the four combination rules, and prices both in
adapter forward-passes per query. No script produced it, no JSON held its
numbers, and `reproduce_paper_numbers.py` did not assert it -- so in a paper
whose claim is that every number re-derives from the release, this one did not.

Reconstructing it fixed the slice definition: groups are
(pool, baseline_kind, metric) over pools that are not `baseline_*`
self-comparisons. Under that definition three of the four published row counts
match exactly (Dec acc 50, Dec ECE 50, Enc ECE 63) and the fourth does not
(Enc acc: 64 here, 54 published). We regenerate the whole table rather than
keep a row we cannot reproduce.

Cost model, as the caption states: vote-based rules (`soft_vote`,
`logit_avg`, `majority_vote`) run one forward pass per pool member, so they
cost N; `greedy_soup` emits a single merged checkpoint and costs 1.

Usage: python3 analysis/cost_savings.py
"""
from __future__ import annotations

import glob
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DECODER = ("qwen", "llama", "mistral", "tinyllama", "pythia", "smollm")
VOTE_RULES = {"soft_vote", "logit_avg", "majority_vote"}
CELL = re.compile(r"^(?P<pool>.+?)_cm_(?P<method>[a-z_]+?)_vs_"
                  r"(?P<kind>best_of_n|n_rank|n_steps|n_data)(?P<ece>_ECE)?\.json$")


def pool_sizes() -> dict[str, int]:
    out = {}
    for f in glob.glob(str(ROOT / "pools/*.json")):
        if "pilot_c1" in f:
            continue
        d = json.loads(Path(f).read_text())
        pid = d.get("pool_id") or Path(f).stem
        out[pid] = d.get("n_adapters") or len(d.get("adapters") or [])
    return out


def load_groups() -> dict[tuple, dict]:
    groups: dict[tuple, dict] = defaultdict(dict)
    for f in sorted(glob.glob(str(ROOT / "analysis/*_cm_*.json"))):
        m = CELL.match(Path(f).name)
        if not m:
            continue
        pool = m.group("pool")
        if pool.startswith("baseline_"):          # self-comparison, not a pool
            continue
        d = json.loads(Path(f).read_text())
        metric = "ece" if m.group("ece") else "accuracy"
        diff = d.get("ece_diff") if metric == "ece" else d.get("accuracy_diff")
        if diff is None:
            continue
        fam = "decoder" if any(t in pool.lower() for t in DECODER) else "encoder"
        groups[(pool, m.group("kind"), metric, fam)][m.group("method")] = {
            "delta_pp": 100.0 * float(diff),
            "verdict": d.get("adjudication_post_fdr") or d.get("adjudication"),
        }
    return groups


def main() -> int:
    sizes = pool_sizes()
    groups = load_groups()
    slices: dict[str, list] = defaultdict(list)

    for (pool, _kind, metric, fam), rules in groups.items():
        if "soft_vote" not in rules:
            continue
        n = sizes.get(pool) or 1
        blind = rules["soft_vote"]
        best = max(rules.items(), key=lambda kv: kv[1]["delta_pp"])
        slices[f"{fam}_{metric}"].append({
            "pool": pool,
            "blind_delta": blind["delta_pp"],
            "blind_reversed": blind["verdict"] == "reversed",
            "blind_cost": n,
            "routed_rule": best[0],
            "routed_delta": best[1]["delta_pp"],
            "routed_reversed": best[1]["verdict"] == "reversed",
            "routed_cost": 1 if best[0] == "greedy_soup" else n,
        })

    order = [("encoder_accuracy", "Enc acc"), ("decoder_accuracy", "Dec acc"),
             ("encoder_ece", "Enc ECE"), ("decoder_ece", "Dec ECE")]
    out, rows = {}, []
    everything: list = []

    def summarise(items):
        return {
            "n": len(items),
            "blind_mean_delta_pp": float(np.mean([x["blind_delta"] for x in items])),
            "blind_pct_reversed": round(100 * np.mean([x["blind_reversed"] for x in items])),
            "blind_median_cost": int(np.median([x["blind_cost"] for x in items])),
            "routed_mean_delta_pp": float(np.mean([x["routed_delta"] for x in items])),
            "routed_pct_reversed": round(100 * np.mean([x["routed_reversed"] for x in items])),
            "routed_median_cost": int(np.median([x["routed_cost"] for x in items])),
        }

    hdr = f"{'slice':10s} {'n':>4s} | {'blind d':>8s} {'REV':>4s} {'cost':>5s} | {'routed d':>9s} {'REV':>4s} {'cost':>5s}"
    print(hdr); print("-" * len(hdr))
    for key, label in order:
        items = slices.get(key, [])
        if not items:
            continue
        everything += items
        s = summarise(items); out[label] = s; rows.append((label, s))
        print(f"{label:10s} {s['n']:4d} | {s['blind_mean_delta_pp']:+7.1f}pp {s['blind_pct_reversed']:3d}% "
              f"{s['blind_median_cost']:4d}x | {s['routed_mean_delta_pp']:+8.1f}pp "
              f"{s['routed_pct_reversed']:3d}% {s['routed_median_cost']:4d}x")
    s = summarise(everything); out["All"] = s
    print("-" * len(hdr))
    print(f"{'All':10s} {s['n']:4d} | {s['blind_mean_delta_pp']:+7.1f}pp {s['blind_pct_reversed']:3d}% "
          f"{s['blind_median_cost']:4d}x | {s['routed_mean_delta_pp']:+8.1f}pp "
          f"{s['routed_pct_reversed']:3d}% {s['routed_median_cost']:4d}x")

    picks = defaultdict(int)
    for x in everything:
        picks[x["routed_rule"]] += 1
    print("\noracle rule chosen:", dict(sorted(picks.items(), key=lambda kv: -kv[1])))

    out["_note"] = ("Groups are (pool, baseline_kind, metric) excluding baseline_* "
                    "self-comparison pools and requiring soft_vote. 'Routed' is a "
                    "per-group oracle over the available rules, in-sample to the "
                    "corpus the D1-D5 rules were derived from (L11). Cost is "
                    "adapter forward-passes per query: N for vote-based rules, 1 "
                    "for greedy_soup.")
    out["_oracle_rule_counts"] = dict(picks)
    p = ROOT / "analysis/cost_savings.json"
    p.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
