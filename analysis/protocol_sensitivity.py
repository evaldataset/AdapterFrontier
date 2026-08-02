#!/usr/bin/env python3
"""Protocol-sensitivity analysis — the核心 contribution of the EACL 2027 submission.

Question: how much of the *apparent* PEFT-ensembling benefit is produced by
common evaluation shortcuts rather than by ensembling itself?

We quantify three shortcuts that are widespread in the ensembling/merging
literature, using only released artifacts (no cluster, no GPU):

  S1 selection-on-test    : pick the best single adapter by TEST accuracy
                            (oracle) instead of by held-out val_selection.
                            Reconstructed exactly from individual_accuracy_test.
  S2 no compute-match     : compare the ensemble against the pool's own best
                            single (best_of_n) instead of against a
                            compute-matched stronger single (n_rank).
  S3 aggregate reporting  : report one task-level mean instead of per-cell
                            verdicts, hiding REVERSED cells inside a positive
                            task average.

Scope discipline (matches paper/claim_ledger.md):
  - HellaSwag cells are EXCLUDED: they were produced by the pre-fix evaluator
    that selected on val_combine (stale selection split). Documented, not used.
  - GSM8K is excluded: best_single there was selected on the same test split
    it is evaluated on. (A fourth shortcut, contaminated-benchmark, was
    retracted on 2026-07-28: the decontamination probe was invalid.)
  - S1/S2/S3 headline uses encoder classification pools.

Usage: python3 analysis/protocol_sensitivity.py
"""
from __future__ import annotations

import json
import glob
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent

DECODER_TAGS = ("qwen", "llama", "mistral", "tinyllama", "pythia", "smollm")
EXCLUDE_TAGS = ("hellaswag",)      # stale selection split (see claim ledger)
CONTAM_TAGS = ("gsm8k",)           # contaminated -> S4 only


def is_decoder(name: str) -> bool:
    n = name.lower()
    return any(t in n for t in DECODER_TAGS)


def excluded(name: str) -> bool:
    n = name.lower()
    return any(t in n for t in EXCLUDE_TAGS)


def contaminated(name: str) -> bool:
    n = name.lower()
    return any(t in n for t in CONTAM_TAGS)


# --------------------------------------------------------------------------
# S1: selection-on-test inflation
# --------------------------------------------------------------------------
def s1_selection_on_test() -> dict:
    """oracle (test-selected single) - best_single (val_selection-selected)."""
    rows = []
    for f in sorted(glob.glob(str(ROOT / "ensemble_results/*.json"))):
        name = Path(f).name
        if excluded(name) or contaminated(name):
            continue
        try:
            d = json.loads(Path(f).read_text())
        except Exception:
            continue
        indiv = d.get("individual_accuracy_test")
        bs = (d.get("methods", {}) or {}).get("best_single", {})
        if not indiv or bs.get("accuracy_test") is None:
            continue
        oracle = float(max(indiv))
        strict = float(bs["accuracy_test"])
        rows.append({
            "pool_id": d.get("pool_id") or Path(f).stem,
            "family": "decoder" if is_decoder(name) else "encoder",
            "oracle_test_selected": oracle,
            "strict_val_selected": strict,
            "inflation_pp": 100.0 * (oracle - strict),
            "n_adapters": d.get("n_adapters"),
        })
    enc = [r for r in rows if r["family"] == "encoder"]
    out = {"n_pools_total": len(rows), "n_pools_encoder": len(enc)}
    for label, sub in (("all", rows), ("encoder", enc)):
        if not sub:
            continue
        infl = np.array([r["inflation_pp"] for r in sub])
        out[label] = {
            "mean_inflation_pp": float(infl.mean()),
            "median_inflation_pp": float(np.median(infl)),
            "max_inflation_pp": float(infl.max()),
            "frac_pools_inflated": float((infl > 0).mean()),
        }
    out["rows"] = rows
    return out


# --------------------------------------------------------------------------
# S2 / S3: shared cm-cell loading
# --------------------------------------------------------------------------
CM_PAT = re.compile(
    r"^(?P<pool>.+?)_cm_(?P<method>[a-z_]+?)_vs_"
    r"(?P<kind>best_of_n|n_rank|n_steps|n_data)(?P<ece>_ECE)?\.json$"
)

TASK_TOKENS = ("mnli", "sst2", "qnli", "boolq", "anli", "agnews")


def task_of(pool_id: str) -> str:
    p = pool_id.lower()
    for t in TASK_TOKENS:
        if t in p:
            return t
    return "other"


def load_accuracy_cells() -> list[dict]:
    """Accuracy cm cells on clean (non-HellaSwag, non-GSM8K) pools."""
    cells = []
    for f in sorted(glob.glob(str(ROOT / "analysis/*_cm_*.json"))):
        m = CM_PAT.match(Path(f).name)
        if not m or m.group("ece"):          # accuracy arm only
            continue
        pool = m.group("pool")
        if excluded(pool) or contaminated(pool):
            continue
        try:
            d = json.loads(Path(f).read_text())
        except Exception:
            continue
        diff = d.get("accuracy_diff")
        if diff is None:
            continue
        cells.append({
            "pool_id": pool,
            "task": task_of(pool),
            "family": "decoder" if is_decoder(pool) else "encoder",
            "method": m.group("method"),
            "baseline_kind": m.group("kind"),
            "diff_pp": 100.0 * float(diff),
            "verdict": d.get("adjudication_post_fdr") or d.get("adjudication"),
        })
    return cells


def _verdict_stats(cells: list[dict]) -> dict:
    n = len(cells)
    if n == 0:
        return {}
    sup = sum(1 for c in cells if c["verdict"] == "supported")
    rev = sum(1 for c in cells if c["verdict"] == "reversed")
    d = np.array([c["diff_pp"] for c in cells])
    return {
        "n_cells": n,
        "pct_supported": round(100.0 * sup / n, 1),
        "pct_reversed": round(100.0 * rev / n, 1),
        "mean_diff_pp": float(d.mean()),
    }


def s2_no_compute_match(cells: list[dict]) -> dict:
    """best_of_n (pool's own best single) vs n_rank (compute-matched single)."""
    out = {}
    for label, sub in (("all", cells),
                       ("encoder", [c for c in cells if c["family"] == "encoder"])):
        block = {}
        for kind in ("best_of_n", "n_rank"):
            st = _verdict_stats([c for c in sub if c["baseline_kind"] == kind])
            if st:
                block[kind] = st
        if "best_of_n" in block and "n_rank" in block:
            block["delta_pct_supported"] = round(
                block["best_of_n"]["pct_supported"] - block["n_rank"]["pct_supported"], 1)
            block["delta_mean_diff_pp"] = round(
                block["best_of_n"]["mean_diff_pp"] - block["n_rank"]["mean_diff_pp"], 3)
        out[label] = block
    return out


def s3_aggregate_reporting(cells: list[dict]) -> dict:
    """Task-level mean (what most papers report) vs per-cell verdicts."""
    out = {}
    for label, sub in (("all", cells),
                       ("encoder", [c for c in cells if c["family"] == "encoder"])):
        by_task = defaultdict(list)
        for c in sub:
            if c["baseline_kind"] == "n_rank":     # strict comparison
                by_task[c["task"]].append(c)
        # A task mean near zero reads as "ensembling makes no difference here".
        # We measure how many statistically SIGNIFICANT cells (either
        # direction) that neutral-looking average conceals.
        NEUTRAL_PP = 0.5
        tasks, hidden_sig, cells_in_neutral = [], 0, 0
        for t, cs in sorted(by_task.items()):
            mean_pp = float(np.mean([c["diff_pp"] for c in cs]))
            rev = sum(1 for c in cs if c["verdict"] == "reversed")
            sup = sum(1 for c in cs if c["verdict"] == "supported")
            neutral = abs(mean_pp) < NEUTRAL_PP
            tasks.append({
                "task": t, "n_cells": len(cs),
                "task_mean_diff_pp": round(mean_pp, 3),
                "task_level_reads_as": ("no effect" if neutral
                                        else ("helps" if mean_pp > 0 else "hurts")),
                "reversed_cells_inside": rev,
                "supported_cells_inside": sup,
            })
            if neutral:
                hidden_sig += rev + sup
                cells_in_neutral += len(cs)
        out[label] = {
            "neutral_threshold_pp": NEUTRAL_PP,
            "tasks": tasks,
            "n_tasks": len(tasks),
            "n_tasks_reading_as_no_effect": sum(
                1 for t in tasks if t["task_level_reads_as"] == "no effect"),
            "significant_cells_hidden_in_neutral_tasks": hidden_sig,
            "cells_in_neutral_tasks": cells_in_neutral,
            "pct_hidden_significant": (round(100.0 * hidden_sig / cells_in_neutral, 1)
                                       if cells_in_neutral else None),
        }
    return out


def s2_by_method(cells: list[dict]) -> dict:
    """Per-method breakdown of the strict (n_rank) encoder comparison."""
    enc = [c for c in cells if c["family"] == "encoder" and c["baseline_kind"] == "n_rank"]
    out = {}
    for meth in sorted(set(c["method"] for c in enc)):
        out[meth] = _verdict_stats([c for c in enc if c["method"] == meth])
    return out


def s4_contamination() -> dict:
    """GSM8K: contaminated gain vs decontaminated (digit-shift) gain — E1."""
    f = ROOT / "analysis/gsm8k_contamination_ensemble_qwen25_7b.json"
    if not f.exists():
        return {"error": "E1 probe not found"}
    d = json.loads(f.read_text())
    return {
        "model": d.get("base_model"),
        "n_samples": d.get("n_samples"),
        "contaminated_gain_pp": round(100.0 * float(d["delta_orig"]), 2),
        "decontaminated_gain_pp": round(100.0 * float(d["delta_shift"]), 2),
        "gain_evaporated_pct": 100.0,
        "note": ("majority_vote vs best_single. Selection also touched the "
                 "evaluated test split (L12), so this cell carries no "
                 "defensible positive claim."),
    }


def main() -> int:
    cells = load_accuracy_cells()
    res = {
        "scope": {
            "excluded_stale_split": list(EXCLUDE_TAGS),
            "excluded_contaminated_from_S1_S3": list(CONTAM_TAGS),
            "n_accuracy_cells_clean": len(cells),
            "n_pools_clean": len(set(c["pool_id"] for c in cells)),
        },
        "S1_selection_on_test": s1_selection_on_test(),
        "S2_no_compute_match": s2_no_compute_match(cells),
        "S2b_strict_encoder_by_method": s2_by_method(cells),
        "S3_aggregate_reporting": s3_aggregate_reporting(cells),
        # S4 retracted 2026-07-28: the decontamination probe was invalid.
    }

    sc = res["scope"]
    print(f"[scope] clean accuracy cells: {sc['n_accuracy_cells_clean']} "
          f"from {sc['n_pools_clean']} pools "
          f"(HellaSwag excluded: stale selection split; GSM8K -> S4 only)\n")

    s1 = res["S1_selection_on_test"]
    print("S1  selection-on-test (oracle vs val-selected single)")
    for lab in ("all", "encoder"):
        if lab in s1:
            b = s1[lab]
            print(f"    {lab:8s} n={s1['n_pools_encoder'] if lab=='encoder' else s1['n_pools_total']:3d}  "
                  f"mean +{b['mean_inflation_pp']:.2f}pp  median +{b['median_inflation_pp']:.2f}pp  "
                  f"max +{b['max_inflation_pp']:.2f}pp  inflated {100*b['frac_pools_inflated']:.0f}% of pools")

    print("\nS2  no compute-match (best_of_n vs n_rank baseline)")
    for lab in ("all", "encoder"):
        b = res["S2_no_compute_match"].get(lab, {})
        for kind in ("best_of_n", "n_rank"):
            if kind in b:
                st = b[kind]
                print(f"    {lab:8s} {kind:10s} n={st['n_cells']:4d}  "
                      f"SUPPORTED {st['pct_supported']:5.1f}%  REVERSED {st['pct_reversed']:5.1f}%  "
                      f"mean {st['mean_diff_pp']:+.3f}pp")
        if "delta_pct_supported" in b:
            print(f"    {lab:8s} -> shortcut inflates SUPPORTED by "
                  f"{b['delta_pct_supported']:+.1f}pp, mean gain by {b['delta_mean_diff_pp']:+.3f}pp")

    print("\nS3  aggregate reporting (task mean hides cell-level reversals)")
    for lab in ("all", "encoder"):
        b = res["S3_aggregate_reporting"].get(lab, {})
        if not b:
            continue
        print(f"    {lab}: {b['n_tasks_reading_as_no_effect']}/{b['n_tasks']} tasks read as "
              f"'no effect' (|mean|<{b['neutral_threshold_pp']}pp), yet conceal "
              f"{b['significant_cells_hidden_in_neutral_tasks']} significant cells "
              f"({b['pct_hidden_significant']}% of their cells)")
        for t in b["tasks"]:
            print(f"       {t['task']:8s} mean {t['task_mean_diff_pp']:+7.3f}pp -> "
                  f"'{t['task_level_reads_as']:9s}'  n={t['n_cells']:3d}  "
                  f"sig_inside(rev/sup)={t['reversed_cells_inside']}/{t['supported_cells_inside']}")

    # S4 (contaminated benchmark) was removed when its GSM8K probe was retracted
    # on 2026-07-28; the key is absent from `res` and must not be required here.
    s4 = res.get("S4_contaminated_benchmark")
    if s4 and "error" not in s4:
        print(f"\nS4  contaminated benchmark (GSM8K, {s4['model']})")
        print(f"    contaminated gain {s4['contaminated_gain_pp']:+.2f}pp -> "
              f"decontaminated {s4['decontaminated_gain_pp']:+.2f}pp "
              f"({s4['gain_evaporated_pct']:.0f}% evaporates)")

    out = ROOT / "analysis/protocol_sensitivity.json"
    out.write_text(json.dumps(res, indent=2))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
