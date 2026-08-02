#!/usr/bin/env python3
"""Is the frontier's accuracy-specificity a metric artifact? (Appendix "mechanism".)

Question: is the diversity->gain predictability asymmetry (accuracy gain
predictable, calibration gain not) real, and does it hold across
{accuracy, ECE, Brier, NLL}?

Method (methodology-consistent across all 4 metrics):
  - Take the exact (pool, method, baseline_kind) triples defined by the
    existing compute_match cells (so the row set matches the published
    accuracy/ECE frontier slices).
  - For each triple, read the ensemble result JSON and the baseline result
    JSON, pull the *scalar* per-method metric (accuracy_test and
    calibration.{ece_15bin_equalmass, brier, nll}) for the ensemble method
    and the baseline method.
  - gain convention: positive = ensemble better.
        accuracy_gain = ens_acc  - base_acc
        ece_gain      = base_ece  - ens_ece   (lower is better)
        brier_gain    = base_brier- ens_brier
        nll_gain      = base_nll  - ens_nll
  - Regress each gain on the SAME diversity feature set + LOPO-CV that
    analysis/frontier.py uses (reuse its functions verbatim).

Cross-check: the scalar accuracy/ECE R2 here should track the paired
cm-cell frontier R2 (accuracy n_rank +0.597, ece n_rank -0.061) closely;
if it does, the Brier/NLL numbers (which cannot be paired-bootstrapped
because full per-example prob vectors were not stored) are trustworthy as
a feasibility signal.

This is a FEASIBILITY signal, not the final paired-CI result. It exists to
GO/KILL Plan B cheaply (no cluster, no GPU).

Usage: python3 analysis/w2_calibration_asymmetry.py
"""
from __future__ import annotations
import json, glob, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "analysis"))
from frontier import load_diversity, load_cm_cells, build_design_matrix, lopo_cv_r2  # noqa: E402


def norm_path(p: str) -> Path:
    """Resolve a result_path stored absolute or repo-relative, without
    hardcoding any machine-specific prefix. Released JSONs use repo-relative
    paths; anything else falls back to basename lookup."""
    s = str(p)
    if s.startswith("./"):
        s = s[2:]
    cand = ROOT / s
    if cand.exists():
        return cand
    # fallback: match by basename under ensemble_results/
    base = Path(s).name
    hits = list((ROOT / "ensemble_results").glob(base))
    return hits[0] if hits else cand


def method_metrics(result_json: dict, method: str) -> dict | None:
    m = result_json.get("methods", {}).get(method)
    if m is None:
        return None
    cal = m.get("calibration", {}) or {}
    acc = m.get("accuracy_test")
    if acc is None:
        return None
    return {
        "accuracy": acc,
        "ece": cal.get("ece_15bin_equalmass", cal.get("ece_15bin")),
        "mce": cal.get("mce"),
        "brier": cal.get("brier"),
        "nll": cal.get("nll"),
    }


def main():
    diversity = load_diversity(ROOT / "analysis")
    cm = load_cm_cells(ROOT / "analysis")
    # Deduplicate triples (accuracy and ECE cm cells describe the same triple)
    seen = {}
    for c in cm:
        if c["pool_id"].startswith("baseline_"):
            continue
        key = (c["pool_id"], c["method"], c["baseline_kind"])
        # read the cell file to get ensemble/baseline result paths + methods
        try:
            cell = json.loads(Path(c["path"]).read_text())
        except Exception:
            continue
        ens = cell.get("ensemble", {})
        base = cell.get("baseline", {})
        ens_rp = ens.get("result_path"); base_rp = base.get("result_path")
        ens_m = ens.get("method"); base_m = base.get("method")
        if not (ens_rp and base_rp and ens_m and base_m):
            continue
        seen[key] = (ens_rp, ens_m, base_rp, base_m)

    # Build metric gains per triple
    metrics = ("accuracy", "ece", "mce", "brier", "nll")
    rows_by = {m: [] for m in metrics}
    cache = {}
    def load(p):
        p = str(p)
        if p not in cache:
            fp = norm_path(p)
            try:
                cache[p] = json.loads(fp.read_text())
            except Exception:
                cache[p] = None
        return cache[p]

    n_triples = 0
    for (pool, method, kind), (ens_rp, ens_m, base_rp, base_m) in seen.items():
        ejson = load(ens_rp); bjson = load(base_rp)
        if ejson is None or bjson is None:
            continue
        em = method_metrics(ejson, ens_m); bm = method_metrics(bjson, base_m)
        if em is None or bm is None:
            continue
        if pool not in diversity:
            continue
        n_triples += 1
        for metric in metrics:
            ev, bv = em[metric], bm[metric]
            if ev is None or bv is None:
                continue
            if metric == "accuracy":
                gain = ev - bv          # higher better
            else:
                gain = bv - ev          # lower better -> ensemble gain = base - ens
            rows_by[metric].append({
                "pool_id": pool, "method": method, "baseline_kind": kind,
                "diff": float(gain),
            })

    print(f"[W2] usable triples: {n_triples}  (pools w/ diversity: {len(diversity)})")
    print(f"\n{'metric':10s} {'baseline':10s} {'n_rows':>6s} {'n_pools':>7s} {'R2_LOPO':>9s} {'mean_gain':>10s}")
    print("-" * 60)

    results = {}
    for metric in metrics:
        for kind in ("best_of_n", "n_rank"):
            rows = [r for r in rows_by[metric] if r["baseline_kind"] == kind]
            if len(rows) < 5:
                print(f"{metric:10s} {kind:10s} {len(rows):6d}   (too few)")
                continue
            X, y, feat_names, used = build_design_matrix(rows, diversity)
            pools = [r["pool_id"] for r in used]
            if len(used) < 5:
                print(f"{metric:10s} {kind:10s} {len(used):6d}   (too few usable)")
                continue
            lopo = lopo_cv_r2(X, y, pools, ridge=0.1)
            r2 = lopo.get("r2_lopo", float("nan"))
            results[(metric, kind)] = {
                "r2_lopo": r2, "n_rows": len(used),
                "n_pools": len(set(pools)), "mean_gain": float(np.mean(y)),
            }
            print(f"{metric:10s} {kind:10s} {len(used):6d} {len(set(pools)):7d} "
                  f"{r2:+9.3f} {np.mean(y):+10.4f}")

    # Asymmetry verdict
    print("\n=== ASYMMETRY VERDICT (n_rank slice) ===")
    acc = results.get(("accuracy", "n_rank"), {}).get("r2_lopo")
    print("accuracy R2 :", f"{acc:+.3f}" if acc is not None else "n/a")
    for m in ("ece", "mce", "brier", "nll"):
        v = results.get((m, "n_rank"), {}).get("r2_lopo")
        print(f"{m:8s} R2 :", f"{v:+.3f}" if v is not None else "n/a")
    if acc is not None:
        # "pure calibration-error" bucket = ECE + MCE (no refinement component)
        cal_r2 = [results.get((m, "n_rank"), {}).get("r2_lopo") for m in ("ece","mce")]
        cal_r2 = [x for x in cal_r2 if x is not None]
        if cal_r2:
            worst = max(cal_r2)  # most favorable calibration R2
            gap = acc - worst
            print(f"\naccuracy R2 - best calibration R2 = {gap:+.3f}")
            if acc >= 0.45 and worst <= 0.25:
                print("VERDICT: ASYMMETRY HOLDS -> C1 supported (GO)")
            elif acc >= 0.45 and worst >= 0.45:
                print("VERDICT: NO asymmetry (calibration also predictable) -> naive-B risk, re-examine C1/C3 boundary")
            else:
                print("VERDICT: PARTIAL / ambiguous -> inspect per-metric, widen pools")

    out = ROOT / "analysis/w2_calibration_asymmetry.json"
    out.write_text(json.dumps(
        {f"{m}__{k}": v for (m, k), v in results.items()}, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
