#!/usr/bin/env python3
"""H3 mechanism test (paper/prereg_calibration_predictability.md, §2 H3).

Claim to test: the diversity->gain predictability spectrum (accuracy/NLL/MCE
predictable; ECE/Brier not) is explained by the Murphy decomposition of a
proper score into RELIABILITY (calibration) and RESOLUTION (refinement):

    Brier_conf = Reliability - Resolution + Uncertainty
      Reliability = sum_k (n_k/N) (conf_k - acc_k)^2     # calibration; lower better
      Resolution  = sum_k (n_k/N) (acc_k - acc_bar)^2    # refinement;  higher better
      Uncertainty = acc_bar (1 - acc_bar)                # base rate; per-eval const

computed in the confidence-vs-correctness space (same space ECE lives in).

H3 predicts, on the n_rank compute-matched slice, regressing each component's
ENSEMBLE GAIN on pool diversity via LOPO-CV:
    RESOLUTION gain  R^2 ~ HIGH  (tracks accuracy R^2 = +0.60)
    RELIABILITY gain R^2 ~ LOW   (tracks ECE R^2 = -0.06)

i.e. the predictable part of the proper score is exactly the refinement part;
the calibration part's gain is diversity-unpredictable. If so, the whole
spectrum is explained and NLL's predictability (refinement-heavy) is expected,
not anomalous.

Non-tautology note: resolution correlates with accuracy but is NOT accuracy
(two models at equal accuracy can differ in resolution). The load-bearing
result is that RELIABILITY gain is unpredictable while the rest is not --
localizing the unpredictability to the calibration component.

Uses only stored per-example confidence/correctness; no cluster, no GPU.
Usage: python3 analysis/h3_mechanism.py
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "analysis"))
from frontier import load_diversity, load_cm_cells, build_design_matrix, lopo_cv_r2  # noqa: E402


def norm_path(p: str) -> Path:
    # Resolve a result_path stored absolute or repo-relative, without
    # hardcoding any machine-specific prefix; falls back to basename lookup.
    s = str(p)
    if s.startswith("./"):
        s = s[2:]
    cand = ROOT / s
    if cand.exists():
        return cand
    hits = list((ROOT / "ensemble_results").glob(Path(s).name))
    return hits[0] if hits else cand


def murphy(conf: np.ndarray, correct: np.ndarray, n_bins: int = 15) -> dict:
    """Murphy decomposition of the confidence-Brier via equal-mass bins."""
    n = len(conf)
    if n == 0:
        return {}
    acc_bar = float(correct.mean())
    order = np.argsort(conf)
    bins = np.array_split(order, min(n_bins, n))
    rel = res = 0.0
    for b in bins:
        if len(b) == 0:
            continue
        w = len(b) / n
        ck = float(conf[b].mean())
        ak = float(correct[b].mean())
        rel += w * (ck - ak) ** 2
        res += w * (ak - acc_bar) ** 2
    return {"reliability": rel, "resolution": res,
            "uncertainty": acc_bar * (1 - acc_bar), "acc_bar": acc_bar}


def get_method(result_json: dict, method: str):
    m = result_json.get("methods", {}).get(method)
    if m is None:
        return None
    preds = m.get("predictions_test")
    conf = m.get("confidence_test")
    labels = result_json.get("labels_test")
    if preds is None or conf is None or labels is None:
        return None
    preds = np.asarray(preds); conf = np.asarray(conf, dtype=float)
    labels = np.asarray(labels)
    if not (len(preds) == len(conf) == len(labels)):
        return None
    correct = (preds == labels).astype(float)
    return conf, correct


def main():
    diversity = load_diversity(ROOT / "analysis")
    cm = load_cm_cells(ROOT / "analysis")
    cache = {}
    def load(p):
        p = str(p)
        if p not in cache:
            try: cache[p] = json.loads(norm_path(p).read_text())
            except Exception: cache[p] = None
        return cache[p]

    # Collect n_rank triples with result paths
    triples = {}
    for c in cm:
        if c["pool_id"].startswith("baseline_") or c["baseline_kind"] != "n_rank":
            continue
        try: cell = json.loads(Path(c["path"]).read_text())
        except Exception: continue
        e, b = cell.get("ensemble", {}), cell.get("baseline", {})
        if all([e.get("result_path"), e.get("method"), b.get("result_path"), b.get("method")]):
            triples[(c["pool_id"], c["method"])] = (
                e["result_path"], e["method"], b["result_path"], b["method"])

    rows_res, rows_rel, rows_acc = [], [], []
    n_ok = 0
    for (pool, method), (erp, em, brp, bm) in triples.items():
        if pool not in diversity:
            continue
        ej, bj = load(erp), load(brp)
        if ej is None or bj is None:
            continue
        e = get_method(ej, em); b = get_method(bj, bm)
        if e is None or b is None:
            continue
        e_conf, e_cor = e; b_conf, b_cor = b
        me = murphy(e_conf, e_cor); mb = murphy(b_conf, b_cor)
        if not me or not mb:
            continue
        n_ok += 1
        # ensemble-favorable gains
        res_gain = me["resolution"] - mb["resolution"]      # higher resolution better
        rel_gain = mb["reliability"] - me["reliability"]     # lower reliability better
        acc_gain = me["acc_bar"] - mb["acc_bar"]
        common = {"pool_id": pool, "method": method, "baseline_kind": "n_rank"}
        rows_res.append({**common, "diff": float(res_gain)})
        rows_rel.append({**common, "diff": float(rel_gain)})
        rows_acc.append({**common, "diff": float(acc_gain)})

    def r2(rows):
        X, y, fn, used = build_design_matrix(rows, diversity)
        if len(used) < 5:
            return None, 0, None
        p = [r["pool_id"] for r in used]
        return lopo_cv_r2(X, y, p, ridge=0.1).get("r2_lopo"), len(used), float(np.mean(y))

    r_res, n_res, m_res = r2(rows_res)
    r_rel, n_rel, m_rel = r2(rows_rel)
    r_acc, n_acc, m_acc = r2(rows_acc)

    # correlation resolution_gain vs accuracy_gain (non-tautology diagnostic)
    key = lambda r: (r["pool_id"], r["method"])
    ad = {key(r): r["diff"] for r in rows_acc}
    rd = {key(r): r["diff"] for r in rows_res}
    common_keys = [k for k in rd if k in ad]
    rr = np.array([rd[k] for k in common_keys]); aa = np.array([ad[k] for k in common_keys])
    corr_res_acc = float(np.corrcoef(rr, aa)[0, 1]) if len(rr) > 3 else float("nan")

    # resolution-gain spread relative to accuracy-gain spread (paper: sigma=0.2x)
    res_all = np.array([r["diff"] for r in rows_res])
    acc_all = np.array([r["diff"] for r in rows_acc])
    res_std_over_acc_std = (float(res_all.std() / acc_all.std())
                            if acc_all.std() > 0 else float("nan"))

    print(f"[H3] usable (pool,method) cells: {n_ok}")
    print(f"\n{'component':14s} {'n':>4s} {'R2_LOPO':>9s} {'mean_gain':>10s}")
    print("-" * 42)
    print(f"{'RESOLUTION':14s} {n_res:4d} {r_res:+9.3f} {m_res:+10.4f}   (refinement; predict HIGH)")
    print(f"{'RELIABILITY':14s} {n_rel:4d} {r_rel:+9.3f} {m_rel:+10.4f}   (calibration; predict LOW)")
    print(f"{'accuracy(chk)':14s} {n_acc:4d} {r_acc:+9.3f} {m_acc:+10.4f}   (should be ~+0.60)")
    print(f"\ncorr(resolution_gain, accuracy_gain) = {corr_res_acc:+.3f}  "
          f"(<0.9 => resolution is not just accuracy)")

    # Verdict
    print("\n=== H3 VERDICT ===")
    ok = (r_res is not None and r_rel is not None and
          r_res >= 0.35 and r_rel <= 0.20)
    if ok:
        print(f"H3 SUPPORTED: predictable part = refinement (res R2={r_res:+.3f}); "
              f"unpredictable part = calibration (rel R2={r_rel:+.3f}).")
        print("=> the spectrum is mechanistically explained; NLL predictability expected.")
    else:
        print(f"H3 NOT cleanly supported: res R2={r_res}, rel R2={r_rel}.")
        print("=> mechanism weak; per prereg §5, risk of downgrade to honest-negative/D&B.")

    out = ROOT / "analysis/h3_mechanism.json"
    out.write_text(json.dumps({
        "n_cells": n_ok,
        "resolution_gain": {"r2_lopo": r_res, "n": n_res, "mean": m_res},
        "reliability_gain": {"r2_lopo": r_rel, "n": n_rel, "mean": m_rel},
        "accuracy_gain_check": {"r2_lopo": r_acc, "n": n_acc, "mean": m_acc},
        "corr_resolution_accuracy": corr_res_acc,
        "res_std_over_acc_std": res_std_over_acc_std,
        "verdict_supported": bool(ok),
    }, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
