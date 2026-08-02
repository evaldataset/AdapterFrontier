#!/usr/bin/env python3
"""Does a single temperature reproduce the ensemble's calibration advantage?

The paper's calibration claim is measured against a single adapter that is
selected by validation *accuracy* and never post-hoc calibrated. Review's
central objection is that this is the weak-baseline failure the paper itself
diagnoses (shortcut S2), applied to its own positive result: if one scalar
fitted on held-out data closes the ECE gap, then populations are not what
buys reliability.

We can answer this properly -- not by approximation -- on the six pools whose
per-adapter logits cache was retained. Those pools carry schema v2, so they
have the pre-registered split indices, and the temperature is fitted where a
combination hyperparameter belongs: on `val_combine`, which is reserved for
exactly this and is otherwise unused. Nothing here touches `test` except the
final measurement.

Four arms per pool, all evaluated on `test`:
  A  ensemble (soft_vote), uncalibrated      -- what the paper reports
  B  best_single, uncalibrated               -- what the paper compares against
  C  best_single + temperature               -- the missing competitor
  D  ensemble + temperature                  -- the symmetric comparison

The comparison the title rests on is A vs B. The comparison that decides
whether the title is honest is A vs C.

Caveat carried into the paper: these six are `best_of_n` pools (the single is
the pool's own best member), not the headline `n_rank` slice, whose logits
cache was not retained. Direction, not magnitude, is what transfers.

Usage: python3 analysis/temperature_control.py
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "ensemble_cache"
N_BINS = 15


def softmax(z: np.ndarray) -> np.ndarray:
    e = np.exp(z - z.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


def ece_equal_mass(conf: np.ndarray, correct: np.ndarray, n_bins: int = N_BINS) -> float:
    """Matches analysis/compute_match._ece_equal_mass."""
    order = np.argsort(conf)
    total = 0.0
    for b in np.array_split(order, n_bins):
        if len(b) == 0:
            continue
        total += len(b) * abs(conf[b].mean() - correct[b].mean())
    return float(total / len(conf))


def nll(logits: np.ndarray, y: np.ndarray, T: float) -> float:
    p = softmax(logits / T)
    return float(-np.log(np.clip(p[np.arange(len(y)), y], 1e-12, None)).mean())


def fit_temperature(logits: np.ndarray, y: np.ndarray) -> float:
    r = minimize_scalar(lambda t: nll(logits, y, t), bounds=(0.05, 20.0),
                        method="bounded", options={"xatol": 1e-4})
    return float(r.x)


def arm(logits: np.ndarray, y: np.ndarray, T: float = 1.0) -> tuple[float, float, float]:
    """-> (ECE, accuracy, mean confidence) on the given logits at temperature T."""
    p = softmax(logits / T)
    pred = p.argmax(-1)
    conf = p.max(-1)
    corr = (pred == y).astype(np.float64)
    return ece_equal_mass(conf, corr), float(corr.mean()), float(conf.mean())


def labels_for_task(task: str, n_expected: int) -> np.ndarray | None:
    """Evaluation-set labels in dataset order, i.e. the order split_indices()
    shuffles. Returns None if the dataset is not available offline."""
    import os
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    import sys
    sys.path.insert(0, str(ROOT))
    from datasets import load_dataset

    from ensemble_eval import TASK_CONFIGS
    cfg = TASK_CONFIGS.get(task)
    if cfg is None:
        return None
    try:
        ds = load_dataset(cfg["dataset"], cfg["config"], split=cfg["split"])
    except Exception:
        return None
    y = np.asarray(ds[cfg["label"]], dtype=np.int64)
    return y if len(y) == n_expected else None


def main() -> int:
    rows = []
    for rf in sorted(glob.glob(str(ROOT / "ensemble_results/pilot_c1/*.json"))):
        d = json.loads(Path(rf).read_text())
        pid = d["pool_id"]
        npys = sorted(CACHE.glob(f"{pid}__*.npy"))
        if len(npys) < 2:
            continue
        L = np.stack([np.load(f) for f in npys], axis=0)      # (M, N, K)

        sel = np.asarray(d["val_selection_indices"])
        comb = np.asarray(d["val_combine_indices"])
        test = np.asarray(d["test_indices"])
        y_sel = np.asarray(d["labels_val_selection"])
        y_test = np.asarray(d["labels_test"])
        # val_combine labels are not stored and the three splits are disjoint,
        # so they cannot come from the result JSON. Rebuild the full label
        # vector from the evaluation dataset in its original order -- the
        # order split_indices() indexes into -- and only trust it if it
        # reproduces both stored label arrays exactly.
        y_full = labels_for_task(d["task"], L.shape[1])
        if y_full is None:
            print(f"  !! {pid}: dataset unavailable offline, skipping")
            continue
        if not (np.array_equal(y_full[sel], y_sel)
                and np.array_equal(y_full[test], y_test)):
            print(f"  !! {pid}: reconstructed labels disagree with the stored "
                  f"splits, skipping rather than guessing")
            continue
        y_comb = y_full[comb]

        # best_single chosen on val_selection by accuracy -- the paper's rule.
        acc_sel = [(L[i][sel].argmax(-1) == y_sel).mean() for i in range(L.shape[0])]
        b = int(np.argmax(acc_sel))

        ens_logits = L.mean(axis=0)          # soft_vote in logit space
        T_single = fit_temperature(L[b][comb], y_comb)
        T_ens = fit_temperature(ens_logits[comb], y_comb)

        A = arm(ens_logits[test], y_test)
        B = arm(L[b][test], y_test)
        C = arm(L[b][test], y_test, T_single)
        D = arm(ens_logits[test], y_test, T_ens)

        rows.append({
            "pool_id": pid, "n_adapters": int(L.shape[0]), "n_test": int(len(test)),
            "T_single": T_single, "T_ensemble": T_ens,
            "ece": {"ensemble": A[0], "single": B[0],
                    "single_temp": C[0], "ensemble_temp": D[0]},
            "accuracy": {"ensemble": A[1], "single": B[1]},
            "mean_conf": {"ensemble": A[2], "single": B[2], "single_temp": C[2]},
            "gain_vs_uncalibrated": B[0] - A[0],
            "gain_vs_temp_scaled": C[0] - A[0],
            "gain_temp_both": C[0] - D[0],
        })

    if not rows:
        print("no pool has both a logits cache and stored split indices")
        return 1

    print(f"{'pool':46s} {'T':>5s} {'ens':>7s} {'single':>7s} "
          f"{'sng+T':>7s} {'A-vs-B':>8s} {'A-vs-C':>8s}")
    print("-" * 96)
    for r in rows:
        e = r["ece"]
        print(f"{r['pool_id'][:46]:46s} {r['T_single']:5.2f} {e['ensemble']:7.4f} "
              f"{e['single']:7.4f} {e['single_temp']:7.4f} "
              f"{r['gain_vs_uncalibrated']:+8.4f} {r['gain_vs_temp_scaled']:+8.4f}")

    n = len(rows)
    w_unc = sum(r["gain_vs_uncalibrated"] > 0 for r in rows)
    w_tmp = sum(r["gain_vs_temp_scaled"] > 0 for r in rows)
    w_both = sum(r["gain_temp_both"] > 0 for r in rows)
    m_unc = float(np.mean([r["gain_vs_uncalibrated"] for r in rows]))
    m_tmp = float(np.mean([r["gain_vs_temp_scaled"] for r in rows]))
    m_both = float(np.mean([r["gain_temp_both"] for r in rows]))

    print("\nensemble better calibrated than ... (positive = ensemble wins)")
    print(f"  uncalibrated single           {w_unc}/{n} pools   mean {m_unc:+.4f}")
    print(f"  temperature-scaled single     {w_tmp}/{n} pools   mean {m_tmp:+.4f}")
    print(f"  temperature-scaled, both arms {w_both}/{n} pools   mean {m_both:+.4f}")

    conf_drop = [r["mean_conf"]["single"] - r["mean_conf"]["ensemble"] for r in rows]
    print(f"\nmechanism: ensemble is less confident than the single in "
          f"{sum(c > 0 for c in conf_drop)}/{n} pools "
          f"(mean confidence drop {np.mean(conf_drop):+.4f}); "
          f"median fitted T = {np.median([r['T_single'] for r in rows]):.2f}")

    out = {
        "note": ("Temperature fitted on the pre-registered val_combine split (reserved for "
                 "combination hyperparameters and otherwise unused), evaluated on test. "
                 "Six pools: the ones whose per-adapter logits cache was retained. These are "
                 "best_of_n pools, not the headline n_rank slice, so direction transfers and "
                 "magnitude does not."),
        "n_pools": n, "n_bins": N_BINS,
        "ensemble_better_than": {
            "uncalibrated_single": {"pools": w_unc, "mean_ece_gain": m_unc},
            "temperature_scaled_single": {"pools": w_tmp, "mean_ece_gain": m_tmp},
            "temperature_scaled_both_arms": {"pools": w_both, "mean_ece_gain": m_both},
        },
        "median_fitted_temperature": float(np.median([r["T_single"] for r in rows])),
        "pools": rows,
    }
    p = ROOT / "analysis/temperature_control.json"
    p.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
