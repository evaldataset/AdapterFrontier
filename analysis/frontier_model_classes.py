#!/usr/bin/env python3
"""Table `tab:frontier_cross` compares three model classes; only one existed.

The paper reports ridge, random forest and gradient boosting per frontier
slice and says ridge is reported as the headline "so the cross-class maximum
is never the reported number". That fairness argument needs the other two
classes to actually have been fit. They never were: `frontier.py` implements
ridge and nothing else, and no RandomForest or GradientBoosting appears
anywhere in the repository. The numbers in the table could not be regenerated
from the release, and its ridge entry (+0.555) contradicted the paper's own
headline (+0.597) for the identical 192-cell / 48-pool slice.

This script fits all three classes under the fold structure `frontier.py`
already uses -- same slice construction, same design matrix (method one-hots
included), same leave-one-pool-out folds, same skip rule when a fold has
fewer training rows than features, same global R^2 on the pooled held-out
predictions. Ridge is delegated to `frontier.lopo_cv_r2` itself so the
headline column is the headline number by construction rather than by
coincidence.

Usage: python3 analysis/frontier_model_classes.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "analysis"))
from frontier import (  # noqa: E402
    build_design_matrix_diagnostics,
    load_cm_cells,
    load_diversity,
    lopo_cv_r2,
)

SEED = 0
METRICS = ("accuracy", "ece")
KINDS = ("best_of_n", "n_rank")


def make_model(name: str):
    if name == "random_forest":
        return RandomForestRegressor(n_estimators=300, min_samples_leaf=2,
                                     random_state=SEED, n_jobs=1)
    if name == "gradient_boosting":
        return GradientBoostingRegressor(n_estimators=200, max_depth=2,
                                         learning_rate=0.05, random_state=SEED)
    raise ValueError(name)


def lopo_sklearn(X, y, pools, name: str) -> dict:
    """Same folds and skip rule as frontier.lopo_cv_r2, different estimator."""
    unique = sorted(set(pools))
    if len(unique) < 2:
        return {"error": f"need >=2 pools, got {len(unique)}"}
    preds = np.full_like(y, np.nan)
    used, skipped = [], 0
    for p in unique:
        mask = np.array([pp == p for pp in pools])
        Xtr, ytr, Xte = X[~mask], y[~mask], X[mask]
        if len(Xtr) < X.shape[1]:          # identical skip rule
            skipped += 1
            continue
        used.append(p)
        mu, sd = Xtr.mean(axis=0), Xtr.std(axis=0)
        sd[sd == 0] = 1.0
        m = make_model(name)
        m.fit((Xtr - mu) / sd, ytr)
        preds[mask] = m.predict((Xte - mu) / sd)
    ok = ~np.isnan(preds)
    if ok.sum() == 0:
        return {"error": "no LOPO predictions", "n_skipped_folds": skipped}
    yo, po = y[ok], preds[ok]
    tss = float(((yo - yo.mean()) ** 2).sum())
    rss = float(((yo - po) ** 2).sum())
    return {"r2_lopo": float(1.0 - rss / tss) if tss > 0 else float("nan"),
            "n_rows": int(len(y)), "n_pools": len(unique),
            "n_used_folds": len(used), "n_skipped_folds": skipped}


def main() -> int:
    adir = ROOT / "analysis"
    diversity = load_diversity(adir)
    cells = [c for c in load_cm_cells(adir) if not c["pool_id"].startswith("baseline_")]

    out = {
        "note": ("Cross-model-class comparison for the diversity-quality frontier. "
                 "Ridge is computed by frontier.lopo_cv_r2 (the function that "
                 "produces the paper's headline R^2); random forest and gradient "
                 "boosting reuse the same slices, design matrix and LOPO folds. "
                 "Reported so the claim that ridge is not the cross-class maximum "
                 "can be checked rather than asserted."),
        "seed": SEED, "by_slice": {},
    }

    hdr = f"{'slice':28s} {'cells':>6s} {'pools':>6s} {'ridge':>9s} {'RF':>9s} {'GBM':>9s}"
    print(hdr); print("-" * len(hdr))
    for metric in METRICS:
        for kind in KINDS:
            rows = [c for c in cells
                    if c["metric"] == metric and c["baseline_kind"] == kind]
            diag = build_design_matrix_diagnostics(rows, diversity,
                                                   include_method_onehots=True)
            X, y, used = diag["X"], diag["y"], diag["used_rows"]
            key = f"{metric}__vs__{kind}"
            if len(used) < 3:
                out["by_slice"][key] = {"error": f"not enough rows: {len(used)}"}
                continue
            pools = [r["pool_id"] for r in used]
            res = {
                "n_cells": int(len(used)), "n_pools": len(set(pools)),
                "ridge": lopo_cv_r2(X, y, pools, ridge=0.1),
                "random_forest": lopo_sklearn(X, y, pools, "random_forest"),
                "gradient_boosting": lopo_sklearn(X, y, pools, "gradient_boosting"),
            }
            r = {k: res[k].get("r2_lopo") for k in
                 ("ridge", "random_forest", "gradient_boosting")}
            res["ridge_is_cross_class_max"] = bool(
                r["ridge"] is not None
                and all(v is not None and r["ridge"] >= v for v in r.values()))
            res["cross_class_max"] = max(v for v in r.values() if v is not None)
            out["by_slice"][key] = res
            print(f"{key:28s} {res['n_cells']:6d} {res['n_pools']:6d} "
                  f"{r['ridge']:+9.3f} {r['random_forest']:+9.3f} "
                  f"{r['gradient_boosting']:+9.3f}"
                  + ("   <- ridge is max" if res["ridge_is_cross_class_max"] else ""))

    n_max = sum(v.get("ridge_is_cross_class_max", False)
                for v in out["by_slice"].values())
    out["n_slices"] = len(out["by_slice"])
    out["n_slices_where_ridge_is_max"] = n_max
    print(f"\nridge is the cross-class maximum on {n_max}/{len(out['by_slice'])} slices")

    p = ROOT / "analysis/frontier_model_classes.json"
    p.write_text(json.dumps(out, indent=2))
    print(f"wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
