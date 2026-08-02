#!/usr/bin/env python3
"""C1 prospective-policy harness: nested LOTO/LOMO evaluation of a
validation-only routing policy, sealed before unsealing (Limitation L11).

Implements, exactly as pre-registered:
  §4 features   — validation-side only, from schema-v2 result JSONs + logits
                  cache sliced by val_selection_indices. Test-side fields are
                  never read for features (enforced by construction: this
                  module never touches individual_*_test for X).
  §5 folds      — outer LOTO (task) and LOMO (model family); inner 5-fold CV
                  over pools fixes ridge lambda and abstention margin tau.
  §2 utility    — u_lambda(m) = dacc(m) - lambda * c_norm(m),
                  c_norm = (forwards-1)/(N_pool-1), lambda in {0, .01, .05}.
  §6 baselines  — blind soft_vote, best_single, random-method (100 draws),
                  oracle (labeled).
  §8 controls   — feature-shuffle and label-shuffle negative controls.
  §10 sealing   — outer-fold utilities are written to analysis/c1_sealed/
                  WITHOUT being printed. Console shows only: row counts,
                  fold structure, and negative-control PASS/FAIL. Unsealing
                  is a separate deliberate step (c1_unseal.py, to be run only
                  after controls pass and protocol compliance is confirmed).

Outcome (dacc) source: pilot result JSONs in ensemble_results/pilot_c1/,
paired against the OLD n_rank baseline result for the same task IFF
test_indices match exactly (alignment assert). Cells without an aligned
n_rank baseline fall back to best_of_n (in-file best_single), flagged.

Usage:
    python3 analysis/c1_policy_harness.py            # run sealed
"""
from __future__ import annotations
import json, glob, hashlib, itertools
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SEALED_DIR = ROOT / "analysis/c1_sealed"
LAMBDAS = (0.0, 0.01, 0.05)
METHODS = ("best_single", "soft_vote", "logit_avg", "majority_vote", "greedy_soup")
RIDGE_GRID = (0.01, 0.1, 1.0)
TAU_GRID = (0.0, 0.002, 0.005, 0.01)
RNG_SEED = 0

TASK_OF = {}     # pool_id -> task (filled from result JSONs)
FAMILY_TAGS = (("roberta", "roberta"), ("deberta", "deberta"),
               ("bert_large", "bert"), ("bert", "bert"))


def family_of(pool_id: str) -> str:
    p = pool_id.lower()
    for tag, fam in FAMILY_TAGS:
        if tag in p:
            return fam
    return "other"


def forwards_of(method: str, n_pool: int, method_block: dict) -> float:
    if method == "best_single":
        return 1.0
    if method == "greedy_soup":
        k = method_block.get("chosen_count") or len(method_block.get("chosen_indices", [])) or n_pool
        return float(k)
    return float(n_pool)


def c_norm(forwards: float, n_pool: int) -> float:
    return (forwards - 1.0) / max(n_pool - 1.0, 1.0)


# ---------------------------------------------------------------- features
def val_features(d: dict, cache_dir: Path) -> dict | None:
    """Prereg §4 allowed features, val_selection side only."""
    accs = d.get("individual_accuracy_val_selection")
    preds = d.get("individual_predictions_val_selection")
    sel_idx = d.get("val_selection_indices")
    if not (accs and preds and sel_idx):
        return None
    A = np.asarray(accs, dtype=float)
    P = np.asarray(preds)                       # (M, n_sel)
    M = P.shape[0]
    # pairwise disagreement on val_selection
    dis = [float((P[i] != P[j]).mean()) for i, j in itertools.combinations(range(M), 2)]
    # pairwise probability correlation on val_selection via cached logits
    corr = None
    pool_id = d.get("pool_id")
    npys = sorted(cache_dir.glob(f"{pool_id}__*.npy"))
    if len(npys) == M:
        sel = np.asarray(sel_idx)
        flat = []
        for f in npys:
            L = np.load(f)[sel]                 # (n_sel, K)
            e = np.exp(L - L.max(-1, keepdims=True))
            pr = e / e.sum(-1, keepdims=True)
            flat.append(pr.reshape(-1))
        cs = [float(np.corrcoef(flat[i], flat[j])[0, 1])
              for i, j in itertools.combinations(range(M), 2)]
        corr = float(np.mean(cs))
    return {
        "val_acc_mean": float(A.mean()), "val_acc_max": float(A.max()),
        "val_acc_std": float(A.std()), "val_acc_gap": float(A.max() - A.mean()),
        "disagreement": float(np.mean(dis)),
        "prob_corr": corr if corr is not None else 0.0,
        "prob_corr_missing": float(corr is None),
        "n_pool": float(M),
        "pool_type_b": float("pool_b" in pool_id), "pool_type_c": float("pool_c" in pool_id),
    }


# ---------------------------------------------------------------- outcomes
EXCLUDED_NO_BASELINE: list[str] = []


def load_cells() -> list[dict]:
    """One row per (pilot pool, method): features + dacc/utility candidates."""
    rows = []
    for fp in sorted(glob.glob(str(ROOT / "ensemble_results/pilot_c1/*.json"))):
        d = json.loads(Path(fp).read_text())
        pool_id, task = d["pool_id"], d["task"]
        TASK_OF[pool_id] = task
        feats = val_features(d, ROOT / "ensemble_cache")
        if feats is None:
            continue
        n_pool = d["n_adapters"]
        # Baseline: aligned n_rank result ONLY (prereg §2/§9 — headline is the
        # compute-matched n_rank comparison). Pools without an exactly aligned
        # n_rank baseline (same test_indices) are EXCLUDED and counted, never
        # silently re-based onto a different comparison.
        base_acc, base_kind = None, None
        for bp in glob.glob(str(ROOT / f"ensemble_results/baseline_compute_matched_{task}_*.json")):
            b = json.loads(Path(bp).read_text())
            if b.get("test_indices") == d.get("test_indices"):
                bs = b.get("methods", {}).get("best_single", {})
                if bs.get("accuracy_test") is not None:
                    base_acc, base_kind = float(bs["accuracy_test"]), "n_rank"
                    break
        if base_acc is None:
            EXCLUDED_NO_BASELINE.append(pool_id)
            continue
        for m in METHODS:
            mb = d["methods"].get(m)
            if not mb or mb.get("accuracy_test") is None:
                continue
            dacc = float(mb["accuracy_test"]) - base_acc
            fw = forwards_of(m, n_pool, mb)
            rows.append({
                "pool_id": pool_id, "task": task, "family": family_of(pool_id),
                "method": m, "dacc": dacc, "c_norm": c_norm(fw, n_pool),
                "baseline_kind": base_kind, "features": feats,
            })
    return rows


# ---------------------------------------------------------------- policy
FEATS = ("val_acc_mean", "val_acc_max", "val_acc_std", "val_acc_gap",
         "disagreement", "prob_corr", "prob_corr_missing", "n_pool",
         "pool_type_b", "pool_type_c")


def xy(rows):
    X = np.array([[r["features"][f] for f in FEATS] for r in rows])
    y = np.array([r["dacc"] for r in rows])
    return X, y


def fit_ridge(X, y, lam):
    mu, sd = X.mean(0), X.std(0); sd[sd == 0] = 1
    Xs = np.column_stack([np.ones(len(X)), (X - mu) / sd])
    I = np.eye(Xs.shape[1]); I[0, 0] = 0
    beta = np.linalg.solve(Xs.T @ Xs + lam * I, Xs.T @ y)
    return {"beta": beta, "mu": mu, "sd": sd}


def predict(model, X):
    Xs = np.column_stack([np.ones(len(X)), (X - model["mu"]) / model["sd"]])
    return Xs @ model["beta"]


def run_policy(train_rows, test_rows, rng):
    """Fit dacc-regressor on train; on test pools pick argmax utility with abstention."""
    if len(train_rows) < 8 or not test_rows:
        return None
    # inner CV over train pools picks ridge lam + tau (joint, utility@lambda avg)
    pools = sorted(set(r["pool_id"] for r in train_rows))
    best = None
    for lam_r in RIDGE_GRID:
        for tau in TAU_GRID:
            inner_u = []
            for i in range(min(5, len(pools))):
                hold = set(pools[i::5])
                tr = [r for r in train_rows if r["pool_id"] not in hold]
                te = [r for r in train_rows if r["pool_id"] in hold]
                if len(tr) < 6 or not te:
                    continue
                mdl = fit_ridge(*xy(tr), lam_r)
                inner_u.append(np.mean(decide_and_score(mdl, te, tau)["utility_mean_over_lambdas"]))
            if inner_u and (best is None or np.mean(inner_u) > best[0]):
                best = (float(np.mean(inner_u)), lam_r, tau)
    _, lam_r, tau = best if best else (0.0, 0.1, 0.005)
    mdl = fit_ridge(*xy(train_rows), lam_r)
    return {"model": mdl, "lam_ridge": lam_r, "tau": tau,
            "outer": decide_and_score(mdl, test_rows, tau)}


def decide_and_score(mdl, rows, tau):
    """Group rows by pool; pick method; score utilities and REVERSED."""
    by_pool: dict[str, list] = {}
    for r in rows:
        by_pool.setdefault(r["pool_id"], []).append(r)
    out = {lam: {"policy": [], "soft": [], "single": [], "oracle": [], "random": []}
           for lam in LAMBDAS}
    rng = np.random.RandomState(RNG_SEED)
    reversed_ct = {"policy": 0, "soft": 0}
    n_pools = 0
    for pid, rws in by_pool.items():
        n_pools += 1
        X, _ = xy(rws)
        pred = predict(mdl, X)
        for lam in LAMBDAS:
            u_pred = pred - lam * np.array([r["c_norm"] for r in rws])
            order = np.argsort(-u_pred)
            top, second = order[0], (order[1] if len(order) > 1 else order[0])
            pick = top
            if u_pred[top] - u_pred[second] < tau or u_pred[top] < 0:
                # abstain -> best_single row
                singles = [i for i, r in enumerate(rws) if r["method"] == "best_single"]
                pick = singles[0] if singles else top
            def u_actual(i):
                return rws[i]["dacc"] - lam * rws[i]["c_norm"]
            out[lam]["policy"].append(u_actual(pick))
            soft = [i for i, r in enumerate(rws) if r["method"] == "soft_vote"]
            sing = [i for i, r in enumerate(rws) if r["method"] == "best_single"]
            if soft: out[lam]["soft"].append(u_actual(soft[0]))
            if sing: out[lam]["single"].append(u_actual(sing[0]))
            out[lam]["oracle"].append(max(u_actual(i) for i in range(len(rws))))
            out[lam]["random"].append(float(np.mean([u_actual(i) for i in
                rng.randint(0, len(rws), size=100)])))
            if lam == 0.0:
                if rws[pick]["dacc"] < 0: reversed_ct["policy"] += 1
                if soft and rws[soft[0]]["dacc"] < 0: reversed_ct["soft"] += 1
    return {
        "n_pools": n_pools,
        "utilities": {str(lam): {k: v for k, v in d.items()} for lam, d in out.items()},
        "utility_mean_over_lambdas": [float(np.mean(out[lam]["policy"])) for lam in LAMBDAS
                                      if out[lam]["policy"]],
        "reversed": reversed_ct,
    }


def main():
    SEALED_DIR.mkdir(parents=True, exist_ok=True)
    rows = load_cells()
    tasks = sorted(set(r["task"] for r in rows))
    fams = sorted(set(r["family"] for r in rows))
    print(f"[c1] rows={len(rows)} pools={len(set(r['pool_id'] for r in rows))} "
          f"tasks={tasks} families={fams}")
    print(f"[c1] excluded (no aligned n_rank baseline): {len(EXCLUDED_NO_BASELINE)} "
          f"{EXCLUDED_NO_BASELINE}")

    rng = np.random.RandomState(RNG_SEED)
    sealed = {"loto": {}, "lomo": {}, "controls": {}}

    for task in tasks:
        tr = [r for r in rows if r["task"] != task]
        te = [r for r in rows if r["task"] == task]
        res = run_policy(tr, te, rng)
        sealed["loto"][task] = res["outer"] if res else None
    for fam in fams:
        tr = [r for r in rows if r["family"] != fam]
        te = [r for r in rows if r["family"] == fam]
        res = run_policy(tr, te, rng)
        sealed["lomo"][fam] = res["outer"] if res else None

    # §8 negative controls (these ARE printed)
    def control_utility(shuffle_what: str) -> float:
        vals = []
        for task in tasks:
            tr = [dict(r) for r in rows if r["task"] != task]
            te = [r for r in rows if r["task"] == task]
            if shuffle_what == "features":
                perm = rng.permutation(len(tr))
                fcopy = [tr[i]["features"] for i in perm]
                for r, f in zip(tr, fcopy):
                    r["features"] = f
            else:  # labels
                perm = rng.permutation(len(tr))
                dcopy = [tr[i]["dacc"] for i in perm]
                for r, d_ in zip(tr, dcopy):
                    r["dacc"] = d_
            res = run_policy(tr, te, rng)
            if res:
                u = res["outer"]["utilities"]["0.0"]
                if u["policy"] and u["soft"]:
                    vals.append(float(np.mean(u["policy"]) - np.mean(u["soft"])))
        return float(np.mean(vals)) if vals else float("nan")

    fs = control_utility("features")
    ls = control_utility("labels")
    sealed["controls"] = {"feature_shuffle_adv_vs_soft": fs, "label_shuffle_adv_vs_soft": ls}
    fs_pass = not (fs > 0.002)   # shuffled must NOT beat soft-vote meaningfully
    ls_pass = not (ls > 0.002)
    print(f"[c1] NEGATIVE CONTROL feature-shuffle: {'PASS' if fs_pass else 'FAIL'}")
    print(f"[c1] NEGATIVE CONTROL label-shuffle  : {'PASS' if ls_pass else 'FAIL'}")

    blob = json.dumps(sealed, indent=2, default=float)
    out = SEALED_DIR / "c1_pilot_sealed.json"
    out.write_text(blob)
    print(f"[c1] sealed results written to {out} (sha256 {hashlib.sha256(blob.encode()).hexdigest()[:16]})")
    print("[c1] outer-fold utilities NOT printed — unseal via analysis/c1_unseal.py "
          "only after controls pass and protocol compliance is confirmed (prereg §10).")


if __name__ == "__main__":
    main()
