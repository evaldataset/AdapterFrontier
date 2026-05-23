#!/usr/bin/env python3
"""Tier 1 analysis batch (no cluster, no local GPU needed):
T1.1 stratified decoder LOPO frontier
T1.2 per-family feature ablation
T1.4 random subset vs greedy_soup
T1.5 confidence distribution comparison

All run on existing ensemble_results + analysis/*_cm_*.json + cached logits.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from frontier import (
    load_diversity, load_cm_cells, build_design_matrix,
    lopo_cv_r2, in_sample_fit,
)


def family(pool_id: str) -> str:
    s = pool_id.lower()
    if "qwen" in s or "llama" in s:
        return "decoder"
    if "deberta" in s:
        return "encoder_deberta"
    return "encoder_bertfam"


def t11_stratified_decoder_lopo(diversity, all_cells):
    """Decoder-only frontier LOPO (now we have ≥9 decoder pools)."""
    out = {}
    decoder_cells = [c for c in all_cells if family(c["pool_id"]) == "decoder"]
    print(f"\n=== T1.1 stratified decoder LOPO ({len(set(c['pool_id'] for c in decoder_cells))} decoder pools) ===")
    for metric in ("accuracy", "ece"):
        for kind in ("best_of_n", "n_rank"):
            rows = [c for c in decoder_cells if c["metric"] == metric and c["baseline_kind"] == kind]
            X, y, fnames, used = build_design_matrix(rows, diversity)
            pools = [r["pool_id"] for r in used]
            if len(set(pools)) < 3:
                out[f"{metric}__{kind}"] = {"note": f"{len(set(pools))} pools — too few"}
                continue
            lopo = lopo_cv_r2(X, y, pools)
            ins = in_sample_fit(X, y, fnames)
            out[f"{metric}__{kind}"] = {
                "n_rows": len(used), "n_pools": len(set(pools)),
                "r2_lopo": lopo.get("r2_lopo"), "r2_in_sample": ins["r2_in_sample"],
                "pools": sorted(set(pools)),
            }
            r2 = lopo.get("r2_lopo")
            r2s = f"{r2:+.3f}" if r2 is not None else "n/a"
            print(f"  {metric:>9}__{kind:>10}  n={len(used):3d} pools={len(set(pools)):2d} R²_LOPO={r2s} R²_IS={ins['r2_in_sample']:+.3f}")
    return out


def t12_per_family_feature_ablation(diversity, all_cells):
    """Re-run per-feature drop on encoder vs decoder slices separately."""
    GROUPS = {
        "prediction_space": ["pred_disagreement_rate", "pred_logit_correlation", "pred_ensemble_entropy"],
        "parameter_space":  ["par_mean_cosine_distance_lora_A", "par_mean_cosine_distance_lora_B",
                             "par_mean_frobenius_distance"],
        "hp_entropy":       ["hp_seed", "hp_peft_method", "hp_lora_r", "hp_lora_alpha", "hp_lr", "hp_total"],
        "method_indicator": ["method_greedy_soup", "method_soft_vote", "method_logit_avg", "method_majority_vote"],
    }
    out = {}
    for fam_name in ("encoder_bertfam", "decoder"):
        fam_cells = [c for c in all_cells if family(c["pool_id"]) == fam_name]
        out[fam_name] = {}
        for slice_key in ("accuracy__vs__n_rank", "ece__vs__n_rank"):
            metric, kind = slice_key.split("__vs__")
            rows = [c for c in fam_cells if c["metric"] == metric and c["baseline_kind"] == kind]
            X, y, fnames, used = build_design_matrix(rows, diversity)
            pools = [r["pool_id"] for r in used]
            if len(set(pools)) < 3:
                out[fam_name][slice_key] = {"note": f"{len(set(pools))} pools"}
                continue
            base_lopo = lopo_cv_r2(X, y, pools)
            base_r2 = base_lopo.get("r2_lopo")
            sl = {"baseline_r2_lopo": base_r2, "n_pools": len(set(pools)), "n_rows": len(used)}
            for gname, gcols in GROUPS.items():
                keep = [i for i, f in enumerate(fnames) if f not in gcols]
                if not keep:
                    continue
                Xs = X[:, keep]
                lopo = lopo_cv_r2(Xs, y, pools)
                drop = (base_r2 - lopo.get("r2_lopo")) if (base_r2 is not None and lopo.get("r2_lopo") is not None) else None
                sl[f"drop_{gname}"] = {
                    "r2_lopo": lopo.get("r2_lopo"),
                    "r2_drop": drop,
                }
            out[fam_name][slice_key] = sl
    print(f"\n=== T1.2 per-family feature ablation (drop = baseline - holdout, larger = more important) ===")
    for fam_name, slices in out.items():
        print(f"\n  Family: {fam_name}")
        for slice_key, sl in slices.items():
            if "note" in sl:
                print(f"    {slice_key}: {sl['note']}")
                continue
            base = sl["baseline_r2_lopo"]
            base_s = f"{base:+.3f}" if base is not None else "n/a"
            print(f"    {slice_key}: baseline R²_LOPO={base_s}, n_pools={sl['n_pools']}")
            for k, v in sl.items():
                if not k.startswith("drop_"):
                    continue
                drop = v.get("r2_drop")
                drop_s = f"{drop:+.4f}" if drop is not None else "n/a"
                r2 = v.get("r2_lopo")
                r2s = f"{r2:+.3f}" if r2 is not None else "n/a"
                print(f"      drop {k[5:]:>20}  → R²={r2s}  Δ={drop_s}")
    return out


def t14_random_subset_vs_greedy(diversity):
    """For each pool with cached logits, compare greedy_soup to random subsets."""
    rng = np.random.RandomState(0)
    cache_dir = Path("ensemble_cache")
    pools_dir = Path("pools")
    out = {}
    print(f"\n=== T1.4 random subset baseline vs greedy_soup ===")
    for p in sorted(pools_dir.glob("pool_*.json")):
        pool = json.loads(p.read_text())
        pool_id = pool.get("pool_id") or pool.get("pool_name") or p.stem
        adapters = pool["adapters"]
        if not adapters:
            continue
        # Find the corresponding ensemble_result for labels_test + greedy_soup info
        ens_path = Path(f"ensemble_results/{pool_id}.json")
        if not ens_path.exists():
            continue
        ens = json.loads(ens_path.read_text())
        labels_test = np.asarray(ens.get("labels_test", []), dtype=np.int64)
        test_indices = np.asarray(ens.get("test_indices", []), dtype=np.int64)
        if labels_test.size == 0 or test_indices.size != labels_test.size:
            continue
        greedy = ens.get("methods", {}).get("greedy_soup", {})
        greedy_acc = greedy.get("accuracy_test")
        greedy_chosen = greedy.get("chosen_count")
        if not greedy_acc or not greedy_chosen:
            continue
        # Load cached logits (full per-adapter, before test split)
        logits_list = []
        for a in adapters:
            aid = a["adapter_id"]
            cache = cache_dir / f"{pool_id}__{aid}.npy"
            if not cache.exists():
                logits_list = None; break
            logits_list.append(np.load(cache))
        if logits_list is None:
            continue
        N = len(logits_list)
        if N < greedy_chosen + 1:
            continue
        # Random subsets at greedy_chosen size — 100 trials, evaluated on the saved test split
        accs = []
        for _ in range(100):
            sub = rng.choice(N, size=greedy_chosen, replace=False)
            stack = np.stack([logits_list[i][test_indices] for i in sub], axis=0)
            preds = stack.mean(axis=0).argmax(axis=-1)
            accs.append(float((preds == labels_test).mean()))
        out[pool_id] = {
            "n_adapters": N, "greedy_chosen": greedy_chosen,
            "greedy_acc": greedy_acc,
            "random_subset_mean_acc": float(np.mean(accs)),
            "random_subset_std_acc": float(np.std(accs)),
            "random_better_than_greedy_pct": float(100 * sum(a > greedy_acc for a in accs) / len(accs)),
        }
        rs = out[pool_id]
        print(f"  {pool_id:<46s}  greedy={rs['greedy_acc']:.4f}  rand_mean={rs['random_subset_mean_acc']:.4f}±{rs['random_subset_std_acc']:.4f}  rand_beats_greedy={rs['random_better_than_greedy_pct']:.0f}%")
    return out


def t15_confidence_distribution(diversity, all_cells):
    """Per-method confidence distribution percentiles, by family."""
    by_family_method = defaultdict(list)
    for p in sorted(Path("ensemble_results").glob("pool_*.json")):
        ens = json.loads(p.read_text())
        pool_id = ens.get("pool_id") or ens.get("pool_name") or p.stem
        fam = family(pool_id)
        for method, m in (ens.get("methods") or {}).items():
            conf = m.get("confidence_test")
            if not conf:
                continue
            arr = np.asarray(conf, dtype=np.float64)
            if arr.size == 0:
                continue
            by_family_method[(fam, method)].append({
                "pool": pool_id, "n": int(arr.size),
                "median": float(np.median(arr)),
                "p10": float(np.quantile(arr, 0.1)),
                "p90": float(np.quantile(arr, 0.9)),
                "mean": float(arr.mean()),
                "std": float(arr.std()),
            })
    out = {}
    print(f"\n=== T1.5 confidence distribution by family × method (median ± std across pools) ===")
    for (fam, method), entries in sorted(by_family_method.items()):
        med = np.array([e["median"] for e in entries])
        spread = np.array([e["p90"] - e["p10"] for e in entries])
        out[f"{fam}__{method}"] = {
            "n_pools": len(entries),
            "median_of_medians": float(med.mean()),
            "median_p10p90_spread": float(spread.mean()),
        }
        print(f"  {fam:<22s} {method:<14s}  n_pools={len(entries):2d}  median≈{med.mean():.3f}  p90-p10 spread≈{spread.mean():.3f}")
    return out


def main() -> int:
    diversity = load_diversity(Path("analysis"))
    all_cells = load_cm_cells(Path("analysis"))
    all_cells = [c for c in all_cells if not c["pool_id"].startswith("baseline_")]

    out = {}
    out["t11_stratified_decoder_lopo"] = t11_stratified_decoder_lopo(diversity, all_cells)
    out["t12_per_family_feature_ablation"] = t12_per_family_feature_ablation(diversity, all_cells)
    out["t14_random_subset_vs_greedy"] = t14_random_subset_vs_greedy(diversity)
    out["t15_confidence_distribution"] = t15_confidence_distribution(diversity, all_cells)

    Path("analysis/tier1_results.json").write_text(json.dumps(out, indent=2, default=float))
    print(f"\nwrote analysis/tier1_results.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
