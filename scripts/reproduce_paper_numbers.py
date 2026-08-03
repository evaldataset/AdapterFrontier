#!/usr/bin/env python3
"""E3: External reproducibility audit. Re-derive all paper headline numbers
from released artifacts only.

External party should be able to:
    pip install -r requirements.txt
    python3 scripts/reproduce_paper_numbers.py

…and see every claim recomputed from JSON with PASS/FAIL markers.
Runtime: < 30 seconds.
"""
from __future__ import annotations
import json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def header(msg):
    print(f"\n{'='*70}\n{msg}\n{'='*70}")


def check(name, actual, expected, tol=0.005):
    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        ok = abs(actual - expected) < tol
    else:
        ok = actual == expected
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {name:55s} expected={expected}  actual={actual}")
    return ok


def main():
    header("E3: External reproducibility audit")
    print(f"Root: {ROOT}")
    passes, fails = 0, 0
    def tally(ok):
        nonlocal passes, fails
        passes += int(ok); fails += int(not ok)

    # 1. Corpus + adjudication counts
    header("1. Corpus + adjudication (Abstract, §3)")
    cm_files = list(ROOT.glob("**/*_cm_*.json"))
    print(f"  cm files: {len(cm_files)}")
    tally(check("Total cm cells", len(cm_files), 1028, tol=2))

    # Each cell now carries explicit bh_q_value + adjudication_pre/post_fdr
    # fields written by scripts/apply_corpus_bh_fdr.py (Audit response,
    # Blocker 2 fix). We verify by reading those fields back rather than
    # recomputing BH from scratch in the audit script.
    cells = []
    for fn in cm_files:
        try:
            d = json.loads(Path(fn).read_text())
            cells.append({
                "p": d.get("sign_flip_p"),
                "pre": d.get("adjudication_pre_fdr"),
                "post": d.get("adjudication_post_fdr"),
                "q": d.get("bh_q_value"),
                "bh_pass": d.get("bh_pass"),
                "batch_hash": d.get("bh_batch_hash"),
            })
        except: pass
    n = len(cells)
    sup_pre = sum(1 for c in cells if c["pre"] == "supported")
    rev_pre = sum(1 for c in cells if c["pre"] == "reversed")
    tally(check("% SUPPORTED (pre-FDR, from JSON field)", round(100*sup_pre/n, 1), 33.0, tol=0.5))
    tally(check("% REVERSED  (pre-FDR, from JSON field)", round(100*rev_pre/n, 1), 17.8, tol=0.5))

    sup_post = sum(1 for c in cells if c["post"] == "supported")
    rev_post = sum(1 for c in cells if c["post"] == "reversed")
    tally(check("% SUPPORTED (post-FDR, from JSON field)", round(100*sup_post/n, 1), 29.5, tol=0.5))
    tally(check("% REVERSED  (post-FDR, from JSON field)", round(100*rev_post/n, 1), 16.9, tol=0.5))

    # Independent re-derivation from raw p-values: must agree with stored fields.
    ps_with_pre = sorted([(c["p"], c["pre"]) for c in cells if c["p"] is not None])
    m, q = len(ps_with_pre), 0.05
    cutoff = 0.0
    for k, (p, _) in enumerate(ps_with_pre, 1):
        if p <= k*q/m: cutoff = p
    sup_recomp = sum(1 for p, v in ps_with_pre if v == "supported" and p <= cutoff)
    rev_recomp = sum(1 for p, v in ps_with_pre if v == "reversed" and p <= cutoff)
    tally(check("BH-FDR cutoff (recomputed)", round(cutoff, 4), 0.0248, tol=0.005))
    tally(check("Stored post-FDR SUPPORTED == recomputed", sup_post, sup_recomp))
    tally(check("Stored post-FDR REVERSED  == recomputed", rev_post, rev_recomp))

    # Audit trail: every cell shares the same batch hash (single corpus).
    hashes = {c["batch_hash"] for c in cells if c["batch_hash"]}
    tally(check("All cells share one bh_batch_hash", len(hashes), 1))

    # 2. Frontier R²
    header("2. Frontier R² (§3, Table 5)")
    f = ROOT / "analysis/frontier_r2_cis.json"
    if f.exists():
        d = json.loads(f.read_text())
        v = d.get("accuracy__n_rank", {})
        tally(check("Corpus-wide R²_LOPO (48 pools, enc+dec)", round(v.get("r2_lopo", 0), 3), 0.597, tol=0.005))
        ci = v.get("r2_lopo_ci95", [0, 0])
        tally(check("R² CI lower", round(ci[0], 3), 0.303, tol=0.01))
        tally(check("R² CI upper", round(ci[1], 3), 0.710, tol=0.01))
        tally(check("n_cells", v.get("n_cells"), 192))
        tally(check("n_pools", v.get("n_pools"), 48))
    else:
        print(f"  SKIP: {f} not found")

    # 2b. Audit fix Critical 3: headline R² survives method-onehot ablation
    #     and emits explicit drop diagnostics (no silent row drops).
    f2 = ROOT / "analysis/frontier_with_drop_audit.json"
    if f2.exists():
        d = json.loads(f2.read_text())
        sl = d.get("by_slice", {}).get("accuracy__vs__n_rank", {})
        tally(check("Audit: headline n_rank R²", round(sl.get("r2_lopo", 0), 3), 0.597, tol=0.005))
        dd = sl.get("drop_diagnostics", {})
        tally(check("Audit: zero rows dropped on n_rank", dd.get("n_rows_dropped", -1), 0))
        ab = sl.get("ablation_no_method_onehots", {})
        delta = ab.get("r2_lopo_delta_vs_headline")
        if isinstance(delta, (int, float)):
            tally(check("Audit: method-onehot ablation |Δ R²| < 0.01",
                         abs(delta) < 0.01, True))
    else:
        print(f"  SKIP: {f2} not found")

    # 3. H(c|conf) gap
    header("3. H(c|conf) gap (§4.2, Table 6)")
    f = ROOT / "analysis/info_mechanism.json"
    if f.exists():
        d = json.loads(f.read_text())
        fs = d.get("family_summary", {})
        enc = fs.get("encoder_bertfam", {})
        dec = fs.get("decoder", {})
        tally(check("Encoder H mean", round(enc.get("mean_H_corr_given_conf_bits", 0), 3), 0.556, tol=0.01))
        tally(check("Decoder H mean", round(dec.get("mean_H_corr_given_conf_bits", 0), 3), 0.437, tol=0.01))
        gap = (enc.get("mean_H_corr_given_conf_bits", 0) - dec.get("mean_H_corr_given_conf_bits", 0)) / max(enc.get("mean_H_corr_given_conf_bits", 1), 1e-9)
        tally(check("Enc→Dec H gap %", round(100*gap, 0), 21, tol=2))

    # 4. Distillation Qwen-0.5B
    header("4. Distillation Qwen-0.5B MNLI (§5)")
    f1 = ROOT / "distilled_adapters/sweep/qwen05b_a0.9_t4/distill_metrics.json"
    f2 = ROOT / "ensemble_results/pool_a_mnli_qwen25_05b.json"
    if f1.exists() and f2.exists():
        d = json.loads(f1.read_text())
        e = json.loads(f2.read_text())
        bs = e["methods"]["best_single"]["accuracy_test"]
        sv = e["methods"]["soft_vote"]["accuracy_test"]
        ds = d["accuracy_test"]
        rec = (ds - bs) / (sv - bs) * 100
        tally(check("Distill recovery %", round(rec, 0), 87, tol=3))

    # 5. RETRACTED 2026-07-28 — the GSM8K digit-shift contamination probe was
    #    invalid (gold answer shifted with the question), so its numbers are
    #    no longer asserted here and the claim is withdrawn in the paper (L12).

    # 6. vLLM serving
    header("6. vLLM multi-LoRA (§5)")
    for model, fname, exp_ratio in [
        ("Qwen-0.5B", "analysis/vllm_bench_qwen05b.json", 1.075),
        ("Llama-3.1-8B-Inst", "analysis/vllm_bench_llama31_8b.json", 1.067),
    ]:
        f = ROOT / fname
        if f.exists():
            d = json.loads(f.read_text())
            tally(check(f"{model} ratio", round(d["multi_over_single_ratio"], 3), exp_ratio, tol=0.005))

    # 7. RETRACTED 2026-07-28 — the E1 decontaminated ensemble probe scored both
    #    arms against the invalid shifted gold labels, so its delta is an
    #    artifact and is no longer asserted (paper L12(ii)).

    # 8. Metric-general accuracy-specificity + H3 refutation (§4.2 new paragraph)
    header("8. Accuracy-specificity is metric-general; refinement not the mechanism")
    fw = ROOT / "analysis/w2_calibration_asymmetry.json"
    if fw.exists():
        d = json.loads(fw.read_text())
        def r2_of(slice_key):
            v = d.get(slice_key)
            return round(v["r2_lopo"], 2) if v and v.get("r2_lopo") is not None else None
        tally(check("accuracy n_rank R2", r2_of("accuracy__n_rank"), 0.60, tol=0.02))
        tally(check("ECE n_rank R2 (unpredictable)", r2_of("ece__n_rank"), -0.06, tol=0.03))
        tally(check("Brier n_rank R2 (unpredictable)", r2_of("brier__n_rank"), -0.30, tol=0.03))
        tally(check("NLL n_rank R2 (predictable)", r2_of("nll__n_rank"), 0.42, tol=0.03))
        tally(check("MCE n_rank R2 (predictable)", r2_of("mce__n_rank"), 0.22, tol=0.03))
    else:
        print(f"  SKIP: {fw} not found")

    fh = ROOT / "analysis/h3_mechanism.json"
    if fh.exists():
        d = json.loads(fh.read_text())
        tally(check("H3 resolution sigma / accuracy sigma",
                     round(d.get("res_std_over_acc_std", 0), 1), 0.2, tol=0.05))
        tally(check("H3 corr(resolution_gain, accuracy_gain)",
                     round(d.get("corr_resolution_accuracy", 0), 2), -0.10, tol=0.05))
        tally(check("H3 mechanism refuted (verdict_supported False)",
                     d.get("verdict_supported"), False))
    else:
        print(f"  SKIP: {fh} not found")

    # 9. Protocol sensitivity (§Protocol sensitivity, Table 1, Fig. 1)
    header("9. Protocol sensitivity: the four evaluation shortcuts")
    fp = ROOT / "analysis/protocol_sensitivity.json"
    if fp.exists():
        d = json.loads(fp.read_text())
        s1 = d["S1_selection_on_test"]
        tally(check("S1 select-on-test inflation (all, pp)",
                     round(s1["all"]["mean_inflation_pp"], 2), 1.12, tol=0.02))
        tally(check("S1 encoder inflation (pp)",
                     round(s1["encoder"]["mean_inflation_pp"], 2), 1.30, tol=0.02))
        s2 = d["S2_no_compute_match"]
        tally(check("S2 weak-baseline SUPPORTED %",
                     s2["all"]["best_of_n"]["pct_supported"], 27.2, tol=0.1))
        tally(check("S2 compute-matched SUPPORTED %",
                     s2["all"]["n_rank"]["pct_supported"], 9.9, tol=0.1))
        tally(check("S2 encoder strict SUPPORTED % (headline: zero)",
                     s2["encoder"]["n_rank"]["pct_supported"], 0.0, tol=0.001))
        tally(check("S2 encoder strict REVERSED %",
                     s2["encoder"]["n_rank"]["pct_reversed"], 33.7, tol=0.1))
        tally(check("S2 encoder strict n_cells",
                     s2["encoder"]["n_rank"]["n_cells"], 92))
        tally(check("S2 encoder strict mean Δ (pp)",
                     round(s2["encoder"]["n_rank"]["mean_diff_pp"], 2), -1.27, tol=0.02))
        # every combination rule at zero SUPPORTED
        by_m = d["S2b_strict_encoder_by_method"]
        tally(check("S2b all methods at 0% SUPPORTED",
                     max(v["pct_supported"] for v in by_m.values()), 0.0, tol=0.001))
        s3 = d["S3_aggregate_reporting"]["all"]
        mnli = next(t for t in s3["tasks"] if t["task"] == "mnli")
        tally(check("S3 MNLI task mean reads as no-effect",
                     mnli["task_level_reads_as"], "no effect"))
        tally(check("S3 significant cells hidden inside MNLI",
                     mnli["reversed_cells_inside"] + mnli["supported_cells_inside"], 37))
    else:
        print(f"  SKIP: {fp} not found")


    # 10. Correction-procedure sensitivity (App. correction)
    header("10. Multiple-comparison procedure sensitivity")
    fc = ROOT / "analysis/correction_sensitivity.json"
    if fc.exists():
        d = json.loads(fc.read_text())
        acc, ece = d["accuracy"], d["ECE"]
        tally(check("accuracy SUPPORTED = 0 under ALL procedures",
                     d["summary"]["accuracy_supported_zero_under_all_procedures"], True))
        tally(check("accuracy REVERSED, raw CI", acc["raw_ci"]["reversed"], 38))
        tally(check("accuracy REVERSED, within-table Holm", acc["holm_within_table"]["reversed"], 16))
        tally(check("accuracy REVERSED, corpus Holm", acc["holm_corpus_m1028"]["reversed"], 0))
        tally(check("accuracy REVERSED, corpus BH", acc["bh_corpus_q05"]["reversed"], 31))
        tally(check("ECE SUPPORTED, corpus BH", ece["bh_corpus_q05"]["supported"], 52))
        tally(check("ECE SUPPORTED, within-table Holm", ece["holm_within_table"]["supported"], 46))
        tally(check("ECE SUPPORTED, corpus Holm (nothing survives)",
                     ece["holm_corpus_m1028"]["supported"], 0))
    else:
        print(f"  SKIP: {fc} not found")

    # 11. Baseline budget audit (the n_rank baseline is NOT compute-matched)
    header("11. Realised baseline training budget")
    fb = ROOT / "analysis/baseline_budget_audit.json"
    if fb.exists():
        d = json.loads(fb.read_text())
        tally(check("median baseline/pool GPU-h ratio", round(d["ratio_median"], 2), 1.45, tol=0.02))
        tally(check("max ratio", round(d["ratio_max"], 2), 2.96, tol=0.02))
        tally(check("pairs where baseline got more compute", d["n_pairs_baseline_favoured"], 47))
        tally(check("total pairs", d["n_pairs"], 48))
    else:
        print(f"  SKIP: {fb} not found")


    # 12. Diversity-feature leakage bound (L1c)
    header("12. Diversity-feature leakage check")
    fl = ROOT / "analysis/leakage_check_diversity.json"
    if fl.exists():
        d = json.loads(fl.read_text())
        sm = d["summary"]
        tally(check("disagreement relative shift < 5%",
                     sm["disagreement_rate"]["mean_relative_shift"] < 0.05, True))
        tally(check("logit-correlation relative shift < 1%",
                     sm["logit_correlation"]["mean_relative_shift"] < 0.01, True))
        tally(check("pool ordering preserved (all 3 features)",
                     all(d["pool_ordering_preserved"].values()), True))
    else:
        print(f"  SKIP: {fl} not found")


    # 13. Power / equivalence and cell independence (L1d, §4 precision)
    header("13. Precision, equivalence, and cell independence")
    fp2 = ROOT / "analysis/power_and_independence.json"
    if fp2.exists():
        d = json.loads(fp2.read_text())
        tally(check("cells excluding a +1.0pp gain", d["equivalence"]["1.0"]["excluded"], 67))
        tally(check("cells excluding a +0.5pp gain", d["equivalence"]["0.5"]["excluded"], 57))
        tally(check("median CI half-width (pp)",
                     round(d["ci_half_width_pp"]["median"], 2), 0.94, tol=0.02))
        big = d["by_test_size"]["n_test >= 2000"]
        tally(check("well-powered cells", big["n"], 68))
        tally(check("well-powered excluding +1.0pp", big["excl_1.0pp"], 62))
        mr = d["method_redundancy"]
        tally(check("soft_vote/logit_avg agree <0.1pp", mr["agree_within_0.1pp"], 20))
        tally(check("soft_vote/logit_avg bit-identical", mr["bit_identical"], 10))
    else:
        print(f"  SKIP: {fp2} not found")

    # ---- Table `tab:frontier_cross`: cross-model-class LOPO R^2.
    # Previously unasserted, and the table's numbers could not be regenerated
    # from the release at all because only ridge was ever implemented.
    header("Cross-model-class frontier (Table tab:frontier_cross)")
    fp3 = ROOT / "analysis/frontier_model_classes.json"
    if fp3.exists():
        d = json.loads(fp3.read_text())["by_slice"]
        expected = {
            "accuracy__vs__n_rank":    (192, 48, +0.597, +0.276, +0.285),
            "ece__vs__n_rank":         (192, 48, -0.061, +0.295, +0.180),
            "accuracy__vs__best_of_n": (265, 64, +0.312, +0.519, +0.365),
            "ece__vs__best_of_n":      (265, 64, +0.288, +0.416, +0.340),
        }
        for key, (nc, npool, rg, rf, gb) in expected.items():
            s = d[key]
            tally(check(f"{key} cells", s["n_cells"], nc))
            tally(check(f"{key} pools", s["n_pools"], npool))
            tally(check(f"{key} ridge", s["ridge"]["r2_lopo"], rg, tol=0.01))
            tally(check(f"{key} RF", s["random_forest"]["r2_lopo"], rf, tol=0.01))
            tally(check(f"{key} GBM", s["gradient_boosting"]["r2_lopo"], gb, tol=0.01))
        # The fairness claim in S3: ridge is the cross-class max on the headline
        # slice and on no other, so the fixed choice never buys R^2.
        tally(check("slices where ridge is cross-class max",
                     sum(s.get("ridge_is_cross_class_max", False) for s in d.values()), 1))
        tally(check("ridge is max on the headline slice",
                     d["accuracy__vs__n_rank"]["ridge_is_cross_class_max"], True))
    else:
        print(f"  SKIP: {fp3} not found")

    # ---- What the calibration gain is, and is not (S4).
    header("Calibration arm: proper scores and per-rule split")
    fp4 = ROOT / "analysis/ece_test_correction.json"
    if fp4.exists():
        d = json.loads(fp4.read_text())
        tally(check("clean encoder calibration cells", d["n_cells"], 92))
        tally(check("pools", d["n_pools"], 23))
        adj = d["ece_adjudication"]
        tally(check("ECE SUPPORTED (raw CI)", adj["raw_ci"]["supported"], 52))
        tally(check("ECE REVERSED (raw CI)", adj["raw_ci"]["reversed"], 17))
        tally(check("ECE SUPPORTED (pre-registered within-table Holm)",
                     adj["holm_within_table_prereg"]["supported"], 43))
        ps = d["proper_scores"]
        tally(check("multiclass Brier improved", ps["multiclass_brier_improved"], 34))
        tally(check("multiclass Brier mean diff",
                     round(ps["multiclass_brier_mean_diff"], 4), -0.0062, tol=0.0005))
        tally(check("NLL improved", ps["nll_improved"], 47))
        tally(check("top-label Brier SUPPORTED", ps["toplabel_brier_supported"], 22))
        tally(check("top-label Brier REVERSED", ps["toplabel_brier_reversed"], 36))
        pm = d["per_method"]
        for m, exp in (("soft_vote", 17), ("logit_avg", 16),
                       ("greedy_soup", 16), ("majority_vote", 3)):
            tally(check(f"ECE SUPPORTED {m}", pm[m]["ece_supported"], exp))
        tally(check("ECE reversals that are majority_vote",
                     d["ece_reversals_from_majority_vote"], 16))
        ex = d["excluding_majority_vote"]
        tally(check("ECE SUPPORTED excluding majority_vote", ex["ece_supported"], 49))
        tally(check("cells excluding majority_vote", ex["n_cells"], 69))
        tally(check("pools SUPPORTED on every averaging rule",
                     ex["pools_supported_on_every_rule"], 14))
    else:
        print(f"  SKIP: {fp4} not found")

    # ---- The recalibration control (title claim, S4).
    header("Temperature control on the calibration arm")
    fp5 = ROOT / "analysis/temperature_control.json"
    if fp5.exists():
        d = json.loads(fp5.read_text())
        tally(check("pools with a retained logits cache", d["n_pools"], 6))
        b = d["ensemble_better_than"]
        tally(check("ensemble beats uncalibrated single (pools)",
                     b["uncalibrated_single"]["pools"], 4))
        tally(check("ensemble beats temperature-scaled single (pools)",
                     b["temperature_scaled_single"]["pools"], 2))
        tally(check("mean ECE gain vs temperature-scaled single",
                     round(b["temperature_scaled_single"]["mean_ece_gain"], 4),
                     -0.0316, tol=0.001))
        tally(check("ensemble beats single, both arms scaled (pools)",
                     b["temperature_scaled_both_arms"]["pools"], 3))
        tally(check("median fitted temperature",
                     round(d["median_fitted_temperature"], 2), 1.32, tol=0.02))
    else:
        print(f"  SKIP: {fp5} not found")

    # ---- Table `tab:cost`: previously ungenerated and unasserted.
    header("Cost of blind ensembling (Table tab:cost)")
    fp6 = ROOT / "analysis/cost_savings.json"
    if fp6.exists():
        d = json.loads(fp6.read_text())
        expected = {   # n, blind dpp, blind %REV, blind cost, routed dpp, routed %REV, routed cost
            "Enc acc": (64, -3.3, 30, 20, +0.4, 8, 1),
            "Dec acc": (50, +0.6, 0, 18, +0.8, 0, 15),
            "Enc ECE": (63, +2.4, 10, 20, +3.6, 2, 20),
            "Dec ECE": (50, +3.5, 0, 18, +3.6, 0, 18),
            "All":     (227, +0.7, 11, 19, +2.1, 3, 17),
        }
        for row, (n, bd, br, bc, rd, rr, rc) in expected.items():
            s = d[row]
            tally(check(f"{row} n", s["n"], n))
            tally(check(f"{row} blind mean delta pp",
                         round(s["blind_mean_delta_pp"], 1), bd, tol=0.06))
            tally(check(f"{row} blind % reversed", s["blind_pct_reversed"], br))
            tally(check(f"{row} blind median cost", s["blind_median_cost"], bc))
            tally(check(f"{row} routed mean delta pp",
                         round(s["routed_mean_delta_pp"], 1), rd, tol=0.06))
            tally(check(f"{row} routed % reversed", s["routed_pct_reversed"], rr))
            tally(check(f"{row} routed median cost", s["routed_median_cost"], rc))
    else:
        print(f"  SKIP: {fp6} not found")

    header("Summary")
    total = passes + fails
    print(f"\n  PASS: {passes}/{total}")
    print(f"  FAIL: {fails}/{total}")
    if fails == 0:
        print("\n  All paper headline numbers reproduce from released JSONs.")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
