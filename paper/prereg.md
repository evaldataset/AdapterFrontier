# Pre-Registration — Adapter Populations

**Project**: Adapter Populations (PEFT adapters as populations vs isolated checkpoints)
**Target venue**: EMNLP 2026 (direct ~June 2026 / ARR ~August 2026). Quality first; venue may slip to ACL/EACL 2027 if Phase 2 is not ready.
**Pre-registration version**: 0.1 (locked at start of Phase 1)
**Date locked**: 2026-04-21
**Status**: Phase 0 in progress; this document becomes binding once Phase 0 exit gate passes.
**Amendments**: tracked in `paper/prereg_amendments.md` with date, rationale, and impact on existing claims.

---

## 1. Central Claim

We pre-register a **conjoined hypothesis with two arms**:

> Under fixed compute, a *diversity–quality frontier* of an adapter pool predicts both
>   (A) the ensemble's accuracy gain over a compute-matched single adapter, and
>   (C) the ensemble's calibration/reliability improvement (ECE reduction, Brier reduction)
> over the same baseline.

Each arm is independently adjudicable. Either supports a paper; both supported is the strongest outcome. Neither supported triggers a pivot per §10.

**Operational form** of the predictions: per pool *p* and combination method *m*, define
- `Δacc(p, m) = acc_ensemble(p, m) − acc_baseline(p, kind)`
- `Δece(p, m) = ece_baseline(p, kind) − ece_ensemble(p, m)` (positive = ensemble more calibrated)

Then the frontier is a regression
- `Δacc ≈ f_A(diversity_features(p))`
- `Δece ≈ f_C(diversity_features(p))`
fit across all (pool × method × baseline_kind) cells, with R² on held-out pools as the primary success metric.

We are **not** claiming "ensembles always win." We are claiming that the frontier *predicts when they win and by how much*.

---

## 2. Data and Splits

### Tasks (Phase 1 mandatory)
- **MNLI** (`nyu-mll/glue` config `mnli`): 393K train, 9.8K validation_matched
- **SST-2** (`nyu-mll/glue` config `sst2`): 67K train, 872 validation

### Tasks (Phase 2 stretch)
- **BoolQ** (NLI-adjacent QA), **CSQA** or **HellaSwag** (commonsense), **HumanEval** (code), **GSM8K** (math). Selection finalized at Phase 2 entry.

### Models (Phase 1 mandatory)
- `bert-base-uncased`

### Models (Phase 2 stretch)
- `bert-large-uncased`, `Qwen2.5-1.5B`, `Llama-3.2-3B`, optionally `Qwen2.5-7B` (QLoRA only)

### Splits protocol (non-negotiable; mirrors CLAUDE.md "Validation/Test Hygiene")

| Split | Source | Purpose | Touched by |
|---|---|---|---|
| `train` | task's training set | fit individual adapters | training only |
| `val_selection` | 40% of task's eval set, fixed split-seed=0 | choose ensemble members, soup subset, best_single | `ensemble_eval.py` only |
| `val_combine` | 20% of task's eval set | tune combination-method hyperparameters (e.g., temperature, soup weights) | `ensemble_eval.py` only |
| `test` | remaining 40% | final reported metrics | reporting only |
| OOD | held-out external sets (ANLI, HANS, Amazon CF, MMLU subdomain split) | OOD generalization & calibration | reporting only |

The 40/20/40 split is implemented in `ensemble_eval.py:split_indices()` with `--split-seed 0` (fixed). Any change requires an amendment.

The `oracle_single` method in `ensemble_eval.py` uses test-set selection and is **for sanity checks only**, never for paper claims.

---

## 3. Pool Definitions

Locked at Phase 1 entry. Pool sizes are **per-cell** (per task × model).

### Pool-A — Seed-only (purpose: isolate seed variance)
- Method: LoRA, fixed `rank=8, alpha=16, lr=3e-4, dropout=0.0, target=q+v`
- Vary: `seed ∈ {11, 21, 31, …, 201}` (20 adapters)
- Sweep config: `sweep_configs/pool_a_*.json`

### Pool-B — Hyperparameter-diverse (purpose: HP diversity vs seed diversity)
- Method: LoRA, one fixed seed per cell
- Vary across {`rank ∈ {4,8,16,32}`, `alpha ∈ {8,16,32}`, `lr ∈ {1e-4,3e-4,1e-3}`, `target ∈ {q+v, q+k+v, all-attn}`}; sample ~24 configs
- Sweep config: `sweep_configs/pool_b_*.json`

### Pool-C — Method-mixed (purpose: cross-method diversity)
- 5 methods × 4 seeds = 20 adapters: LoRA, DoRA, rsLoRA, PiSSA, AdaLoRA
- Fixed rank/alpha at each method's typical setting
- Sweep config: `sweep_configs/pool_c_*.json`

### Phase 2 — Pool size scaling
- For at least one (task, model) cell: replicate Pool-A at sizes {4, 8, 16, 32, 64} to produce a *scaling curve* of ensemble gain vs N. Same fixed-HP recipe as Pool-A; just longer seed range.

### Stretch / explicitly allowed extensions
- MoLE / router-based combination (Phase 2)
- Larger models via QLoRA (Phase 2)
- Distillation (`distill_pool.py`, Phase 3)

---

## 4. Combination Methods

Locked. Implemented in `ensemble_eval.py` (prediction-space) and `weight_space_merge.py` (parameter-space).

| Method | Space | Selection |
|---|---|---|
| `majority_vote` | prediction | none |
| `soft_vote` | prediction | none |
| `logit_avg` | prediction | none |
| `uniform_soup` (prediction-space alias of `logit_avg`) | prediction | none |
| `greedy_soup` | prediction | greedy on `val_selection` |
| `best_single` | prediction (singleton) | argmax on `val_selection` |
| `weight_space_merge --methods uniform` | parameter | none |
| `weight_space_merge --methods ties` | parameter | TIES (Yadav et al. 2023) |
| `weight_space_merge --methods dare` | parameter | DARE (Yu et al. 2024) |

`oracle_single` is implemented for sanity only (see §2 caveat).

Phase 2 may add **MoLE/router**; addition logged as an amendment but pre-allowed here.

---

## 5. Baselines (Compute-Matched)

Every ensemble claim is reported through `analysis/compute_match.py` against **all four** baseline kinds where applicable:

| Kind | Definition | Notes |
|---|---|---|
| `best_of_n` | Best individual adapter from the same pool, selected on `val_selection` | "free" — same pool |
| `n_rank` | Single LoRA with rank `N × base_rank` | separate sweep |
| `n_steps` | Single LoRA with `N × training_steps` | separate sweep |
| `n_data` | Single LoRA with `N × training_data` (if available) | separate sweep; skipped when at full data |

`baseline_compute_matched_mnli_bert.json` already exists in `sweep_configs/` and will be replicated for SST-2 and Phase 2 tasks.

---

## 6. Evaluation Metrics (Locked)

Every result JSON from `ensemble_eval.py` carries:

**Accuracy arm (claim A)**
- `accuracy_test` (primary metric; for MNLI/SST-2 = exact match accuracy)
- per-adapter `individual_accuracy_test`
- per-method `predictions_test` (for paired statistics)

**Calibration arm (claim C)**
- `ece_15bin_equalmass` (primary calibration metric, equal-mass binning)
- `mce`, `brier`, `nll`
- `reliability_bins` (per-bin {n, mean_confidence, mean_accuracy})
- per-adapter `individual_calibration_test`

**OOD reliability (Phase 2 mandatory; Phase 1 optional)**
- Per-method `ood_scores`: max-softmax-prob (`msp`) and energy arrays
- AUROC computed downstream when an OOD set is paired with the ID set

**Inference cost (every claim)**
- `n_adapters` actually used (for greedy_soup this is `chosen_count`)
- Estimated relative latency multiplier vs single-adapter inference

---

## 7. Statistical Protocol

All comparisons must go through `analysis/compute_match.py`. No paper-claim statistic is computed ad hoc.

- **CI**: Percentile bootstrap, **B = 5000**, paired over evaluation examples, alpha = 0.05
- **Permutation test**: Two-sided sign-flip, n_perm = 10000 (Monte Carlo); exact when N ≤ 20
- **Multi-cell correction**: Holm step-down across every cell reported in a single table, applied via `compute_match.holm_correct_batch()`
- **Bootstrap over pools** (Phase 2+): when scaling-curve cells are reported, additionally bootstrap over pool samples (re-draw adapters with replacement) to express within-pool sampling uncertainty
- **Statistical backend**: `analysis/compute_match.py` falls back to its local implementation if `project_a_pairwise_summary.py` interface drifts; the choice is recorded in each result JSON's `stats_backend` field

---

## 8. Adjudication

Per-cell adjudication uses the corrected CI on `accuracy_diff` (and on `ece_diff` for the C arm):

| Outcome | Rule | Reporting |
|---|---|---|
| `supported` | CI lower bound > 0 | bold in table; counts toward arm support |
| `unsupported` | CI contains 0 | reported, no claim |
| `reversed` | CI upper bound < 0 | reported as a documented anti-finding |

The arm-A and arm-C adjudications are **kept separate** in tables. A cell with `supported` on C and `unsupported` on A is not a failure — it directly evidences the calibration angle of the central claim.

---

## 9. Frontier Regression (Pre-Specified)

The frontier is the central object of the paper. Pre-specified setup:

- **Diversity features** (independent variables): pairwise prediction disagreement, error-overlap Q-statistic, mean logit Pearson correlation, mean weight-cosine distance, hyperparameter Shannon entropy. Computed by `diversity_metrics.py`.
- **Targets** (dependent variables): `Δacc` and `Δece` per cell against the four baseline kinds.
- **Model class** (Phase 1): linear regression with feature standardization. Lasso for feature importance. Phase 2 adds a non-linear comparison (gradient-boosted regressor) but the linear model remains the primary report.
- **Validation**: leave-one-pool-out cross-validation. The R² used for §10 success/kill is the LOPO-CV R², **not** the in-sample R².
- **Sample-size note**: with only 3 pool types × 2 tasks × 1 model in Phase 1, the regression has 6 cells. We pre-acknowledge this is underpowered for arm decisions; Phase 1 R² is the *direction-of-evidence* signal that guides Phase 2 scale-up.

---

## 10. Phase Gates and Kill Criteria

Mirrors the gate structure in `CLAUDE.md` § "Phase Gates". This document is the source of truth.

### Phase 1 → Phase 2 (after Pool-A/B/C × MNLI/SST-2 × BERT-base, ≥6 cells)

- **GO to Phase 2**: LOPO-CV R² ≥ 0.5 on **at least one of** {Δacc, Δece} for ≥ 2 of 3 pool types
- **PIVOT**: LOPO-CV R² < 0.3 on **both** arms across **all** pool types → re-select claim (re-enter Phase 0 §1 with one of the rejected candidates B or D from the original roadmap), or reframe as a negative-result + frontier-characterization paper
- **EXTEND PILOT**: anything in between → run Phase 1 on one additional model (BERT-large or Qwen2.5-1.5B) and re-evaluate before deciding

### Phase 2 → Phase 3 (after scale-up)
- **GO to Phase 3**: scaling curves (pool size {4,8,16,32,64}) show monotone or saturating Δacc / Δece on at least 2 task families and the LOPO-CV R² holds in the larger sample
- **STOP at Phase 2**: write the paper as observation-only (no mechanism) if R² collapses with more data — still publishable as an empirical resource paper

### Cross-phase invariants
- No paper claim without a `compute_match.py` output JSON in the artifact bundle
- Negative results (`unsupported`, `reversed`) are kept and reported, not deleted
- All pre-spec changes go to `paper/prereg_amendments.md` with date and rationale

---

## 11. Out of Scope (deliberate)

- Pretraining-data ensembles (only PEFT adapters of a frozen base model)
- Multi-task pools (each pool is single-task in this paper; multi-task is future work)
- Distillation as a *primary* result (Phase 3 only)
- Real-time online ensembling / adaptive routing in production

These are listed so a reviewer asking "why not?" gets a direct answer in the limitations section.

---

## 12. Reproducibility Bundle (released with paper)

- `pools/*.json` — schema v1, validated by `scripts/validate_pool.py`
- All `ensemble_results/*.json` and `analysis/*.json` carry the `provenance` block from `peft_utils.provenance()`
- `sweep_configs/*.json` for every cell that appears in the paper
- This pre-registration plus `paper/prereg_amendments.md`
- Code at the git SHA recorded in each result file's `provenance.git_sha`
- README in the artifact bundle pointing reviewers at `analysis/compute_match.py` as the canonical comparison harness

---

## 13. Author / AI Disclosure

Per EMNLP policy, the use of AI assistance (Claude Code) for code scaffolding, statistical implementation, and pre-registration drafting will be disclosed in the paper. AI was *not* used to generate the central claim, the empirical results, or to interpret findings.
