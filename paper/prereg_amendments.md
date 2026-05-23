# Pre-registration amendments

The original `prereg.md` was locked at git SHA `02a5a6bc` before any pool was
trained. This file records all subsequent deviations and additions, with date,
rationale, and effect on adjudication. The benchmark v1.0 release uses only
v1.0 amendments; v1.1 adds new task families.

---

## 2026-04-29 — v1.1: HellaSwag and GSM8K task-family extension

**Change.** Add two new task families to the benchmark suite:

1. **HellaSwag** (4-way multi-choice commonsense). Trainer:
   `real_lora_multichoice.py` (newly added; uses
   `AutoModelForMultipleChoice` with PEFT `task_type=SEQ_CLS` for hook
   compatibility). Metric: accuracy via argmax over the 4 candidate
   logits. Soft-vote and logit-avg ensemble methods average per-candidate
   logits across pool members; greedy soup picks the candidate-logit
   mixing weights. Calibration metrics (ECE/Brier/NLL/MCE) are computed
   over the 4-way softmax distribution.
2. **GSM8K** (free-form arithmetic, exact-match). Trainer:
   `real_lora_train.py --eval-metric gsm8k_exact_match` (already
   present; smoke-tested). Metric: exact-match on the integer answer
   extracted by the existing `_extract_gsm8k_answer` regex. Ensemble:
   majority-vote over the 20 seed predictions per example. ECE on
   GSM8K is reported on the per-token confidence of the first generated
   answer token (cf. published GSM8K calibration practice).

**Rationale.** Reviewers of the EMNLP draft noted that the in-domain
6-task set is classification-heavy. HellaSwag tests 4-way multi-choice
(more candidates → diversity matters more); GSM8K tests free-form
generation (the chain-of-thought makes ensembling non-trivial because
predictions must be aggregated over text, not logits). Including both
output spaces tests the diversity-quality frontier on a strictly
broader domain.

**Pools.** Five Pool-A sweeps, 20 seeds each, fixed-HP:

| pool | model | task | n_seeds | seq_len | epochs |
|--|--|--|--|--|--|
| pool_a_hellaswag_bert | bert-base-uncased | HellaSwag | 20 | 128 | 3 |
| pool_a_hellaswag_roberta | roberta-base | HellaSwag | 20 | 128 | 3 |
| pool_a_hellaswag_deberta | microsoft/deberta-v3-base | HellaSwag | 20 | 128 | 3 |
| pool_a_gsm8k_qwen25_05b | Qwen/Qwen2.5-0.5B | GSM8K | 20 | 384 | 3 |
| pool_a_gsm8k_qwen25_15b | Qwen/Qwen2.5-1.5B | GSM8K | 20 | 384 | 3 |

**Adjudication.** Same 40/20/40 split protocol; same paired bootstrap
$B=5000$; same Holm correction. Compute-matched baseline (`n_rank` with
$N$ adapters worth of training compute) trained per task-model pair.

**Effect on existing v1.0 cells.** None. v1.1 amendments do not modify
any v1.0 pool, baseline, or adjudication cell. v1.1 cells are reported
in a separate table (Section "v1.1 extensions").

---

## 2026-04-29 — v1.1: vLLM multi-LoRA serving benchmark

**Change.** Section 5 ("Inference economics") gains a vLLM 0.10
multi-LoRA serving measurement on top of the sequential PyTorch
baseline. Three engineering blockers documented in `phase8_status.md`:
transformers-5.x removed `all_special_tokens_extended` (patched in
vLLM source); CUDA must be initialized via `spawn`; vLLM rejects PEFT
adapters with `modules_to_save` set (classification heads). The third
constraint required `analysis/strip_modules_to_save.py` to produce
serving-compatible adapter copies.

**Effect on adjudication.** None — vLLM serving is reported as
infrastructure cost; it does not change accuracy or ECE numbers.

---

## 2026-04-29 — v1.1: information-theoretic mechanism analysis

**Change.** Add `analysis/info_mechanism.py` and Figures 4--5 reporting
$H(c | \mathrm{conf})$ and $I(c; \mathrm{conf})$ per pool family. This
is *additional* mechanism evidence; the existing
prediction-space-vs-weight-space ablation remains the primary
mechanism analysis.

**Effect on adjudication.** None — this is mechanism interpretation;
it does not change any cell verdict.
