# Adapter Populations

Empirical study of PEFT adapters as **populations** rather than isolated checkpoints. Uses a 50-node RTX 3060 cluster to train large adapter pools and characterize how **diversity, selection pressure, and combination rules** affect downstream performance.

- Pre-registration: [paper/prereg.md](paper/prereg.md)
- Amendment ledger: [paper/prereg_amendments.md](paper/prereg_amendments.md)

## Reproduction Paths

This repository supports **JSON-level reproducibility** (every headline number re-derivable from the released JSONs, no GPU). Byte-level training replay is deferred — see "Scope of reproducibility" below.

### A. Verify all headline numbers (recommended, <30 seconds, no GPU)

```bash
pip install -r requirements-verify.txt       # numpy/scipy/sklearn/matplotlib/pytest
python3 scripts/reproduce_paper_numbers.py   # 145 assertions, all PASS
python3 -m pytest tests/ -q                  # 17 tests
```

`requirements.txt` is the full training and serving stack (torch, vLLM,
transformers, peft) and is needed only for path C. The verification script
itself is pure standard library.

This re-derives every headline number in the paper from the released JSONs:
corpus counts, BH-FDR cutoff (post-FDR 29.5% SUPPORTED / 16.9% REVERSED),
the corpus-wide frontier R² (48 pools, encoder+decoder, LOPO-CV = +0.597;
encoder-only +0.616), H(c|conf) decoder gap (21%), vLLM serving ratios,
distillation recovery, the measured baseline training budget (median 1.45x
the pool's compute), the four-way multiple-comparison sensitivity, the
three protocol-sensitivity shortcuts (S1-S3), and the temperature control
on the calibration arm.

Headline: 0 of 92 clean encoder cells improve accuracy over the stronger
single adapter, and 33.7% are significantly worse. 52 of the same 92 improve
ECE — but that comparison is against a single adapter picked by validation
accuracy and never calibrated. On the six pools whose logits cache survives,
fitting one temperature on the held-out `val_combine` split takes the
ensemble from better-calibrated in 4/6 pools to 2/6 (mean ΔECE −0.032). The
ECE gain is confidence shrinkage, and a scalar reproduces it at 1x inference
cost instead of Nx.

Note: a fourth shortcut (contaminated benchmark) and the GSM8K
contamination probe behind it were **retracted on 2026-07-28** — the probe
shifted the reference answer along with the question, forcing near-zero
accuracy on the shifted split regardless of contamination. See the
RETRACTED header in `analysis/gsm8k_contamination_probe.py`.

Each per-cell `analysis/*_cm_*.json` carries explicit `bh_q_value`,
`bh_pass`, `adjudication_pre_fdr`, and `adjudication_post_fdr` fields
written by `scripts/apply_corpus_bh_fdr.py`. The corpus-wide
`bh_batch_hash` pins the cell set the correction was computed over.

### B. Regenerate figures (analysis only, no GPU)

```bash
python3 analysis/plot_protocol_sensitivity.py # Figure 1 (body)
python3 analysis/protocol_sensitivity.py      # S1-S3 shortcut numbers
python3 analysis/temperature_control.py       # recalibration control (title claim)
python3 analysis/correction_sensitivity.py    # four-way correction grid
python3 analysis/baseline_budget_audit.py     # measured baseline GPU-hours
python3 analysis/plot_frontier_scatter.py     # appendix figure
python3 analysis/plot_info_mechanism.py       # appendix figure
python3 analysis/plot_scaling_curves.py       # appendix figure
python3 analysis/plot_verdict_landscape.py    # appendix figure
python3 analysis/plot_two_regime.py           # appendix figure
python3 analysis/cost_savings.py              # Table 12 (cost of blind ensembling)

# Re-fit frontier regression with drop diagnostics + method-onehot ablation
python3 analysis/frontier.py --out analysis/frontier_with_drop_audit.json
```

### C. Cluster reproduction (full training pipeline — requires the original 50-node RTX 3060 cluster)

The orchestration scripts (`experiment_manager.py` / `sweep_runner.py` /
`cluster_controller.py`) target the specific cluster used by the authors and
will not run elsewhere. See "Quick Start" below for the cluster-side commands.

### Scope of reproducibility

- **JSON-level** (guaranteed): every paper number reproducible from released
  JSONs by `scripts/reproduce_paper_numbers.py`. Adjudication labels carry
  explicit corpus-wide BH-FDR provenance (`bh_batch_hash`).
- **Per-adapter logits cache**: retained for 6 of 99 pool manifests, and for
  none of the 23 headline pools. The rest was not preserved. This is why the
  temperature control above covers six pools rather than all of them.
- **Adapter weights**: present for 253 of 1,744 declared adapters (15%) —
  6 pools complete, 22 partial, 71 empty. "Available on request" can be
  honoured for that minority only; the remainder was not retained.
- **Byte-level training replay**: not available for this drop, rather than
  merely deferred. 1,027 of 1,028 result JSONs record `git_dirty=true` and
  every adapter records an unknown `git_sha`, so results cannot be tied back
  to the registered SHA `02a5a6bc` at the byte level.

## Quick Start

```bash
# 0. Set cluster password (required by sweep_runner and cluster_controller)
export CLUSTER_SSH_PASSWORD=...

# 1. Submit and run the first pool
python3 experiment_manager.py submit --config sweep_configs/pool_a_mnli_bert.json
python3 experiment_manager.py run --name pool_a_mnli_bert --all

# 2. Collect adapters from cluster
python3 collect_remote_adapters.py --sweep pool_a_mnli_bert --out collected_adapters/pool_a_mnli_bert/

# 3. Build pool manifest
python3 build_adapter_pool.py --sweep-dir collected_adapters/pool_a_mnli_bert/ --out pools/pool_a_mnli_bert.json

# 4. Evaluate ensemble methods
python3 ensemble_eval.py --pool pools/pool_a_mnli_bert.json --task mnli \
    --methods majority_vote,soft_vote,greedy_soup,best_single \
    --out ensemble_results/pool_a_mnli_bert.json

# 5. Diversity metrics
python3 diversity_metrics.py --pool pools/pool_a_mnli_bert.json \
    --logits-cache ensemble_cache/ --out analysis/pool_a_diversity.json

# 6. Weight-space merging baselines
python3 weight_space_merge.py --pool pools/pool_a_mnli_bert.json \
    --methods uniform,ties,dare --out-dir merged_adapters/pool_a/

# 7. Diversity-quality frontier across pools
python3 analysis/frontier.py \
    --ensemble ensemble_results/pool_a_*.json ensemble_results/pool_b_*.json \
    --diversity analysis/pool_a_diversity.json analysis/pool_b_diversity.json \
    --out analysis/frontier_mnli_bert.json
```

## Monitor Cluster

```bash
# Live dashboard with per-node GPU process names
python3 host_status_monitor.py

# Kill all GPU processes on 50 nodes (emergency cleanup)
CLUSTER_SSH_PASSWORD=$CLUSTER_SSH_PASSWORD python3 cluster_controller.py \
    --run-name kill_gpu --max-workers 20 \
    --command 'nvidia-smi --query-compute-apps=pid --format=csv,noheader | xargs -r kill -9'
```
