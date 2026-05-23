# Adapter Populations

Empirical study of PEFT adapters as **populations** rather than isolated checkpoints. Uses a 50-node RTX 3060 cluster to train large adapter pools and characterize how **diversity, selection pressure, and combination rules** affect downstream performance.

- Detailed agent guidance: [CLAUDE.md](CLAUDE.md)
- 2-week MVP roadmap: [PLAN.md](PLAN.md)
- Parent project (REPEFT): `./JS` (parent path before sanitization)

## Reproduction Paths

This repository supports two reproduction modes:

### A. Cluster reproduction (full pipeline)
Requires SSH access to a 50-node RTX 3060 cluster (the original training cluster; node list in `hosts_wol.csv`). The Quick Start below covers this path. **Without cluster access, the orchestration scripts in `experiment_manager.py` / `sweep_runner.py` / `cluster_controller.py` will not work.**

### B. Non-cluster reproduction (analysis + figures only)
External users without the cluster can reproduce **all paper numbers and figures** from the released artifacts (no training required):

```bash
pip install -r requirements.txt

# Reproduce headline statistics from cm cell JSONs
python3 -c "
import json, subprocess
fns = subprocess.run(['find','.','-name','*_cm_*.json'], capture_output=True, text=True).stdout.strip().split('\n')
sup = rev = 0
for fn in fns:
    d = json.load(open(fn))
    if d.get('adjudication') == 'supported': sup += 1
    elif d.get('adjudication') == 'reversed': rev += 1
print(f'{len(fns)} cells: {100*sup/len(fns):.1f}% SUPPORTED, {100*rev/len(fns):.1f}% REVERSED')
"

# Reproduce all 6 paper figures
python3 analysis/plot_frontier_scatter.py     # Fig 4
python3 analysis/plot_info_mechanism.py       # Fig 5
python3 analysis/plot_scaling_curves.py       # Fig 6
python3 analysis/plot_verdict_landscape.py    # Fig 3
python3 analysis/plot_two_regime.py           # body Fig 2
python3 analysis/plot_gsm8k_scale_doubling.py # body Fig 1

# Re-fit frontier regression
python3 analysis/frontier.py
```

Adapter weights themselves are not released in this repository (size). Pool manifests (`pools/*.json`) reference adapter paths under `./collected_adapters/`; to re-run `ensemble_eval.py` on raw weights, download the companion HuggingFace dataset (de-anonymized at acceptance).

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
