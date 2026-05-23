#!/usr/bin/env python3
"""Generate the verdict-landscape figure for the EMNLP paper.

Reads all analysis/*_cm_*.json compute_match cells, parses (pool, method,
baseline, metric, adjudication), and renders a (pool × method-and-metric)
heatmap colored by adjudication: green=supported, gray=unsupported, red=reversed.
"""
from __future__ import annotations
import json, glob, re
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

PAT = re.compile(r"^(?P<pool>.+?)_cm_(?P<method>[a-z_]+?)_vs_(?P<kind>best_of_n|n_rank|n_steps|n_data)(?P<ece>_ECE)?\.json$")

# Pool ordering: group by family, then by task
def pool_sort_key(pool: str) -> tuple:
    fam_order = ['bert', 'roberta', 'deberta', 'qwen25', 'tinyllama', 'smollm', 'pythia']
    task_order = ['mnli', 'sst2', 'boolq', 'anli', 'agnews', 'qnli', 'hellaswag', 'gsm8k']
    fam = next((i for i, f in enumerate(fam_order) if f in pool), 99)
    task = next((i for i, t in enumerate(task_order) if t in pool), 99)
    pool_type = 0 if 'pool_a' in pool else (1 if 'pool_b' in pool else (2 if 'pool_c' in pool else 9))
    return (task, fam, pool_type, pool)


def main():
    cells = sorted(glob.glob('analysis/*_cm_*.json'))
    rows = defaultdict(dict)  # rows[pool][col] = adjudication code
    pools_set = set()
    cols_set = set()
    for f in cells:
        m = PAT.match(Path(f).name)
        if not m:
            continue
        pool = m.group('pool')
        if pool.startswith('baseline_'):
            continue
        method = m.group('method')
        kind = m.group('kind')
        metric = 'ece' if m.group('ece') else 'acc'
        col = f"{method}|{kind}|{metric}"
        try:
            c = json.loads(Path(f).read_text())
        except Exception:
            continue
        adj = c.get('adjudication', 'unknown')
        rows[pool][col] = adj
        pools_set.add(pool)
        cols_set.add(col)

    pools = sorted(pools_set, key=pool_sort_key)
    # Column ordering: method order, kind=best_of_n first, then acc/ece
    method_order = ['soft_vote', 'logit_avg', 'majority_vote', 'uniform_soup', 'greedy_soup']
    def col_key(col: str):
        method, kind, metric = col.split('|')
        m_idx = method_order.index(method) if method in method_order else 99
        k_idx = 0 if kind == 'best_of_n' else 1
        met_idx = 0 if metric == 'acc' else 1
        return (k_idx, m_idx, met_idx)
    cols = sorted(cols_set, key=col_key)

    code_map = {'supported': 1, 'unsupported': 0, 'reversed': -1, 'unknown': np.nan}
    M = np.full((len(pools), len(cols)), np.nan)
    for i, p in enumerate(pools):
        for j, c in enumerate(cols):
            adj = rows[p].get(c)
            if adj is not None:
                M[i, j] = code_map.get(adj, np.nan)

    print(f'pools={len(pools)} cols={len(cols)} cells={int(np.isfinite(M).sum())}')

    # 3-color discrete map: red (reversed), gray (unsupported), green (supported)
    cmap = plt.matplotlib.colors.ListedColormap(['#d62728', '#bbbbbb', '#2ca02c'])
    bounds = [-1.5, -0.5, 0.5, 1.5]
    norm = plt.matplotlib.colors.BoundaryNorm(bounds, cmap.N)

    # Bigger figure to give text room; taller row pitch so y-labels read clearly
    fig, ax = plt.subplots(figsize=(13, max(10, len(pools) * 0.20)))
    ax.imshow(M, aspect='auto', cmap=cmap, norm=norm, interpolation='nearest')
    # NaN cells: light cream
    nan_mask = ~np.isfinite(M)
    Y, X = np.where(nan_mask)
    ax.scatter(X, Y, marker='.', color='#fafafa', s=18, zorder=2)

    # Pool labels (truncated)
    short_pools = [p.replace('pool_a_', 'A·').replace('pool_b_', 'B·').replace('pool_c_', 'C·')[:36] for p in pools]
    ax.set_yticks(range(len(pools)))
    ax.set_yticklabels(short_pools, fontsize=10)
    # Column labels: short
    short_cols = []
    for c in cols:
        method, kind, metric = c.split('|')
        m_short = {'soft_vote':'sv','logit_avg':'la','majority_vote':'mv','uniform_soup':'us','greedy_soup':'gs'}.get(method, method[:2])
        short_cols.append(f"{m_short}·{kind[:3]}·{metric[:3]}")
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(short_cols, rotation=75, ha='right', fontsize=11)
    ax.tick_params(axis='both', which='major', labelsize=10)

    ax.set_title(f'Verdict landscape: {int(np.isfinite(M).sum())} adjudicated cells across {len(pools)} pools\n'
                 'green=supported, gray=unsupported, red=reversed (white=cell not generated)', fontsize=14)
    legend_elems = [
        mpatches.Patch(color='#2ca02c', label='supported (CI > 0)'),
        mpatches.Patch(color='#bbbbbb', label='unsupported (CI ⊃ 0)'),
        mpatches.Patch(color='#d62728', label='reversed (CI < 0)'),
    ]
    ax.legend(handles=legend_elems, loc='lower center', bbox_to_anchor=(0.5, 1.02),
              ncol=3, fontsize=12, frameon=False)
    plt.tight_layout()
    out_pdf = Path('paper/figures/fig_verdict_landscape.pdf')
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_pdf, bbox_inches='tight')
    plt.savefig(str(out_pdf).replace('.pdf', '.png'), dpi=150, bbox_inches='tight')
    print(f'wrote {out_pdf}')

    # Stats summary
    flat = M[np.isfinite(M)]
    n_sup = int((flat == 1).sum())
    n_uns = int((flat == 0).sum())
    n_rev = int((flat == -1).sum())
    print(f'supported={n_sup} ({100*n_sup/len(flat):.1f}%)')
    print(f'unsupported={n_uns} ({100*n_uns/len(flat):.1f}%)')
    print(f'reversed={n_rev} ({100*n_rev/len(flat):.1f}%)')


if __name__ == '__main__':
    main()
