#!/usr/bin/env python3
"""Two-regime visualization: per-pool min/median val_acc vs ensemble effect.
Shows that pools where most adapters fail to converge (low min val_acc) require
greedy_soup to escape the soft_vote collapse.
"""
from __future__ import annotations
import json, glob, re
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

PAT = re.compile(r"^(?P<pool>.+?)_diversity\.json$")

# Known v1.1 HellaSwag pools (skip GSM8K since multi-choice eval differs)
HELLASWAG_POOLS = [
    'pool_a_hellaswag_bert', 'pool_a_hellaswag_bertlarge',
    'pool_a_hellaswag_roberta', 'pool_a_hellaswag_deberta',
    'pool_b_hellaswag_bert', 'pool_b_hellaswag_roberta', 'pool_b_hellaswag_deberta',
    'pool_c_hellaswag_bert', 'pool_c_hellaswag_roberta', 'pool_c_hellaswag_deberta',
]

POOL_TYPE = lambda p: 'A' if 'pool_a' in p else ('B' if 'pool_b' in p else 'C')
def ARCH(p):
    if 'bertlarge' in p: return 'BERT-large'
    if 'roberta' in p: return 'RoBERTa'
    if 'deberta' in p: return 'DeBERTa'
    if 'bert' in p: return 'BERT-base'
    return '?'

points = []
for pool in HELLASWAG_POOLS:
    ens_path = Path(f'ensemble_results/{pool}.json')
    if not ens_path.exists(): continue
    e = json.loads(ens_path.read_text())
    va = e.get('per_adapter_val_acc') or []
    if not va: continue
    methods = e.get('methods', {})
    bs_acc = methods.get('best_single', {}).get('accuracy', 0)
    sv_acc = methods.get('soft_vote', {}).get('accuracy', 0)
    gs_acc = methods.get('greedy_soup', {}).get('accuracy', 0)
    points.append({
        'pool': pool, 'arch': ARCH(pool), 'type': POOL_TYPE(pool),
        'min_va': float(min(va)), 'med_va': float(np.median(va)), 'frac_below_chance': float(sum(1 for x in va if x < 0.30) / len(va)),
        'best': bs_acc, 'soft': sv_acc, 'greedy': gs_acc,
        'soft_minus_best': sv_acc - bs_acc,
        'greedy_minus_best': gs_acc - bs_acc,
    })

print(f'plotted {len(points)} pools')
fig, ax = plt.subplots(figsize=(7.2, 4.6))
markers = {'A': 'o', 'B': 's', 'C': '^'}
colors = {'BERT-base': '#1f77b4', 'BERT-large': '#aec7e8', 'RoBERTa': '#ff7f0e', 'DeBERTa': '#2ca02c'}
for p in points:
    # x = fraction of adapters above chance, y = greedy_soup gain over best_single
    x = 1.0 - p['frac_below_chance']
    y = p['greedy_minus_best']
    ax.scatter(x, y, c=colors.get(p['arch'], 'k'),
               marker=markers.get(p['type'], 'x'), s=70, edgecolors='k', linewidths=0.6, alpha=0.85)
    ax.annotate(f"{p['arch']}-{p['type']}", (x, y), xytext=(5, 4), textcoords='offset points', fontsize=7)

ax.axhline(0, color='gray', linewidth=0.6, linestyle='--')
ax.axvline(0.5, color='gray', linewidth=0.6, linestyle=':')
# label two regimes
ax.fill_betweenx([-0.02, 0.10], 0.0, 0.5, alpha=0.08, color='red', zorder=-1)
ax.fill_betweenx([-0.02, 0.10], 0.5, 1.05, alpha=0.08, color='green', zorder=-1)
ax.text(0.25, 0.085, 'under-converged regime\n(soft_vote collapses;\ngreedy_soup rescues)',
        ha='center', fontsize=8, color='#7a1f1f', style='italic')
ax.text(0.78, 0.005, 'converged regime\n(soft_vote works;\ngreedy_soup matches)',
        ha='center', fontsize=8, color='#1f5c1f', style='italic')

ax.set_xlim(-0.02, 1.05); ax.set_ylim(-0.025, 0.10)
ax.set_xlabel(r'fraction of pool adapters with val_acc $\geq 0.30$ (above chance)')
ax.set_ylabel('greedy_soup acc minus best_single acc')
ax.set_title('Two regimes on HellaSwag: per-adapter convergence rate predicts\nwhether subset selection beats best_single', fontsize=10)
# Legend
from matplotlib.lines import Line2D
arch_handles = [Line2D([0],[0], marker='o', color='w', markerfacecolor=c, markeredgecolor='k', label=a, markersize=8)
                for a, c in colors.items()]
type_handles = [Line2D([0],[0], marker=m, color='w', markerfacecolor='gray', markeredgecolor='k', label=f'Pool-{t}', markersize=8)
                for t, m in markers.items()]
ax.legend(handles=arch_handles + type_handles, loc='upper right', fontsize=7, ncol=2, frameon=True)
plt.tight_layout()
out_pdf = Path('paper/figures/fig_two_regime.pdf')
plt.savefig(out_pdf, bbox_inches='tight')
plt.savefig(str(out_pdf).replace('.pdf','.png'), dpi=150, bbox_inches='tight')
print(f'wrote {out_pdf}')
for p in points:
    print(f"  {p['arch']:<12s} Pool-{p['type']} above-chance={1-p['frac_below_chance']:.2f}  "
          f"best={p['best']:.3f} soft={p['soft']:.3f} greedy={p['greedy']:.3f}")
