#!/usr/bin/env python3
"""GSM8K Qwen scale-doubling: bar plot of best_single vs majority_vote
ensemble accuracy and per-token logprob ECE, with paired bootstrap CIs.
Visualizes Finding (i): decoder scale roughly doubles ensemble effect on
both arms.
"""
from __future__ import annotations
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

np.random.seed(0)
B = 5000


def _ece_15bin(c, r):
    order = np.argsort(c); bins = np.array_split(order, 15); e = 0.0
    for b in bins:
        if len(b) == 0: continue
        e += (len(b) / len(c)) * abs(c[b].mean() - r[b].mean())
    return e


def stats_for_pool(path: str) -> dict:
    e = json.loads(Path(path).read_text())
    cal = e['per_token_logprob_calibration']
    pa_corr = np.array(cal['per_adapter_correct'])
    pa_conf = np.array(cal['per_adapter_confidence'])
    M, Nq = pa_corr.shape
    bs_idx = e['best_single_idx']
    bs_corr = pa_corr[bs_idx]; bs_conf = pa_conf[bs_idx]
    ens_corr = np.array([1 if p == g and g != "" else 0 for p, g in zip(e['majority_predictions'], e['golds'])])
    ens_conf = np.array(cal['ensemble_confidence_test'])
    out = {
        'best_acc': float(bs_corr.mean()),
        'best_ece': float(_ece_15bin(bs_conf, bs_corr)),
        'ens_acc': float(ens_corr.mean()),
        'ens_ece': float(_ece_15bin(ens_conf, ens_corr)),
        'Nq': Nq,
    }
    # bootstrap CIs on ens-best diffs
    boot_acc, boot_ece = [], []
    for _ in range(B):
        idx = np.random.randint(0, Nq, Nq)
        boot_acc.append(ens_corr[idx].mean() - bs_corr[idx].mean())
        boot_ece.append(_ece_15bin(ens_conf[idx], ens_corr[idx]) - _ece_15bin(bs_conf[idx], bs_corr[idx]))
    out['diff_acc'] = float(np.mean(boot_acc))
    out['acc_ci'] = (float(np.percentile(boot_acc, 2.5)), float(np.percentile(boot_acc, 97.5)))
    out['diff_ece'] = float(np.mean(boot_ece))
    out['ece_ci'] = (float(np.percentile(boot_ece, 2.5)), float(np.percentile(boot_ece, 97.5)))
    return out


pool_05b = stats_for_pool('ensemble_results/pool_a_gsm8k_qwen25_05b.json')
pool_15b = stats_for_pool('ensemble_results/pool_a_gsm8k_qwen25_15b.json')

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3.4))

x = np.arange(2); w = 0.36
labels = ['Qwen-0.5B', 'Qwen-1.5B']
best_accs = [pool_05b['best_acc'], pool_15b['best_acc']]
ens_accs = [pool_05b['ens_acc'], pool_15b['ens_acc']]
ax1.bar(x - w/2, best_accs, w, label='best_single', color='#888')
ax1.bar(x + w/2, ens_accs, w, label='majority_vote ensemble', color='#2ca02c')
# annotate with Δ + CI
for i, p in enumerate([pool_05b, pool_15b]):
    lo, hi = p['acc_ci']
    ax1.errorbar([i + w/2], [p['ens_acc']],
                 yerr=[[p['ens_acc'] - (p['best_acc'] + lo)],
                       [(p['best_acc'] + hi) - p['ens_acc']]],
                 fmt='none', ecolor='black', capsize=3)
    ax1.text(i + w/2, p['ens_acc'] + 0.01,
             f"+{p['diff_acc']*100:.1f}pp\n[{lo*100:+.1f},{hi*100:+.1f}]",
             ha='center', va='bottom', fontsize=7)
ax1.set_xticks(x); ax1.set_xticklabels(labels)
ax1.set_ylabel('GSM8K exact-match'); ax1.set_ylim(0, 0.65)
ax1.set_title('Accuracy: ensemble effect doubles with scale'); ax1.legend(fontsize=8, loc='upper left')

best_eces = [pool_05b['best_ece'], pool_15b['best_ece']]
ens_eces = [pool_05b['ens_ece'], pool_15b['ens_ece']]
ax2.bar(x - w/2, best_eces, w, label='best_single', color='#888')
ax2.bar(x + w/2, ens_eces, w, label='majority_vote ensemble', color='#2ca02c')
for i, p in enumerate([pool_05b, pool_15b]):
    lo, hi = p['ece_ci']
    ax2.text(i + w/2, p['ens_ece'] + 0.01,
             f"{p['diff_ece']:+.3f}\n[{lo:+.3f},{hi:+.3f}]",
             ha='center', va='bottom', fontsize=7)
ax2.set_xticks(x); ax2.set_xticklabels(labels)
ax2.set_ylabel('per-token logprob ECE (lower is better)'); ax2.set_ylim(0, 0.7)
ax2.set_title('Calibration: ECE drop doubles with scale'); ax2.legend(fontsize=8, loc='upper right')

plt.tight_layout()
out_pdf = Path('paper/figures/fig_gsm8k_scale_doubling.pdf')
out_pdf.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(out_pdf, bbox_inches='tight')
plt.savefig(str(out_pdf).replace('.pdf','.png'), dpi=150, bbox_inches='tight')
print(f'wrote {out_pdf}')
print(f"  0.5B Δacc {pool_05b['diff_acc']*100:+.1f}pp CI {pool_05b['acc_ci']}")
print(f"  1.5B Δacc {pool_15b['diff_acc']*100:+.1f}pp CI {pool_15b['acc_ci']}")
print(f"  0.5B Δece {pool_05b['diff_ece']:+.3f} CI {pool_05b['ece_ci']}")
print(f"  1.5B Δece {pool_15b['diff_ece']:+.3f} CI {pool_15b['ece_ci']}")
