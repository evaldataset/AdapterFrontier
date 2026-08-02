#!/usr/bin/env python3
"""GSM8K scale curve across decoder family (v1.2 generalization of
plot_gsm8k_scale_doubling.py). Handles arbitrary scale points and an
optional out-of-family overlay (e.g., Llama-3.1-8B-Instruct).

Inputs:
    --pools  N ensemble_results JSON paths in scale-ascending order
    --labels N human-readable labels (same order)
    --overlay PATH LABEL  (optional, plotted as separate marker shape)
    --out  output PDF path

Each pool JSON must carry `per_token_logprob_calibration` with
`per_adapter_correct`, `per_adapter_confidence`, `ensemble_confidence_test`.
"""
from __future__ import annotations
import argparse, json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

np.random.seed(0)
B = 5000


def ece_15bin(c, r):
    order = np.argsort(c)
    bins = np.array_split(order, 15)
    e = 0.0
    for b in bins:
        if len(b) == 0:
            continue
        e += (len(b) / len(c)) * abs(c[b].mean() - r[b].mean())
    return e


def stats_for_pool(path: str) -> dict:
    e = json.loads(Path(path).read_text())
    cal = e["per_token_logprob_calibration"]
    pa_corr = np.array(cal["per_adapter_correct"])
    pa_conf = np.array(cal["per_adapter_confidence"])
    M, Nq = pa_corr.shape
    bs_idx = e["best_single_idx"]
    bs_corr, bs_conf = pa_corr[bs_idx], pa_conf[bs_idx]
    ens_corr = np.array(
        [1 if p == g and g != "" else 0
         for p, g in zip(e["majority_predictions"], e["golds"])]
    )
    ens_conf = np.array(cal["ensemble_confidence_test"])

    boot_acc, boot_ece = [], []
    for _ in range(B):
        idx = np.random.randint(0, Nq, Nq)
        boot_acc.append(ens_corr[idx].mean() - bs_corr[idx].mean())
        boot_ece.append(
            ece_15bin(ens_conf[idx], ens_corr[idx])
            - ece_15bin(bs_conf[idx], bs_corr[idx])
        )
    return {
        "best_acc": float(bs_corr.mean()),
        "best_ece": float(ece_15bin(bs_conf, bs_corr)),
        "ens_acc": float(ens_corr.mean()),
        "ens_ece": float(ece_15bin(ens_conf, ens_corr)),
        "Nq": Nq,
        "diff_acc": float(np.mean(boot_acc)),
        "acc_ci": (
            float(np.percentile(boot_acc, 2.5)),
            float(np.percentile(boot_acc, 97.5)),
        ),
        "diff_ece": float(np.mean(boot_ece)),
        "ece_ci": (
            float(np.percentile(boot_ece, 2.5)),
            float(np.percentile(boot_ece, 97.5)),
        ),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pools", nargs="+", required=True)
    ap.add_argument("--labels", nargs="+", required=True)
    ap.add_argument("--overlay", nargs="+", default=None,
                    help="Pairs of PATH LABEL for overlay points (can be repeated; e.g. P1 L1 P2 L2)")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    assert len(args.pools) == len(args.labels), "pools/labels must match length"

    pts = [stats_for_pool(p) for p in args.pools]

    # Parse overlay as list of (stats, label) pairs (--overlay P1 L1 P2 L2 ...)
    overlays = []
    if args.overlay:
        if len(args.overlay) % 2 != 0:
            raise SystemExit("--overlay must be PATH LABEL pairs (even count)")
        for i in range(0, len(args.overlay), 2):
            overlays.append((stats_for_pool(args.overlay[i]), args.overlay[i + 1]))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 3.6))
    x = np.arange(len(args.labels))
    w = 0.36

    best_accs = [p["best_acc"] for p in pts]
    ens_accs = [p["ens_acc"] for p in pts]
    ax1.bar(x - w / 2, best_accs, w, label="best_single", color="#888")
    ax1.bar(x + w / 2, ens_accs, w, label="majority_vote ens.", color="#2ca02c")
    for i, p in enumerate(pts):
        lo, hi = p["acc_ci"]
        ax1.text(i + w / 2, p["ens_acc"] + 0.01,
                 f"{p['diff_acc']*100:+.1f}pp", ha="center", va="bottom", fontsize=7)

    # Overlays (each in distinct hatch/color)
    overlay_colors = ["#1f77b4", "#d62728", "#9467bd", "#8c564b"]
    for j, (op, olab) in enumerate(overlays):
        xpos = len(x) + 0.5 + j
        ax1.bar([xpos - w / 2], [op["best_acc"]], w, color="#ccc", hatch="//")
        ax1.bar([xpos + w / 2], [op["ens_acc"]], w,
                color=overlay_colors[j % len(overlay_colors)], hatch="//")
        ax1.text(xpos + w / 2, op["ens_acc"] + 0.01,
                 f"{op['diff_acc']*100:+.1f}pp", ha="center", va="bottom", fontsize=7)

    all_labels = list(args.labels) + [lab for _, lab in overlays]
    all_x = list(x) + [len(x) + 0.5 + j for j in range(len(overlays))]
    ax1.set_xticks(all_x)
    ax1.set_xticklabels(all_labels, rotation=25, ha="right", fontsize=8)
    ax1.set_ylabel("GSM8K exact-match")
    all_ens_acc = ens_accs + [op["ens_acc"] for op, _ in overlays]
    ax1.set_ylim(0, max(0.95, max(all_ens_acc) + 0.1))
    ax1.set_title("Accuracy: decoder scale curve (Qwen) + cross-family overlay")
    ax1.legend(fontsize=8, loc="upper left")

    best_eces = [p["best_ece"] for p in pts]
    ens_eces = [p["ens_ece"] for p in pts]
    ax2.bar(x - w / 2, best_eces, w, label="best_single", color="#888")
    ax2.bar(x + w / 2, ens_eces, w, label="majority_vote ens.", color="#2ca02c")
    for i, p in enumerate(pts):
        ax2.text(i + w / 2, p["ens_ece"] + 0.005,
                 f"{p['diff_ece']:+.3f}", ha="center", va="bottom", fontsize=7)
    for j, (op, olab) in enumerate(overlays):
        xpos = len(x) + 0.5 + j
        ax2.bar([xpos - w / 2], [op["best_ece"]], w, color="#ccc", hatch="//")
        ax2.bar([xpos + w / 2], [op["ens_ece"]], w,
                color=overlay_colors[j % len(overlay_colors)], hatch="//")
        ax2.text(xpos + w / 2, op["ens_ece"] + 0.005,
                 f"{op['diff_ece']:+.3f}", ha="center", va="bottom", fontsize=7)
    ax2.set_xticks(all_x)
    ax2.set_xticklabels(all_labels, rotation=25, ha="right", fontsize=8)
    ax2.set_ylabel("per-token logprob ECE (lower is better)")
    all_ens_ece = ens_eces + [op["ens_ece"] for op, _ in overlays]
    ax2.set_ylim(0, max(0.6, max(all_ens_ece) + 0.05))
    ax2.set_title("Calibration: ECE delta across decoder scale")
    ax2.legend(fontsize=8, loc="upper left")

    plt.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out, bbox_inches="tight")
    plt.savefig(str(args.out).replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    print(f"wrote {args.out}")
    for lab, p in zip(args.labels, pts):
        print(f"  {lab:<22s}  best={p['best_acc']:.3f} ens={p['ens_acc']:.3f}  "
              f"dacc={p['diff_acc']*100:+.1f}pp  dece={p['diff_ece']:+.3f}")
    for op, olab in overlays:
        print(f"  {olab:<22s}  best={op['best_acc']:.3f} ens={op['ens_acc']:.3f}  "
              f"dacc={op['diff_acc']*100:+.1f}pp  dece={op['diff_ece']:+.3f}")


if __name__ == "__main__":
    main()
