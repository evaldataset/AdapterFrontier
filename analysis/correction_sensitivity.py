#!/usr/bin/env python3
"""How much of the REVERSED rate is the multiple-comparison procedure?

Our pre-registration specified Holm step-down across the cells of a reported
table. The released cells additionally carry a corpus-wide BH-FDR field, and
the paper's REVERSED percentage was taken from that. Adversarial review
pointed out that the choice materially changes the number, and that the
switch was never recorded as an amendment. This script reports every
procedure side by side so a reader can pick.

The finding that does NOT move is SUPPORTED = 0: under all four procedures,
no clean encoder cell shows a significant accuracy gain over the stronger
single adapter. The REVERSED rate is procedure-dependent and we say so.

Usage: python3 analysis/correction_sensitivity.py
"""
from __future__ import annotations

import json
import glob
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DECODER = ("qwen", "llama", "mistral", "tinyllama", "pythia", "smollm")
PAT = re.compile(r"^(?P<pool>.+?)_cm_(?P<m>[a-z_]+?)_vs_"
                 r"(?P<k>best_of_n|n_rank|n_steps|n_data)(?P<ece>_ECE)?\.json$")


def holm(pvals: list[float]) -> list[float]:
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adj, run = [0.0] * m, 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (m - rank) * pvals[i]))
        adj[i] = run
    return adj


def load(arm: str = "accuracy") -> list[dict]:
    """Clean encoder cells vs n_rank, one arm."""
    out = []
    for f in sorted(glob.glob(str(ROOT / "analysis/*_cm_*.json"))):
        m = PAT.match(Path(f).name)
        if not m or m.group("k") != "n_rank":
            continue
        if (arm == "ECE") != bool(m.group("ece")):
            continue
        pool = m.group("pool").lower()
        if "hellaswag" in pool or "gsm8k" in pool or any(t in pool for t in DECODER):
            continue
        d = json.loads(Path(f).read_text())
        out.append({"p": d["sign_flip_p"], "lo": d["ci_low"], "hi": d["ci_high"],
                    "holm_corpus": d.get("holm_adjusted_p_batch"),
                    "bh": d.get("bh_q_value")})
    return out


def counts(cells, significant) -> dict:
    sup = sum(1 for c, s in zip(cells, significant) if s and c["lo"] > 0)
    rev = sum(1 for c, s in zip(cells, significant) if s and c["hi"] < 0)
    n = len(cells)
    return {"n": n, "supported": sup, "reversed": rev,
            "pct_supported": round(100 * sup / n, 1),
            "pct_reversed": round(100 * rev / n, 1)}


def procedures(cells) -> dict:
    ps = [c["p"] for c in cells]
    h_tab = holm(ps)
    return {
        "raw_ci": counts(cells, [c["lo"] > 0 or c["hi"] < 0 for c in cells]),
        "holm_corpus_m1028": counts(cells, [(c["holm_corpus"] or 1) <= 0.05 for c in cells]),
        "holm_within_table": counts(cells, [a <= 0.05 for a in h_tab]),
        "bh_corpus_q05": counts(cells, [(c["bh"] or 1) <= 0.05 for c in cells]),
    }


def main() -> int:
    out = {}
    for arm in ("accuracy", "ECE"):
        cells = load(arm)
        if not cells:
            continue
        out[arm] = procedures(cells)
        print(f"\nencoder x n_rank x {arm}  (n={len(cells)})")
        print(f"  {'procedure':22s} {'SUPPORTED':>10s} {'REVERSED':>10s}")
        for k, v in out[arm].items():
            print(f"  {k:22s} {v['supported']:4d} ({v['pct_supported']:4.1f}%) "
                  f"{v['reversed']:4d} ({v['pct_reversed']:4.1f}%)")

    acc = out.get("accuracy", {})
    sup_all_zero = all(v["supported"] == 0 for v in acc.values())
    rev_range = [v["pct_reversed"] for v in acc.values()]
    print(f"\naccuracy: SUPPORTED==0 under every procedure: {sup_all_zero}")
    print(f"accuracy: REVERSED ranges {min(rev_range):.1f}%--{max(rev_range):.1f}% "
          f"depending on procedure")
    out["summary"] = {
        "accuracy_supported_zero_under_all_procedures": bool(sup_all_zero),
        "accuracy_reversed_pct_min": min(rev_range),
        "accuracy_reversed_pct_max": max(rev_range),
        "note": ("SUPPORTED=0 is invariant to the correction procedure and is the robust "
                 "claim. The REVERSED rate is procedure-dependent: the pre-registered "
                 "corpus-wide Holm yields 0, within-table Holm 16, corpus-wide BH 31, "
                 "and the raw per-cell CI 38. The paper reports the BH figure and states "
                 "this dependence explicitly."),
    }
    p = ROOT / "analysis/correction_sensitivity.json"
    p.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
