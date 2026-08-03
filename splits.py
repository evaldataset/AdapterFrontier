#!/usr/bin/env python3
"""The evaluation split, on its own.

`ensemble_eval.py` imports torch, transformers, peft and datasets at module
level, so importing it to reach this one function pulls in the whole training
stack. That made `pytest tests/` impossible in the lightweight, no-GPU
environment the README's verification path describes: the split test failed at
collection with ModuleNotFoundError: torch.

The split is the load-bearing piece of the protocol -- every selection,
combination and reported number depends on it partitioning the same way every
time -- so it lives here, importable with nothing but numpy, and
`ensemble_eval` re-exports it.
"""
from __future__ import annotations

import numpy as np


def split_indices(n: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Deterministic 40/20/40 split -> (val_selection, val_combine, test).

    val_selection picks ensemble members and soup weights; val_combine tunes
    combination hyperparameters (and the temperature in the recalibration
    control); test is touched only for the final measurement.
    """
    rng = np.random.RandomState(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_sel = int(0.4 * n)
    n_combo = int(0.2 * n)
    return idx[:n_sel], idx[n_sel:n_sel + n_combo], idx[n_sel + n_combo:]
