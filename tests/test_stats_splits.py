"""Deterministic unit tests for the statistics and split logic that every
headline number depends on.

Run: python3 -m pytest tests/ -q
"""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analysis"))
sys.path.insert(0, str(ROOT / "scripts"))

from splits import split_indices                             # noqa: E402
from compute_match import (                                   # noqa: E402
    _ece_equal_mass, _paired_bootstrap_diff_ci, _sign_flip_pvalue, holm_correct,
)
from apply_corpus_bh_fdr import bh_qvalues, bh_cutoff_p       # noqa: E402
from c1_policy_harness import c_norm, forwards_of             # noqa: E402


# ------------------------------------------------------------------ splits
class TestSplitIndices:
    def test_ratios_and_partition(self):
        sel, com, test = split_indices(1000, seed=0)
        assert len(sel) == 400 and len(com) == 200 and len(test) == 400
        allidx = np.concatenate([sel, com, test])
        assert len(np.unique(allidx)) == 1000          # disjoint + complete

    def test_deterministic_same_seed(self):
        a = split_indices(777, seed=0)
        b = split_indices(777, seed=0)
        for x, y in zip(a, b):
            assert np.array_equal(x, y)

    def test_seed_changes_split(self):
        a, _, _ = split_indices(777, seed=0)
        b, _, _ = split_indices(777, seed=1)
        assert not np.array_equal(a, b)

    def test_no_test_leak_into_selection(self):
        sel, com, test = split_indices(500, seed=0)
        assert set(sel).isdisjoint(test) and set(com).isdisjoint(test)


# ------------------------------------------------------------------ ECE
class TestECE:
    def test_perfectly_calibrated_two_bins(self):
        # bin1: conf .6 acc .6 / bin2: conf .9 acc .9  -> ECE 0
        conf = np.array([0.6] * 50 + [0.9] * 50)
        correct = np.array([1] * 30 + [0] * 20 + [1] * 45 + [0] * 5, dtype=float)
        assert _ece_equal_mass(conf, correct, n_bins=2) == pytest.approx(0.0, abs=1e-9)

    def test_fully_overconfident(self):
        conf = np.full(100, 1.0)
        correct = np.zeros(100)
        assert _ece_equal_mass(conf, correct, n_bins=15) == pytest.approx(1.0)

    def test_empty_returns_nan(self):
        assert np.isnan(_ece_equal_mass(np.array([]), np.array([])))


# ------------------------------------------------------------------ bootstrap
class TestPairedBootstrap:
    def test_known_positive_diff_ci_excludes_zero(self):
        rng = np.random.RandomState(0)
        a = (rng.rand(2000) < 0.80).astype(float)
        b = (rng.rand(2000) < 0.70).astype(float)
        obs, lo, hi = _paired_bootstrap_diff_ci(a, b, n_boot=1000, seed=0)
        assert lo > 0 and lo < obs < hi

    def test_identical_arrays_ci_is_zero(self):
        a = np.array([1.0, 0.0] * 100)
        obs, lo, hi = _paired_bootstrap_diff_ci(a, a.copy(), n_boot=500, seed=0)
        assert obs == lo == hi == 0.0


class TestSignFlip:
    def test_no_effect_large_p(self):
        a = np.array([1.0, 0.0] * 50)
        assert _sign_flip_pvalue(a, a.copy(), n_perm=500, seed=0) > 0.9

    def test_strong_effect_small_p(self):
        a = np.ones(200); b = np.zeros(200)
        assert _sign_flip_pvalue(a, b, n_perm=1000, seed=0) < 0.01


# ------------------------------------------------------------------ Holm / BH
class TestCorrections:
    def test_holm_hand_example(self):
        # p=(.01,.04,.03), m=3 -> sorted (.01,.03,.04) -> adj (.03,.06,.06)
        adj = holm_correct([0.01, 0.04, 0.03])
        assert adj[0] == pytest.approx(0.03)
        assert adj[1] == pytest.approx(0.06)
        assert adj[2] == pytest.approx(0.06)

    def test_holm_monotone_and_capped(self):
        adj = holm_correct([0.5, 0.9, 0.7])
        assert all(0 < a <= 1.0 for a in adj)

    def test_bh_qvalues_hand_example(self):
        # p=(.01,.02,.03,.04), m=4:
        # q_(k)=p*m/k -> (.04,.04,.04,.04) after step-up min
        q, _ = bh_qvalues([0.01, 0.02, 0.03, 0.04])
        assert q == pytest.approx([0.04, 0.04, 0.04, 0.04])

    def test_bh_cutoff(self):
        # q=.05, m=4: largest p_(k) <= k*.05/4
        assert bh_cutoff_p([0.01, 0.02, 0.03, 0.04], 0.05) == pytest.approx(0.04)
        assert bh_cutoff_p([0.9, 0.95], 0.05) == 0.0


# ------------------------------------------------------------------ C1 math
class TestC1Utility:
    def test_c_norm_endpoints(self):
        assert c_norm(1.0, 10) == 0.0          # single
        assert c_norm(10.0, 10) == 1.0         # full ensemble
        assert 0 < c_norm(4.0, 10) < 0.5       # subset

    def test_forwards_of(self):
        assert forwards_of("best_single", 10, {}) == 1.0
        assert forwards_of("soft_vote", 10, {}) == 10.0
        assert forwards_of("greedy_soup", 10, {"chosen_count": 3}) == 3.0
