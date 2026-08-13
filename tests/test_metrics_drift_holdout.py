"""Metrics, calibration, drift detection and holdout stability."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from churnrisk import drift, holdout
from churnrisk import metrics as mx


# --------------------------------------------------------------- metrics ---
def test_precision_at_k_reads_the_top_of_the_ranking():
    y = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    perfect = np.array([0.9, 0.8, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    assert mx.precision_at_k(y, perfect, 2) == 1.0
    assert mx.recall_at_k(y, perfect, 2) == 1.0
    assert mx.precision_at_k(y, perfect, 10) == 0.2
    assert mx.lift_at_k(y, perfect, 2) == pytest.approx(5.0)


def test_precision_at_k_clamps_to_the_population():
    y = np.array([1, 0, 0])
    assert mx.precision_at_k(y, np.array([0.9, 0.1, 0.1]), 99) == pytest.approx(1 / 3)


def test_cost_threshold_matches_the_article():
    assert mx.cost_threshold(500, 150) == pytest.approx(0.769, abs=0.001)
    assert mx.cost_threshold(1, 1) == 0.5           # symmetric costs give the default


def test_cost_threshold_moves_with_the_costs():
    """A cheaper false positive should make you act sooner."""
    assert mx.cost_threshold(100, 150) < mx.cost_threshold(500, 150)


def test_total_cost_counts_both_mistakes():
    y = np.array([1, 0, 1, 0])
    p = np.array([0.9, 0.9, 0.1, 0.1])
    # At 0.5: one false positive (500), one missed positive (150).
    assert mx.total_cost(y, p, 0.5, 500, 150) == 650
    assert mx.total_cost(y, p, 0.0, 500, 150) == 1000     # block everything: 2 FP
    assert mx.total_cost(y, p, 1.01, 500, 150) == 300     # block nothing: 2 FN


def test_perfect_calibration_scores_zero():
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 1, 60_000)
    y = (rng.random(60_000) < p).astype(int)
    assert mx.expected_calibration_error(y, p) < 0.01
    assert mx.brier_score(y, p) < 0.25


def test_a_confidently_wrong_model_scores_badly():
    y = np.zeros(1000, dtype=int)
    p = np.full(1000, 0.9)
    assert mx.expected_calibration_error(y, p) == pytest.approx(0.9, abs=0.01)
    assert mx.brier_score(y, p) == pytest.approx(0.81, abs=0.01)


def test_quantile_reliability_bins_survive_imbalance():
    """Equal-width bins put every row in bin zero at fraud-level imbalance."""
    rng = np.random.default_rng(1)
    p = np.concatenate([rng.uniform(0, 0.01, 9_900), rng.uniform(0.5, 1.0, 100)])
    y = (rng.random(10_000) < p).astype(int)

    predicted, _, counts = mx.reliability_curve(y, p, n_bins=10, strategy="quantile")
    assert len(predicted) >= 5                    # several usable bins
    uniform, _, _ = mx.reliability_curve(y, p, n_bins=10, strategy="uniform")
    assert len(uniform) < len(predicted)          # equal-width collapses


def test_headline_survives_a_single_class():
    y = np.zeros(50, dtype=int)
    head = mx.headline(y, np.full(50, 0.1))
    assert np.isnan(head["pr_auc"]) and np.isnan(head["roc_auc"])
    assert head["brier"] == pytest.approx(0.01)


# ----------------------------------------------------------------- drift ---
def test_psi_is_zero_against_itself():
    rng = np.random.default_rng(2)
    sample = rng.normal(size=20_000)
    assert drift.psi(sample, sample) == pytest.approx(0.0, abs=1e-9)


def test_psi_grows_with_the_shift():
    rng = np.random.default_rng(2)
    reference = rng.normal(size=20_000)
    small = drift.psi(reference, rng.normal(0.1, 1.0, 20_000))
    large = drift.psi(reference, rng.normal(2.0, 1.0, 20_000))
    assert small < drift.PSI_WATCH < large
    assert large > drift.PSI_ALERT


def test_ks_statistic_matches_a_known_case():
    assert drift.ks_statistic(np.arange(100), np.arange(100)) == pytest.approx(0.0)
    # Two disjoint ranges: the CDFs separate completely.
    assert drift.ks_statistic(np.arange(100), np.arange(100, 200)) == pytest.approx(1.0)


def test_monitor_matches_the_direct_computation():
    rng = np.random.default_rng(3)
    reference = pd.DataFrame({"a": rng.normal(size=8_000), "b": rng.gamma(2, size=8_000)})
    current = pd.DataFrame({"a": rng.normal(0.6, size=4_000), "b": rng.gamma(2, size=4_000)})

    monitor = drift.Monitor(reference, ["a", "b"], max_rows=10_000)
    fast = monitor.check(current)
    slow = drift.drift_report(reference, current, ["a", "b"])
    for column in ("a", "b"):
        assert fast[column]["psi"] == pytest.approx(slow[column]["psi"], rel=1e-9)


def test_monitor_handles_a_constant_column():
    reference = pd.DataFrame({"flat": np.ones(500)})
    monitor = drift.Monitor(reference, ["flat"])
    assert monitor.check(pd.DataFrame({"flat": np.ones(100)}))["flat"]["psi"] == 0.0


def test_verdict_thresholds():
    assert drift.verdict({"x": {"psi": 0.01}}) == "stable"
    assert drift.verdict({"x": {"psi": 0.15}}) == "watch"
    assert drift.verdict({"x": {"psi": 0.90}}) == "alert"


def test_flag_rate_catches_a_small_concentrated_attack():
    """The signal population PSI cannot see: 1% of traffic, all of it at the top."""
    rng = np.random.default_rng(4)
    normal = rng.uniform(0, 0.05, 10_000)
    attacked = np.concatenate([rng.uniform(0, 0.05, 9_900), rng.uniform(0.4, 0.9, 100)])

    assert drift.psi(normal, attacked) < drift.PSI_ALERT       # invisible here
    baseline = drift.flag_rate(normal, 0.30)
    assert drift.flag_rate(attacked, 0.30) > max(baseline, 1e-6) * 5     # obvious here


# --------------------------------------------------------------- holdout ---
def test_holdout_membership_is_stable_across_orderings():
    ids = [f"A{i:05d}" for i in range(5_000)]
    first = holdout.assign(ids, 0.15)
    reshuffled = list(reversed(ids))
    second = holdout.assign(reshuffled, 0.15)
    assert (first == second[::-1]).all()


def test_holdout_share_is_close_to_requested():
    ids = [f"A{i:05d}" for i in range(20_000)]
    assert holdout.assign(ids, 0.15).mean() == pytest.approx(0.15, abs=0.01)


def test_holdout_survives_the_book_growing():
    """New accounts must not reshuffle the existing control group."""
    ids = [f"A{i:05d}" for i in range(5_000)]
    before = holdout.assign(ids, 0.15)
    after = holdout.assign(ids + [f"B{i:05d}" for i in range(2_000)], 0.15)
    assert (before == after[:5_000]).all()


def test_changing_the_salt_redraws_the_holdout():
    ids = [f"A{i:05d}" for i in range(5_000)]
    a = holdout.assign(ids, 0.15, salt="v1")
    b = holdout.assign(ids, 0.15, salt="v2")
    assert (a != b).mean() > 0.1


def test_holdout_rejects_a_nonsense_share():
    with pytest.raises(ValueError):
        holdout.assign(["a"], 1.5)
