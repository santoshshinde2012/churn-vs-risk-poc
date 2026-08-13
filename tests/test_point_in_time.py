"""The leakage guarantees, held to.

These are the tests worth having. Every other property in this repo shows up in
a metric if it breaks; a point-in-time violation shows up as a *better* metric,
which is why it survives code review and dies in production.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from churnrisk import churn_data as cd
from churnrisk import risk_data as rd


# --------------------------------------------------------------- risk ------
@pytest.fixture(scope="module")
def stream():
    return rd.generate_stream(n_accounts=60, days=40, drift_day=32, seed=3)


def test_risk_features_ignore_the_future(stream):
    """Truncating the stream must not change any row that survives.

    This is the whole point-in-time claim: a row's features depend on the rows
    before it and on nothing else. If a later event could reach back, the two
    builds would disagree.
    """
    full = rd.build_features(stream)
    half = len(stream) // 2
    truncated = rd.build_features(stream.iloc[:half].copy())

    shared = rd.FEATURES
    pd.testing.assert_frame_equal(
        full.iloc[:half][shared].reset_index(drop=True),
        truncated[shared].reset_index(drop=True),
        check_exact=False, rtol=1e-12,
    )


def test_risk_features_require_sorted_stream(stream):
    """An unsorted stream is a bug, not something to silently tolerate."""
    shuffled = stream.sample(frac=1.0, random_state=1)
    with pytest.raises(ValueError, match="sorted"):
        rd.build_features(shuffled)


def test_first_event_for_an_account_has_no_history(stream):
    """The very first transaction on an account cannot know anything about it."""
    features = rd.build_features(stream)
    first = features.groupby("account").head(1)
    assert (first["txn_60s"] == 0).all()
    assert (first["txn_10min"] == 0).all()
    assert (first["txn_24h"] == 0).all()
    assert (first["new_device"] == 1).all()
    assert np.allclose(first["amount_ratio"], 1.0)


def test_network_features_count_only_prior_accounts(stream):
    """device_accounts is a count of accounts seen *before* this row."""
    features = rd.build_features(stream)
    merged = features.join(stream[["device"]])
    first_per_device = merged.groupby("device").head(1)
    assert (first_per_device["device_accounts"] == 0).all()


# -------------------------------------------------------------- churn ------
@pytest.fixture(scope="module")
def engagement():
    return cd.generate_engagement(n_accounts=400, days=200, seed=3)


def test_churn_features_ignore_the_future(engagement):
    """Features at a snapshot must not move when later history is appended."""
    snapshot = 120
    full = cd.build_features(engagement, snapshot)

    clipped = cd.Engagement(
        accounts=engagement.accounts,
        actions=engagement.actions[:, :snapshot],
        tickets=engagement.tickets[:, :snapshot],
        negative=engagement.negative[:, :snapshot],
        days=snapshot,
    )
    truncated = cd.build_features(clipped, snapshot)

    pd.testing.assert_frame_equal(
        full[cd.FEATURES].reset_index(drop=True),
        truncated[cd.FEATURES].reset_index(drop=True),
        check_exact=False, rtol=1e-12,
    )


def test_labels_need_a_fully_observed_window(engagement):
    """A label whose window runs past the data is not a label."""
    with pytest.raises(ValueError, match="runs past"):
        cd.build_labels(engagement, engagement.days - 10, window=30)


def test_book_excludes_accounts_that_already_left(engagement):
    """The nightly job scores the customers you still have.

    Without this the same departed account is re-labelled churned every night
    forever and the measured churn rate climbs with the age of the dataset.
    """
    snapshot = 150
    book = cd.on_book(engagement, snapshot)
    recent = engagement.actions[:, snapshot - 30:snapshot].sum(axis=1)
    assert (recent[book] > 0).all()
    assert (recent[~book] == 0).all() or (
        engagement.accounts["signup_day"].values[~book] > snapshot).any()


def test_cold_start_borrows_and_flags(engagement):
    """Cold-start rows get a cohort prior and a flag saying so."""
    snapshot = 150
    features = cd.build_features(engagement, snapshot, min_history=30)
    cold = features["is_cold_start"] == 1
    if not cold.any():
        pytest.skip("no cold-start accounts at this snapshot")
    # Borrowed values are cohort means, so they are not the raw zeros an
    # account with no history would otherwise carry.
    assert features.loc[cold, "actions_90d"].gt(0).all()
    assert features.loc[cold, "account_age_days"].lt(30).all()


def test_design_matrix_column_order_is_pinned(engagement):
    """The serving path positions features by order, so the order is a contract."""
    features = cd.build_features(engagement, 150)
    matrix = cd.design_matrix(features)
    assert list(matrix.columns) == cd.FEATURES
    assert matrix.notna().all().all()
