"""A card transaction stream with the three things the original POC lacked.

  1. Late labels. Every fraudulent transaction carries a chargeback filing
     delay, so at any moment the recent past is unlabelled rather than clean.
     This is the constraint the article says sets the retraining schedule.

  2. Rings. Organised fraud shares devices and cards across many accounts.
     Per-user velocity features cannot see a ring; network features can. The
     original generator gave every fraud its own unique device, so there was
     literally no ring in the data to find.

  3. Account age. Brand-new accounts are the highest-fraud segment, and the
     features that work on a tenured account are undefined on a fresh one.

Three fraud patterns run in the stream:

  takeover      a real account, a new device, a foreign country, big amounts.
                Runs for the first hundred days.
  ring          mule accounts opened in batches, sharing a small pool of
                devices and cards. Runs throughout, quieter and harder.
  card_testing  the tactic change. Local country, tiny amounts, high velocity,
                starting on the drift day. The model has never seen it, and
                because of label maturation it will not see a label from it
                for another three months.
"""

from __future__ import annotations

from collections import defaultdict, deque

import numpy as np
import pandas as pd

from .config import (
    DAY_SECONDS,
    RISK_ACCOUNTS,
    RISK_DAYS,
    RISK_DRIFT_DAY,
    RISK_MATURATION_DAYS,
    SEED,
)

FEATURES = [
    "txn_60s", "txn_10min", "txn_24h", "secs_since_last",
    "new_device", "new_country", "amount", "amount_ratio",
    "account_age_days", "device_accounts", "card_accounts", "device_txn_1h",
]

COLUMNS = ["ts", "account", "device", "card", "country", "amount", "account_age_days",
           "is_fraud", "pattern", "cb_delay_days", "label_ready_ts"]


def generate_stream(n_accounts: int = RISK_ACCOUNTS, days: int = RISK_DAYS,
                    drift_day: int = RISK_DRIFT_DAY, seed: int = SEED) -> pd.DataFrame:
    """Simulate the transaction stream. Returns one row per transaction."""
    rng = np.random.default_rng(seed)
    events: list[tuple] = []

    # ---------------------------------------------------------- accounts --
    # A quarter of accounts open during the window, so account age is a live
    # signal rather than a constant.
    signup_day = np.where(
        rng.random(n_accounts) < 0.75,
        -rng.integers(30, 1000, n_accounts),
        rng.integers(0, days - 5, n_accounts),
    ).astype(float)

    home_country = rng.integers(0, 20, n_accounts)
    mean_amount = np.exp(rng.normal(3.4, 0.6, n_accounts))       # about $30 typical
    rate = rng.uniform(0.8, 2.5, n_accounts)                     # transactions per day

    for account in range(n_accounts):
        start = max(0.0, signup_day[account])
        span = days - start
        if span <= 0:
            continue

        devices = [f"u{account}-d0"]
        if rng.random() < 0.35:
            devices.append(f"u{account}-d1")
        card = f"c{account}-0"

        n = rng.poisson(rate[account] * span)
        times = np.sort(rng.uniform(start, days, n)) * DAY_SECONDS
        n_times = len(times)
        if n_times == 0:
            continue

        # Occasional new device or travel, so those flags are not free wins.
        device_pick = rng.integers(0, len(devices), n_times)
        odd_device = rng.random(n_times) < 0.03
        odd_country = rng.random(n_times) < 0.01
        amounts = np.maximum(1.0, rng.normal(mean_amount[account],
                                             mean_amount[account] * 0.5, n_times))

        for i, ts in enumerate(times):
            device = (f"u{account}-d{rng.integers(2, 9)}" if odd_device[i]
                      else devices[device_pick[i]])
            country = int(rng.integers(0, 20)) if odd_country[i] else int(home_country[account])
            events.append((float(ts), account, device, card, country,
                           float(amounts[i]), 0, "legit"))

        # ------------------------------------------------------- takeover --
        if rng.random() < 0.14 and start < drift_day - 2:
            burst = int(rng.integers(3, 9))
            begin = float(rng.uniform(max(start, 0.5), drift_day - 1.5)) * DAY_SECONDS

            # Most takeovers arrive on an unknown device from somewhere else.
            # Some are session hijacks: the attacker is inside a legitimate
            # session, on a known device, in the home country. Nothing about
            # those rows looks wrong, and no feature set separates them. They
            # are the ceiling on how good any model here can be, and every real
            # fraud dataset has them.
            hijack = rng.random() < 0.30
            if hijack:
                device = devices[int(rng.integers(0, len(devices)))]
                country = int(home_country[account])
                scale = 1.6
            else:
                device = f"u{account}-dx{rng.integers(0, 9999)}"
                country = int((home_country[account] + rng.integers(5, 15)) % 20)
                scale = 2.6

            values = rng.normal(mean_amount[account] * scale, mean_amount[account], burst)
            for j in range(burst):
                events.append((begin + j * float(rng.uniform(10, 90)), account, device,
                               card, country, max(0.9, float(values[j])), 1, "takeover"))

        # --------------------------------------------------- card testing --
        # The drift week. Same accounts, completely different behaviour: small
        # amounts, local country, and fast. Fast enough that a velocity limit
        # sees it even though the model has never met the pattern.
        if rng.random() < 0.05 and days > drift_day:
            burst = int(rng.integers(5, 14))
            begin = float(rng.uniform(drift_day, days - 0.3)) * DAY_SECONDS
            device = f"u{account}-dt{rng.integers(0, 9999)}"
            for j in range(burst):
                events.append((begin + j * float(rng.uniform(2, 12)), account, device,
                               card, int(home_country[account]),
                               float(rng.uniform(0.9, 4.9)), 1, "card_testing"))

    # -------------------------------------------------------------- rings --
    # Mule accounts opened in a batch, sharing a handful of devices and cards.
    # Nothing about any single account looks unusual. The sharing is the tell.
    ring_events, ring_signups = _rings(rng, n_accounts, days, drift_day)
    events.extend(ring_events)
    signup_day = np.concatenate([signup_day, ring_signups])

    stream = pd.DataFrame(
        events,
        columns=["ts", "account", "device", "card", "country", "amount", "is_fraud", "pattern"],
    ).sort_values("ts", kind="mergesort").reset_index(drop=True)

    stream["account_age_days"] = (
        stream["ts"].values / DAY_SECONDS - signup_day[stream["account"].values]
    ).clip(0, 1000)

    # ------------------------------------------------------- maturation ---
    # When the label becomes knowable. Fraud is confirmed by a chargeback that
    # lands 30 to 90 days later, most often; a legitimate transaction is only
    # confirmed legitimate once the filing window closes without one.
    n = len(stream)
    delay = np.clip(rng.lognormal(np.log(45), 0.55, n), 7, 120)
    stream["cb_delay_days"] = np.where(stream["is_fraud"] == 1, delay, 120.0).round(2)
    stream["label_ready_ts"] = stream["ts"] + stream["cb_delay_days"] * DAY_SECONDS
    return stream[COLUMNS]


def _rings(rng, first_account_id: int, days: int, drift_day: int):
    """Fraud rings: many accounts, few devices, few cards."""
    events, signups = [], []
    account_id = first_account_id
    n_rings = 14

    for ring in range(n_rings):
        size = int(rng.integers(5, 14))
        opened = float(rng.uniform(3, days - 12))
        devices = [f"ring{ring}-d{i}" for i in range(int(rng.integers(1, 4)))]
        cards = [f"ring{ring}-c{i}" for i in range(int(rng.integers(1, 4)))]
        country = int(rng.integers(0, 20))

        for _ in range(size):
            signups.append(opened + float(rng.uniform(0, 4)))
            n = int(rng.integers(2, 7))
            times = np.sort(rng.uniform(opened + 0.2, min(opened + 11, days), n)) * DAY_SECONDS
            for ts in times:
                events.append((
                    float(ts), account_id,
                    devices[int(rng.integers(0, len(devices)))],
                    cards[int(rng.integers(0, len(cards)))],
                    country, float(rng.uniform(20, 180)), 1, "ring",
                ))
            account_id += 1
    return events, np.array(signups, dtype=float)


# ------------------------------------------------------------- features ----
def build_features(stream: pd.DataFrame) -> pd.DataFrame:
    """Point-in-time features in a single pass over the sorted stream.

    The invariant that makes this leakage-free: every feature for row i is read
    out of state that contains only rows 0..i-1, and the state is updated only
    after the row has been scored. That is the same guarantee an online feature
    store gives you at serving time, which is the whole reason to compute
    training features this way rather than with a group-by over the finished
    table.

    tests/test_point_in_time.py holds this to it by re-running the build on a
    truncated stream and checking every shared row comes out identical.
    """
    if not stream["ts"].is_monotonic_increasing:
        raise ValueError("stream must be sorted by ts; a later event cannot inform an earlier one")

    recent = defaultdict(deque)              # account -> timestamps in the last 24h
    device_recent = defaultdict(deque)       # device  -> timestamps in the last hour
    seen_devices = defaultdict(set)
    seen_countries = defaultdict(set)
    amount_history = defaultdict(deque)      # account -> last 20 amounts
    device_accounts = defaultdict(set)       # device  -> accounts seen on it
    card_accounts = defaultdict(set)         # card    -> accounts seen on it
    last_seen: dict[int, float] = {}

    n = len(stream)
    out = np.empty((n, len(FEATURES)), dtype=float)

    ts_all = stream["ts"].values
    account_all = stream["account"].values
    device_all = stream["device"].values
    card_all = stream["card"].values
    country_all = stream["country"].values
    amount_all = stream["amount"].values
    age_all = stream["account_age_days"].values

    for i in range(n):
        ts = ts_all[i]
        account = account_all[i]
        device = device_all[i]
        card = card_all[i]
        amount = amount_all[i]

        history = recent[account]
        while history and ts - history[0] > DAY_SECONDS:
            history.popleft()
        txn_24h = len(history)
        txn_10min = 0
        txn_60s = 0
        for earlier in reversed(history):        # newest first, stop early
            gap = ts - earlier
            if gap > 600:
                break
            txn_10min += 1
            if gap <= 60:
                txn_60s += 1

        device_history = device_recent[device]
        while device_history and ts - device_history[0] > 3600:
            device_history.popleft()

        amounts = amount_history[account]
        typical = (sum(amounts) / len(amounts)) if amounts else amount

        out[i] = (
            txn_60s,
            txn_10min,
            txn_24h,
            min(ts - last_seen.get(account, ts - 7 * DAY_SECONDS), 7 * DAY_SECONDS),
            float(device not in seen_devices[account]),
            float(country_all[i] not in seen_countries[account]),
            amount,
            amount / max(typical, 1.0),
            age_all[i],
            len(device_accounts[device]),
            len(card_accounts[card]),
            len(device_history),
        )

        # The row has been scored, so now it becomes history.
        history.append(ts)
        device_history.append(ts)
        seen_devices[account].add(device)
        seen_countries[account].add(country_all[i])
        amounts.append(amount)
        if len(amounts) > 20:
            amounts.popleft()
        device_accounts[device].add(account)
        card_accounts[card].add(account)
        last_seen[account] = ts

    frame = pd.DataFrame(out, columns=FEATURES)
    frame.insert(0, "account", account_all)
    frame.insert(0, "ts", ts_all)
    return frame


# ------------------------------------------------------------- labelling ---
def label_as_of(stream: pd.DataFrame, today_day: float) -> np.ndarray:
    """The label a pipeline would actually see if it ran today.

    A fraud whose chargeback has not landed yet reads as a zero. That is not a
    missing value the pipeline can notice; it is a wrong value it will happily
    train on.
    """
    ready = stream["label_ready_ts"].values <= today_day * DAY_SECONDS
    return np.where(ready, stream["is_fraud"].values, 0).astype(int)


def matured_mask(stream: pd.DataFrame, today_day: float,
                 maturation_days: float = RISK_MATURATION_DAYS) -> np.ndarray:
    """Rows old enough that their label can be trusted.

    Everything newer than this is in the blind window: partially labelled, and
    biased in one direction, because the labels still in flight are all
    positives.
    """
    cutoff = (today_day - maturation_days) * DAY_SECONDS
    return stream["ts"].values <= cutoff


def maturation_curve(stream: pd.DataFrame, max_days: int = 120) -> pd.DataFrame:
    """Share of fraud labels that have arrived, by days since the transaction.

    The article's Figure 1b, measured rather than asserted.
    """
    delays = stream.loc[stream["is_fraud"] == 1, "cb_delay_days"].values
    grid = np.arange(0, max_days + 1)
    return pd.DataFrame({
        "days_elapsed": grid,
        "share_confirmed": [(delays <= d).mean() for d in grid],
    })
