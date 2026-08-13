"""A SaaS engagement history with a real time axis.

The original POC used the IBM Telco snapshot, which has no time in it at all.
That makes four of the article's churn claims impossible to demonstrate: the
label is a window you choose, features must be built point-in-time, cold start
needs 30 to 90 days of history, and interventions erase the labels you would
retrain on. All four need to know when things happened.

So the engagement history is simulated here, and Telco stays in the pipeline as
a real-data cross-check that the metric story holds on data nobody made up.

The simulation has four things in it that matter:

  decay        accounts lose interest gradually, over weeks, the way the
               article describes churn behaving
  abrupt exits some accounts leave with no warning at all (a contract ends),
               which is the irreducible part no engagement feature can catch
  vacations    long gaps that are not churn. These are what a 14-day label
               window mislabels, and why the window is a decision
  reachability every account has a latent persuadability that decays as its
               decline progresses, so the accounts a churn model is most sure
               about are the ones a call is least likely to reach. That is
               Ascarza's result, and 04_feedback_loops.py measures it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import CHURN_ACCOUNTS, CHURN_DAYS, CHURN_MIN_HISTORY, SEED

PLANS = ("starter", "team", "business", "enterprise")

# How an account responds to a retention call. The names are the article's, and
# here they are a readable bucketing of two continuous latent traits rather
# than a hidden column: see treatment_effects() and describe_type() below.
UPLIFT_TYPES = ("persuadable", "sure_thing", "lost_cause", "sleeping_dog")

FEATURES = [
    "actions_30d", "actions_60d", "actions_90d",
    "active_days_30d", "wow_change", "trend_30_over_90",
    "days_since_last_action", "tickets_30d", "negative_ticket_share",
    "account_age_days", "seats", "monthly_price", "is_cold_start",
]


@dataclass
class Engagement:
    """A simulated engagement history, ready to be sliced as of any day."""

    accounts: pd.DataFrame       # one row per account, static attributes
    actions: np.ndarray          # (n_accounts, days) actions taken that day
    tickets: np.ndarray          # (n_accounts, days) support tickets opened
    negative: np.ndarray         # (n_accounts, days) of those, angry ones
    days: int

    def __len__(self) -> int:
        return len(self.accounts)


def generate_engagement(n_accounts: int = CHURN_ACCOUNTS, days: int = CHURN_DAYS,
                        seed: int = SEED) -> Engagement:
    """Simulate daily product usage for a book of SaaS accounts."""
    rng = np.random.default_rng(seed)

    # ---------------------------------------------------------- accounts --
    # Most accounts predate the window; the rest sign up during it, which is
    # what makes cold start a real condition rather than a hypothetical.
    signup = np.where(
        rng.random(n_accounts) < 0.75,
        -rng.integers(30, 900, n_accounts),          # already tenured on day 0
        rng.integers(0, days - 40, n_accounts),      # signs up mid-window
    )
    plan_idx = rng.choice(len(PLANS), n_accounts, p=[0.40, 0.32, 0.20, 0.08])
    seats = np.maximum(1, np.round(rng.lognormal(1.1, 0.9, n_accounts))).astype(int)
    price = np.array([12, 30, 75, 190])[plan_idx] * np.clip(seats / 3, 0.5, 12)

    # Baseline intensity: bigger, pricier accounts use the product more.
    base_rate = np.clip(
        rng.lognormal(0.5, 0.7, n_accounts) * (1 + 0.35 * plan_idx) * np.clip(seats / 4, 0.4, 3.0),
        0.05, 40.0,
    )

    # ------------------------------------------------------ decay & exits --
    # A daily hazard of entering decline, plus a rarer abrupt exit.
    decay_start = np.full(n_accounts, 10**6, dtype=float)
    hazard = 0.0011 * (1.6 - 0.22 * plan_idx)        # small accounts drift off faster
    draws = rng.random((n_accounts, days))
    started = draws < hazard[:, None]
    has_decay = started.any(axis=1)
    decay_start[has_decay] = started.argmax(axis=1)[has_decay]
    tau = rng.uniform(8.0, 30.0, n_accounts)          # how fast interest fades

    abrupt = np.full(n_accounts, 10**6, dtype=float)
    is_abrupt = rng.random(n_accounts) < 0.035
    abrupt[is_abrupt] = rng.integers(40, days, is_abrupt.sum())

    # -------------------------------------------------------- vacations --
    # Long gaps that are not churn. A 14-day label window calls these churn.
    vac_start = np.full(n_accounts, 10**6, dtype=float)
    on_vacation = rng.random(n_accounts) < 0.13
    vac_start[on_vacation] = rng.integers(20, days - 30, on_vacation.sum())
    vac_len = rng.integers(15, 27, n_accounts)

    # ------------------------------------------------------- daily rates --
    grid = np.arange(days)[None, :]
    active = grid >= signup[:, None]                                   # exists yet
    decayed = np.exp(-np.maximum(0.0, grid - decay_start[:, None]) / tau[:, None])
    alive = grid < abrupt[:, None]                                     # not gone
    vacationing = (grid >= vac_start[:, None]) & (grid < (vac_start + vac_len)[:, None])

    # Weekends are quieter. It keeps short-window features honest.
    weekday = 1.0 - 0.55 * ((grid % 7) >= 5)

    rate = base_rate[:, None] * decayed * weekday
    rate = np.where(active & alive & ~vacationing, rate, 0.0)

    actions = rng.poisson(rate)

    # Support tickets scale with usage but spike while an account is souring.
    souring = (grid >= decay_start[:, None]) & (grid < (decay_start + 45)[:, None])
    ticket_rate = np.where(active & alive, 0.006 * base_rate[:, None] + 0.02 * souring, 0.0)
    tickets = rng.poisson(ticket_rate)
    negative = rng.binomial(tickets, np.where(souring, 0.65, 0.18))

    # ------------------------------------------------ response to contact --
    # How an account responds to being contacted is a latent trait, not a
    # column anyone gets to see. Two of them:
    #
    #   persuadability  how much a call moves this account, at its best moment
    #   annoyance       how likely a call is to remind it to cancel
    #
    # Both are correlated with observables (tenure, plan, price) so a
    # treatment-effect model has something to find, and neither is determined
    # by them, so it stays a genuine estimation problem.
    tenure_score = np.clip((-signup) / 600.0, 0, 1)
    persuadability = np.clip(
        rng.beta(2.0, 3.0, n_accounts) * (1.35 - 0.45 * tenure_score)
        * (1.15 - 0.10 * plan_idx),
        0.0, 0.95)
    annoyance = np.where(rng.random(n_accounts) < 0.22,
                         rng.uniform(0.06, 0.24, n_accounts), 0.0)
    # A contract that simply ends cannot be talked out of anything.
    persuadability = np.where(is_abrupt, persuadability * 0.15, persuadability)

    accounts = pd.DataFrame({
        "account_id": [f"A{i:06d}" for i in range(n_accounts)],
        "signup_day": signup,
        "plan": [PLANS[i] for i in plan_idx],
        "seats": seats,
        "monthly_price": price.round(2),
        "persuadability": persuadability.round(4),
        "annoyance": annoyance.round(4),
        "decay_start": decay_start,
        "decay_tau": tau.round(3),
    })
    return Engagement(accounts=accounts, actions=actions, tickets=tickets,
                      negative=negative, days=days)


# ------------------------------------------------- response to contact -----
# How far into its decline an account can still be reached. Reachability decays
# with e-folding time REACHABLE_TAU, against declines that take 35 to 130 days
# to reach silence, so early decliners are broadly saveable and late ones are
# largely not.
REACHABLE_TAU = 90.0
LOST_CAUSE_BELOW = 0.12       # save rate under which a call is not worth making
SLEEPING_DOG_ABOVE = 0.06     # wake rate over which contacting is net harmful


def treatment_effects(eng: Engagement, snapshot: int) -> pd.DataFrame:
    """What a retention contact would actually do, per account, on this day.

    This is where Ascarza's result comes from, and it is a statement about how
    churn works rather than a convenience for the demo. An account's
    reachability decays with how far into its decline it already is. By the
    time a churn model is confident about someone, they have usually been
    leaving for weeks, and the call that would have worked was six weeks ago.

    So the highest-scoring accounts are systematically the least saveable, the
    saveable ones sit lower down the list still wavering, and "call the top of
    the risk list" is the wrong policy for a reason that has nothing to do with
    the model being bad at its job.

    None of these columns exist in production. They are the simulator's ground
    truth, used to score targeting policies in 04_feedback_loops.py.
    """
    progress = np.maximum(0.0, snapshot - eng.accounts["decay_start"].values)
    progress = np.minimum(progress, 400.0)                 # never-decayed accounts
    reachability = np.exp(-progress / REACHABLE_TAU)
    return pd.DataFrame({
        "account_id": eng.accounts["account_id"].values,
        "save_rate": eng.accounts["persuadability"].values * reachability,
        "wake_rate": eng.accounts["annoyance"].values,
    })


def describe_type(save_rate, wake_rate, would_churn) -> np.ndarray:
    """Bucket accounts into the article's four names, for exposition only.

    The names are a readable summary of two continuous traits, not a hidden
    column the model could have found. Nothing in the pipeline branches on them.
    """
    save_rate = np.asarray(save_rate)
    wake_rate = np.asarray(wake_rate)
    would_churn = np.asarray(would_churn)
    return np.where(
        would_churn == 1,
        np.where(save_rate >= LOST_CAUSE_BELOW, "persuadable", "lost_cause"),
        np.where(wake_rate >= SLEEPING_DOG_ABOVE, "sleeping_dog", "sure_thing"),
    )


# --------------------------------------------------------------- labels ----
def on_book(eng: Engagement, snapshot: int, lookback: int = 30) -> np.ndarray:
    """Accounts the nightly job should actually score.

    An account is on the book if it has signed up and has done something in the
    recent past. This is not a detail: without it, an account that went quiet
    six months ago is re-labelled "churned" every night forever, the measured
    churn rate climbs with the age of the dataset, and the model spends its
    capacity re-identifying customers who left long ago.

    A churn model scores the customers you still have.
    """
    exists = eng.accounts["signup_day"].values <= snapshot
    recent = eng.actions[:, max(0, snapshot - lookback):snapshot].sum(axis=1) > 0
    return exists & recent


def build_labels(eng: Engagement, snapshot: int, window: int,
                 lookback: int = 30) -> pd.DataFrame:
    """Label the book at `snapshot` using `window` days of forward silence.

    This is the article's Figure 1a: the label is a window you choose, not a
    fact about the world. Pick it too short and holidays look like churn; pick
    it too long and the label confirms after it is too late to act on.
    """
    if snapshot + window > eng.days:
        raise ValueError(
            f"snapshot {snapshot} + window {window} runs past the {eng.days}-day history; "
            "the forward window has to be fully observed or the label is not known")

    book = on_book(eng, snapshot, lookback)
    forward = eng.actions[:, snapshot:snapshot + window].sum(axis=1)
    return pd.DataFrame({
        "account_id": eng.accounts["account_id"].values[book],
        "label_ts": snapshot,
        "churned": (forward[book] == 0).astype(int),
    }).reset_index(drop=True)


def returned_after_window(eng: Engagement, snapshot: int, window: int,
                          lookback: int = 30) -> np.ndarray:
    """Accounts labelled churned at this window that came back afterwards.

    A short window's false positives: people on vacation, not people leaving.
    Returned as a mask over the whole account table.
    """
    forward = eng.actions[:, snapshot:snapshot + window].sum(axis=1)
    later = eng.actions[:, snapshot + window:].sum(axis=1)
    return (forward == 0) & (later > 0) & on_book(eng, snapshot, lookback)


# ------------------------------------------------------------- features ----
def build_features(eng: Engagement, snapshot: int,
                   min_history: int = CHURN_MIN_HISTORY,
                   lookback: int = 30,
                   book_snapshot: int | None = None) -> pd.DataFrame:
    """Features as they were on `snapshot`, using no data from on or after it.

    `book_snapshot` decides which accounts are in the population, and defaults
    to `snapshot`. They come apart only when deliberately building the leaky
    counterfactual in 01_churn_batch_pipeline.py, where the population has to
    stay fixed at the label date while the features are allowed to peek ahead.

    The point-in-time guarantee is structural rather than trusted: every slice
    below stops at `snapshot`, so there is no future column available to leak
    even by accident. On Databricks this is what create_training_set does with
    a time series feature table and an as-of join.
    """
    if not 0 < snapshot <= eng.days:
        raise ValueError(f"snapshot {snapshot} outside the 0..{eng.days} history")

    past = eng.actions[:, :snapshot]                       # strictly before the label
    tick = eng.tickets[:, :snapshot]
    neg = eng.negative[:, :snapshot]

    def window(matrix, n):
        return matrix[:, max(0, snapshot - n):].sum(axis=1).astype(float)

    actions_30 = window(past, 30)
    actions_60 = window(past, 60)
    actions_90 = window(past, 90)
    last_7 = window(past, 7)
    prev_7 = past[:, max(0, snapshot - 14):max(0, snapshot - 7)].sum(axis=1).astype(float)

    active_days_30 = (past[:, max(0, snapshot - 30):] > 0).sum(axis=1).astype(float)

    # Days since the last action. Accounts with no history at all get the full
    # window rather than a sentinel, so the column stays interpretable.
    had_action = past > 0
    ever = had_action.any(axis=1)
    last_idx = np.where(ever, snapshot - 1 - had_action[:, ::-1].argmax(axis=1), -1)
    days_since = np.where(ever, snapshot - 1 - last_idx, 90.0).astype(float)

    age = np.clip(snapshot - eng.accounts["signup_day"].values, 0, None).astype(float)

    frame = pd.DataFrame({
        "account_id": eng.accounts["account_id"].values,
        "actions_30d": actions_30,
        "actions_60d": actions_60,
        "actions_90d": actions_90,
        "active_days_30d": active_days_30,
        "wow_change": (last_7 - prev_7) / np.maximum(prev_7, 1.0),
        "trend_30_over_90": actions_30 / np.maximum(actions_90 / 3.0, 0.5),
        "days_since_last_action": np.minimum(days_since, 90.0),
        "tickets_30d": window(tick, 30),
        "negative_ticket_share": window(neg, 30) / np.maximum(window(tick, 30), 1.0),
        "account_age_days": np.minimum(age, 900.0),
        "seats": eng.accounts["seats"].values.astype(float),
        "monthly_price": eng.accounts["monthly_price"].values.astype(float),
        "plan": eng.accounts["plan"].values,
        "snapshot": snapshot,
    })

    book = on_book(eng, snapshot if book_snapshot is None else book_snapshot, lookback)
    frame = frame[book].reset_index(drop=True)
    return apply_cold_start(frame, min_history)


def apply_cold_start(frame: pd.DataFrame, min_history: int = CHURN_MIN_HISTORY) -> pd.DataFrame:
    """Fall back to cohort averages where an account has no history yet.

    Churn features need 30 to 90 days of history. Below that the rolling
    windows are not small, they are meaningless: an account that signed up
    yesterday has zero actions in the last 90 days for the same reason a
    departing account does, and the model cannot tell those apart.

    So the behavioural columns are replaced with the account's plan-cohort
    average, and a flag tells the model the numbers are borrowed. The flag
    matters more than the fill: it lets the model learn a different response
    for accounts it knows nothing about.
    """
    frame = frame.copy()
    cold = frame["account_age_days"] < min_history
    frame["is_cold_start"] = cold.astype(float)

    borrowed = ["actions_30d", "actions_60d", "actions_90d", "active_days_30d",
                "wow_change", "trend_30_over_90", "days_since_last_action",
                "tickets_30d", "negative_ticket_share"]

    if cold.any() and (~cold).any():
        cohort = frame.loc[~cold].groupby("plan")[borrowed].mean()
        overall = frame.loc[~cold, borrowed].mean()
        for column in borrowed:
            fill = frame.loc[cold, "plan"].map(cohort[column]).fillna(overall[column])
            frame.loc[cold, column] = fill.values
    return frame


def design_matrix(frame: pd.DataFrame) -> pd.DataFrame:
    """The numeric matrix the model sees. Column order is fixed and asserted."""
    matrix = frame[FEATURES].astype(float)
    if list(matrix.columns) != FEATURES:
        raise AssertionError("feature order drifted; the serving path depends on it")
    return matrix
