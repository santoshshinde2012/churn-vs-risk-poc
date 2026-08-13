"""Both models change their own training data.

This is the part of the article with no code anywhere else in this repo, and
almost none anywhere else at all. Both systems act on their predictions, and
both damage the data they would retrain on. They damage it in opposite
directions, which is why the fixes are different.

  Churn. The retention call works, the customer stays, and the row that made
  the model right now says "did not churn". The model looks wrong exactly
  where it was right, and retraining on that teaches it to stop finding the
  people it just saved.

  Risk. The transaction is blocked, so nobody ever learns whether it was
  fraud. Retrain on what was allowed and the model goes blind precisely where
  it acts. This is the selective labels problem, the same structure Lakkaraju
  and coauthors documented in bail decisions, where the outcome is only
  observed for the defendants the judge released.

And then the churn question underneath all of it: a high churn score says who
might leave, not who can be talked out of it.

Run 01 and 02 first, then: python 04_feedback_loops.py
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import average_precision_score

from churnrisk import churn_data as cd
from churnrisk import holdout, registry, viz
from churnrisk import risk_data as rd
from churnrisk.config import (
    CHURN_CAMPAIGN_SNAPSHOT,
    CHURN_LABEL_WINDOW,
    CHURN_MIN_HISTORY,
    CHURN_SCORE_SNAPSHOT,
    DAY_SECONDS,
    METRICS,
    MODELS,
    PLOTS,
    RISK_DAYS,
    RISK_DRIFT_DAY,
    RISK_OOT_START,
    SEED,
    ensure_dirs,
)
from churnrisk.policy import load_policy

ensure_dirs()
rng = np.random.default_rng(SEED)

churn_policy = load_policy("churn_policy_v1")
risk_policy = load_policy("risk_policy_v1")
HOLDOUT_SHARE = float(churn_policy.holdout["share"])

print("=" * 78)
print("FEEDBACK LOOPS: WHAT ACTING ON A PREDICTION DOES TO THE NEXT TRAINING SET")
print("=" * 78)


def new_model():
    return CalibratedClassifierCV(
        HistGradientBoostingClassifier(random_state=SEED, max_iter=200),
        method="isotonic", cv=3,
    )


def proportion_ci(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wald interval. Printed so nobody reads a noisy difference as a result."""
    if n == 0:
        return (float("nan"), float("nan"))
    rate = successes / n
    half = z * np.sqrt(max(rate * (1 - rate), 1e-12) / n)
    return (rate - half, rate + half)


# ###########################################################################
# PART A. CHURN: THE INTERVENTION ERASES THE LABEL
# ###########################################################################
print("\n" + "#" * 78)
print("PART A. CHURN")
print("#" * 78)

accounts = pd.read_parquet(MODELS / "churn_accounts.parquet")
cached = np.load(MODELS / "churn_engagement.npz")
eng = cd.Engagement(accounts=accounts, actions=cached["actions"], tickets=cached["tickets"],
                    negative=cached["negative"], days=cached["actions"].shape[1])

model = registry.load("ml.churn.scorer")

# The programme runs every month, not once, and two different control groups
# are doing two different jobs.
#
# The standing holdout is 15% of the book, permanent, and its job is to keep
# measuring whether the programme works at all. It is deliberately small,
# because every account in it is an account nobody is allowed to help.
#
# A treatment-effect model needs far more than that. Estimating who responds is
# a harder question than estimating whether anyone responds, and 15% of an
# eligible slice cannot answer it. So for three months the programme also ran a
# proper randomised test inside the eligible pool: half contacted, half not.
# That is a real cost, paid deliberately, for a targeting rule that pays it back.
CAMPAIGN_MONTHS = (90, 110, 130, 150)
TARGETING_MONTH = CHURN_CAMPAIGN_SNAPSHOT           # held out from the pooled fit
ELIGIBLE_AT = churn_policy.bands["nurture"]         # the monthly campaign reaches these
EXPERIMENT_SALT = "uplift-experiment-v1"


def run_campaign(snapshot: int, control_share: float,
                 salt: str = "retention-holdout-v1") -> pd.DataFrame:
    """One month: score the book, contact the eligible, record what happened.

    `churned` from the simulator is the counterfactual truth - what would have
    happened if nobody was contacted. Production never sees that column. What
    it sees is `observed_churn`, after the campaign has already changed it.
    """
    features = cd.build_features(eng, snapshot, CHURN_MIN_HISTORY)
    labels = cd.build_labels(eng, snapshot, CHURN_LABEL_WINDOW)
    month = (features.merge(labels, on="account_id")
                     .merge(cd.treatment_effects(eng, snapshot), on="account_id")
                     .rename(columns={"churned": "would_churn"}))
    month["score"] = model.predict_proba(cd.design_matrix(month))[:, 1]
    month["holdout"] = holdout.assign(month["account_id"], control_share, salt)

    eligible = month["score"] >= ELIGIBLE_AT
    month["eligible"] = eligible
    month["treated"] = eligible & ~month["holdout"]
    month["control"] = eligible & month["holdout"]

    draws = rng.random(len(month))
    save = month["save_rate"].values
    wake = month["wake_rate"].values
    would = month["would_churn"].values
    contacted = month["treated"].values
    month["uplift_type"] = cd.describe_type(save, wake, would)

    month["saved"] = contacted & (would == 1) & (draws < save)
    month["woken"] = contacted & (would == 0) & (draws < wake)
    month["observed_churn"] = np.where(month["saved"], 0,
                                       np.where(month["woken"], 1, would)).astype(int)
    month["snapshot"] = snapshot
    return month


campaigns = pd.concat([run_campaign(s, 0.50, EXPERIMENT_SALT) for s in CAMPAIGN_MONTHS],
                      ignore_index=True)
frame = run_campaign(TARGETING_MONTH, HOLDOUT_SHARE)
pooled = pd.concat([campaigns, frame], ignore_index=True)

treated = frame["treated"].values
control = frame["control"].values
eligible = frame["eligible"].values
saved = frame["saved"].values
woken = frame["woken"].values

print(f"randomised test months (50/50): {list(CAMPAIGN_MONTHS)}")
print(f"targeting month (standing {HOLDOUT_SHARE:.0%} holdout): day {TARGETING_MONTH}")
print(f"book on day {TARGETING_MONTH}: {len(frame):,} accounts   "
      f"counterfactual churn rate {frame['would_churn'].mean():.2%}")
print(f"\neligible (score >= {ELIGIBLE_AT:.2f}) on day {TARGETING_MONTH}: "
      f"{int(eligible.sum()):,}")
print(f"  contacted (treated): {int(treated.sum()):,}")
print(f"  held out (control):  {int(control.sum()):,}   <- nobody touches these")
print(f"pooled across all {len(CAMPAIGN_MONTHS) + 1} months: "
      f"{int(pooled['treated'].sum()):,} treated, {int(pooled['control'].sum()):,} control")

print(f"\non day {TARGETING_MONTH} the campaign saved {int(saved.sum())} accounts and talked "
      f"{int(woken.sum())} into leaving.")
print("both numbers are invisible in production. what the warehouse records is")
print("simply whether each account churned, with no note about why.")

# ------------------------------------------- 2. incremental retention -----
print("\n" + "-" * 78)
print("A1. INCREMENTAL RETENTION: THE ONLY HONEST MEASURE")
print("-" * 78)

treated_churn = pooled.loc[pooled["treated"], "observed_churn"]
control_churn = pooled.loc[pooled["control"], "observed_churn"]
lift = control_churn.mean() - treated_churn.mean()
lo_t, hi_t = proportion_ci(int(treated_churn.sum()), len(treated_churn))
lo_c, hi_c = proportion_ci(int(control_churn.sum()), len(control_churn))
se = np.sqrt((treated_churn.var(ddof=1) / len(treated_churn))
             + (control_churn.var(ddof=1) / len(control_churn)))

print(f"churn among the {len(treated_churn):,} called:      {treated_churn.mean():.2%}   "
      f"95% CI [{lo_t:.2%}, {hi_t:.2%}]")
print(f"churn among the {len(control_churn):,} held out:    {control_churn.mean():.2%}   "
      f"95% CI [{lo_c:.2%}, {hi_c:.2%}]")
print(f"\nincremental retention: {lift:+.2%} +/- {1.96 * se:.2%} "
      f"({'significant' if abs(lift) > 1.96 * se else 'NOT significant'} at 95%)")
print("\nthis number does not exist without the holdout. the campaign's own")
print("numbers would say the called accounts churned less than the book average,")
print("which is true and means nothing: they were picked for being different.")

# -------------------------------------- 3. the labels are now damaged -----
print("\n" + "-" * 78)
print("A2. THE MODEL LOOKS WRONG EXACTLY WHERE IT WAS RIGHT")
print("-" * 78)

saved_rows = frame[saved]
print(f"the {len(saved_rows)} saved accounts scored "
      f"{saved_rows['score'].mean():.2f} on average, well into the call band.")
print("in the next training set they are labelled 'did not churn'. the model")
print("gets punished for the prediction that caused the save.")

# Round two, trained on what the warehouse now holds, against round two
# trained on labels the campaign never touched.
X_round2 = cd.design_matrix(frame)
contaminated = new_model().fit(X_round2, frame["observed_churn"].values)
clean = new_model().fit(X_round2, frame["would_churn"].values)

# Evaluated on a later snapshot with no campaign in it, so the test labels are
# clean for both and the only difference is what they learned from.
later = cd.build_features(eng, CHURN_SCORE_SNAPSHOT, CHURN_MIN_HISTORY).merge(
    cd.build_labels(eng, CHURN_SCORE_SNAPSHOT, CHURN_LABEL_WINDOW), on="account_id")
X_later, y_later = cd.design_matrix(later), later["churned"].values

pr_contaminated = average_precision_score(
    y_later, contaminated.predict_proba(X_later)[:, 1])
pr_clean = average_precision_score(y_later, clean.predict_proba(X_later)[:, 1])

print("\nnext month's model, trained on the labels the campaign left behind: "
      f"PR-AUC {pr_contaminated:.3f}")
print("next month's model, trained on labels no campaign touched:          "
      f"PR-AUC {pr_clean:.3f}")
print(f"difference: {pr_contaminated - pr_clean:+.3f}")

persuadable = frame["uplift_type"].values == "persuadable"
score_contaminated = contaminated.predict_proba(X_round2)[:, 1][persuadable]
score_clean = clean.predict_proba(X_round2)[:, 1][persuadable]
print("\naverage score given to persuadable accounts:")
print(f"  by the contaminated model: {score_contaminated.mean():.4f}")
print(f"  by the clean model:        {score_clean.mean():.4f}")
print(f"  the contaminated model rates them {1 - score_contaminated.mean() / max(score_clean.mean(), 1e-9):.1%} "
      "lower")
# The measure that matters is not average score but who ends up on tomorrow's
# call list, so count the saveable churners each model would put there.
saveable = (frame["would_churn"].values == 1) & (frame["save_rate"].values >= cd.LOST_CAUSE_BELOW)
budget_next = int(churn_policy.capacity["calls_per_night"])
in_list = {}
for name, fitted in (("contaminated", contaminated), ("clean", clean)):
    ranked = np.argsort(fitted.predict_proba(X_round2)[:, 1])[::-1][:budget_next]
    in_list[name] = int(saveable[ranked].sum())

print(f"\nsaveable churners in tomorrow's top {budget_next}, by which labels trained the model:")
print(f"  contaminated: {in_list['contaminated']}")
print(f"  clean:        {in_list['clean']}")
print("\nthis is one round. the effect is small and it is directional, which is the")
print("dangerous combination: nothing in a monitoring dashboard fires, and the")
print("model gets a little worse at finding the segment the programme is best at")
print("keeping, every month, forever. the holdout is what arrests it. those rows")
print("never got called, so their labels still say what would have happened, and")
print("a model trained with them in it keeps a memory of the world before the")
print("campaign existed.")

# ----------------------------------------------- 4. uplift, not risk ------
print("\n" + "-" * 78)
print("A3. WHO CAN BE TALKED OUT OF IT (UPLIFT, NOT RISK)")
print("-" * 78)

# Both uplift estimators are fitted on the randomised months and applied to the
# held-out one, so what follows is a targeting rule tested out of time rather
# than a fit being admired.
history = campaigns[campaigns["eligible"]].reset_index(drop=True)
X_history = cd.design_matrix(history)
was_control = history["control"].values
y_history = history["observed_churn"].values

pool = frame[eligible].reset_index(drop=True)
X_pool = cd.design_matrix(pool)

# --- estimator 1: the T-learner, which is what everyone writes first --------
# One outcome model per arm, uplift is the gap. It is intuitive and it is a
# variance trap: each arm carries its own error, the difference carries both,
# and the treatment effect being estimated is smaller than either error.
def arm(mask):
    return HistGradientBoostingClassifier(
        random_state=SEED, max_iter=120, max_depth=4, min_samples_leaf=40,
    ).fit(X_history[mask], y_history[mask])


treated_outcome, control_outcome = arm(~was_control), arm(was_control)
pool["uplift_t_learner"] = (control_outcome.predict_proba(X_pool)[:, 1]
                            - treated_outcome.predict_proba(X_pool)[:, 1])

# --- estimator 2: the transformed outcome, which is what to use -------------
# Rewrite each row's outcome so that its conditional mean IS the treatment
# effect, then regress on it directly. With a 50/50 randomised assignment,
#     Z = Y * (W - e) / (e * (1 - e))   and   E[Z | X] = tau(X)
# One model, one source of error, and the thing being fitted is the quantity
# actually wanted rather than a difference between two things that are not.
propensity = 0.5                                    # by construction, in these months
W = (~was_control).astype(float)
Z = y_history * (W - propensity) / (propensity * (1 - propensity))
transformed = HistGradientBoostingRegressor(
    random_state=SEED, max_iter=200, max_depth=4, min_samples_leaf=40,
).fit(X_history, Z)
pool["uplift"] = -transformed.predict(X_pool)       # negated: churn avoided

print(f"uplift models fitted on months {list(CAMPAIGN_MONTHS)} "
      f"({int((~was_control).sum()):,} contacted, {int(was_control.sum()):,} control)")
print(f"applied to the {len(pool):,} eligible accounts on day {TARGETING_MONTH}\n")

print("composition of the top decile by churn score:")
top_decile = pool.nlargest(max(len(pool) // 10, 1), "score")
composition = top_decile["uplift_type"].value_counts(normalize=True)
for kind in cd.UPLIFT_TYPES:
    print(f"  {kind:<13} {composition.get(kind, 0.0):.1%}")
print(f"\n{composition.get('lost_cause', 0.0):.0%} of the accounts a risk model is most confident about "
      "cannot be saved:")
print("by the time the signal is unambiguous, the decision is old. another")
print(f"{composition.get('sleeping_dog', 0.0):.0%} were staying until someone reminded them they were paying")
print("for this. neither belongs on a call list, and no churn score can tell")
print("you which is which, because both look exactly like leaving.")

# Score each targeting policy by the effect it would really have. The
# simulator knows every account's type, so this is the true expected number of
# customers kept, not an estimate.
save_rate = pool["save_rate"].values
wake_rate = pool["wake_rate"].values
would_churn = pool["would_churn"].values
true_effect = np.where(would_churn == 1, save_rate, -wake_rate)

budget = int(churn_policy.capacity["calls_per_night"])


def spend(ranking, eligible_mask=None):
    """Pick `budget` accounts by a ranking, optionally screened first."""
    ranking = np.asarray(ranking, dtype=float)
    if eligible_mask is not None:
        ranking = np.where(eligible_mask, ranking, -np.inf)
    return np.argsort(ranking)[::-1][:budget]


screen = pool["uplift"].values > 0          # only the sign is used, not the size
strategies = {
    "highest risk score": spend(pool["score"].values),
    "highest uplift (T-learner)": spend(pool["uplift_t_learner"].values),
    "highest uplift (transformed outcome)": spend(pool["uplift"].values),
    "highest risk, screened by uplift sign": spend(pool["score"].values, screen),
    "at random": spend(rng.random(len(pool))),
    "clairvoyant ceiling": spend(true_effect),
}

strategy_rows = []
for name, chosen in strategies.items():
    kinds = pool["uplift_type"].values[chosen]
    strategy_rows.append({
        "strategy": name,
        "customers_kept": float(true_effect[chosen].sum()),
        "per_100_calls": float(true_effect[chosen].sum() / budget * 100),
        "lost_causes": int((kinds == "lost_cause").sum()),
        "sleeping_dogs": int((kinds == "sleeping_dog").sum()),
    })

uplift_table = pd.DataFrame(strategy_rows)
print(f"\nsame budget of {budget} calls, six ways of spending it")
print("(customers kept is the simulator's truth, not an estimate):")
print(uplift_table.to_string(index=False, float_format=lambda v: f"{v:.2f}"))

by_risk = uplift_table.iloc[0]["customers_kept"]
by_screen = uplift_table.set_index("strategy").loc[
    "highest risk, screened by uplift sign", "customers_kept"]
by_rank = uplift_table.set_index("strategy").loc[
    "highest uplift (transformed outcome)", "customers_kept"]
ceiling = uplift_table.iloc[-1]["customers_kept"]

# Why one use of the same model works and the other does not.
corr_score = float(np.corrcoef(pool["score"].values, true_effect)[0, 1])
corr_uplift = float(np.corrcoef(pool["uplift"].values, true_effect)[0, 1])

print("\nscreening out the accounts with a non-positive estimated uplift and then")
print(f"calling the riskiest of what is left keeps {by_screen - by_risk:+.1f} customers "
      f"({by_screen / by_risk - 1:+.0%})")
print(f"for the same {budget} calls. it drops "
      f"{int((~screen).sum()):,} of {len(pool):,} accounts before ranking.")
print("\nranking by the same uplift estimate does worse than the plain risk score "
      f"({by_rank:.1f} vs {by_risk:.1f}).")
print("that is not a contradiction, and it is the practical lesson in this section.")
print(f"  correlation of the risk score with the true effect: {corr_score:+.3f}")
print(f"  correlation of the uplift estimate with it:         {corr_uplift:+.3f}")
print("\nthe sign of a treatment effect is estimable from a quarter of experiments.")
print("its magnitude, per account, is not: it is a difference between two noisy")
print("quantities, and the noise is larger than the thing being measured. so use")
print("the uplift model for what it can support - excluding the people a call")
print("would waste or backfire on - and let the churn score do the ranking inside")
print("what survives. treating an uplift score as a league table is how uplift")
print("projects get cancelled.")
print(f"\nand the headroom is real: a clairvoyant list keeps {ceiling:.0f}, "
      f"against {by_screen:.0f} for the best")
print("rule here. most of that gap buys more experiments, not a better model.")

fig, axes = viz.new_grid(2, width=11.5)
shown = uplift_table[uplift_table["strategy"] != "clairvoyant ceiling"]
short = {"highest risk score": "risk", "highest uplift (T-learner)": "uplift\n(T-learner)",
         "highest uplift (transformed outcome)": "uplift\n(transformed)",
         "highest risk, screened by uplift sign": "risk, uplift\nscreened",
         "at random": "random"}
axes[0].bar([short[s] for s in shown["strategy"]], shown["customers_kept"],
            color=[viz.BLUE, viz.INK_SOFT, viz.INK_SOFT, viz.AQUA, viz.INK_SOFT], width=0.55)
for i, value in enumerate(shown["customers_kept"]):
    axes[0].text(i, value + 0.3, f"{value:.1f}", ha="center", color=viz.INK, fontsize=9)
axes[0].axhline(ceiling, color=viz.ORANGE, linestyle="--", linewidth=1.5)
axes[0].text(len(shown) - 0.5, ceiling * 0.94, f"clairvoyant ceiling {ceiling:.0f}",
             ha="right", color=viz.INK_SOFT, fontsize=8)
axes[0].tick_params(axis="x", labelsize=7)
viz.label(axes[0], f"Customers kept per {budget} calls", "", "customers kept")

# Orange for the two the call list should not contain: a lost cause leaves
# anyway, and a sleeping dog leaves *because* you called.
order = list(cd.UPLIFT_TYPES)
worth_calling = {"persuadable": viz.AQUA, "sure_thing": viz.INK_SOFT,
                 "lost_cause": viz.ORANGE, "sleeping_dog": viz.ORANGE}
shares = [composition.get(k, 0.0) for k in order]
axes[1].bar(order, shares, color=[worth_calling[k] for k in order], width=0.55)
for i, value in enumerate(shares):
    axes[1].text(i, value + 0.006, f"{value:.0%}", ha="center", color=viz.INK, fontsize=9)
wasted = composition.get("lost_cause", 0.0) + composition.get("sleeping_dog", 0.0)
axes[1].set_ylim(0, max(shares) * 1.25)
axes[1].tick_params(axis="x", labelsize=8)
viz.label(axes[1], f"Who is in the top decile by risk "
          f"({wasted:.0%} of it should not be called)", "", "share of the decile")
viz.save(fig, PLOTS / "churn_uplift.png")


# ###########################################################################
# PART B. RISK: THE BLOCK ERASES THE OUTCOME
# ###########################################################################
print("\n" + "#" * 78)
print("PART B. RISK")
print("#" * 78)

stream = pd.read_parquet(MODELS / "risk_stream.parquet")
feats = pd.read_parquet(MODELS / "risk_features.parquet")
scores = np.load(MODELS / "risk_scores.npy")
day = stream["ts"].values / DAY_SECONDS
truth = stream["is_fraud"].values
X_risk = feats[rd.FEATURES]

deployed = (day >= RISK_OOT_START) & (day < RISK_DRIFT_DAY)   # the model was live here
future = day >= RISK_DRIFT_DAY                                # and is judged here

actions = np.array([risk_policy.decide(s) for s in scores])
blocked = (actions == "block") & deployed
reviewed = (actions == "review") & deployed
allowed = deployed & ~blocked

print(f"the model was live over days {RISK_OOT_START}-{RISK_DRIFT_DAY}: "
      f"{int(deployed.sum()):,} transactions")
print(f"  blocked: {int(blocked.sum()):,}   of which really fraud: "
      f"{int(truth[blocked].sum()):,}")
print(f"  allowed: {int(allowed.sum()):,}")
print("\nblocking is also a decision not to find out. those "
      f"{int(blocked.sum()):,} transactions never")
print("completed, so no chargeback can ever arrive against them, and no label")
print("will ever exist. the fraud among them is not mislabelled; it is absent.")

lost = int(truth[blocked].sum())
kept = int(truth[allowed].sum())
print(f"\nfraud visible to the next retrain: {kept:,} of {kept + lost:,} "
      f"({kept / max(kept + lost, 1):.0%})")
print("the model is blindest on exactly the transactions it is most sure about.")

# -------------------------------------------------- the retrain options ---
print("\n" + "-" * 78)
print("B1. FOUR WAYS TO BUILD THE NEXT TRAINING SET")
print("-" * 78)

# An exploration slice: a small random share of would-be-blocked traffic is let
# through on purpose, so an outcome exists for it. Those rows are rare by
# construction, so they carry an inverse-propensity weight to stand in for the
# ones that stayed blocked.
EXPLORE = 0.02
explore_draw = rng.random(len(stream)) < EXPLORE
explored = blocked & explore_draw

# The review queue: an analyst looks at the review band and calls it, imperfectly.
REVIEWER_ERROR = 0.12
reviewer_wrong = rng.random(len(stream)) < REVIEWER_ERROR
review_label = np.where(reviewer_wrong, 1 - truth, truth)


def fit_and_score(mask, labels, weights=None, name=""):
    fitted = new_model().fit(X_risk[mask], labels[mask],
                             **({"sample_weight": weights[mask]} if weights is not None else {}))
    predicted = fitted.predict_proba(X_risk[future])[:, 1]
    return average_precision_score(truth[future], predicted), fitted


base_weight = np.ones(len(stream))
explore_weight = np.where(explored, 1.0 / EXPLORE, 1.0)

variants = {}
variants["naive: retrain on what was allowed"] = fit_and_score(allowed, truth)
variants["+ exploration slice (2%, IPW)"] = fit_and_score(
    allowed | explored, truth, explore_weight)
variants["+ review-queue labels"] = fit_and_score(
    allowed | explored | reviewed,
    np.where(reviewed, review_label, truth), explore_weight)
variants["oracle: every outcome observed"] = fit_and_score(deployed, truth)

print("each retrained on the deployment window, then scored on days "
      f"{RISK_DRIFT_DAY}-{RISK_DAYS}:\n")
naive_pr = variants["naive: retrain on what was allowed"][0]
oracle_pr = variants["oracle: every outcome observed"][0]
for name, (pr_auc, _) in variants.items():
    recovered = ""
    if oracle_pr > naive_pr and name not in (list(variants)[0], list(variants)[-1]):
        recovered = f"   recovers {(pr_auc - naive_pr) / (oracle_pr - naive_pr):.0%} of the gap"
    print(f"  {name:<42} PR-AUC {pr_auc:.3f}{recovered}")

explore_pr = variants["+ exploration slice (2%, IPW)"][0]
review_pr = variants["+ review-queue labels"][0]
if review_pr < explore_pr:
    print("\nnote the third row. adding review-queue labels made things worse, not")
    print(f"better ({review_pr:.3f} against {explore_pr:.3f}), and that is the reviewers, not the queue.")
    print(f"they are wrong {REVIEWER_ERROR:.0%} of the time here, and a wrong label in the band")
    print("the model is least sure about does more damage than a missing one. use")
    print("review decisions as labels, carefully, and measure the reviewers first.")

print("\nthe naive retrain is the one that happens by default, because nothing")
print("about it looks wrong: the query runs, the rows come back, the model")
print(f"trains. it is {oracle_pr - naive_pr:.3f} PR-AUC behind what the same window could")
print(f"have produced, and it costs {EXPLORE:.0%} of blocked traffic to buy most of that")
print("back. that is the trade: a small, deliberate, budgeted amount of fraud")
print("let through, in exchange for a model that can still see next year.")
print("\none thing an exploration slice is not: a shadow challenger. a challenger")
print("in shadow mode scores everything and decides nothing, which makes it a")
print("good way to compare two models on the same traffic, and no help at all")
print("here. it inherits the champion's censored labels, because the champion")
print("is still the one deciding what completes.")

fig, ax = viz.new_axes(width=9.0)
names = list(variants)
values = [variants[n][0] for n in names]
colours = [viz.ORANGE, viz.BLUE, viz.AQUA, viz.INK_SOFT]
bars = ax.barh(range(len(names)), values, color=colours, height=0.55)
for i, value in enumerate(values):
    ax.text(value + 0.005, i, f"{value:.3f}", va="center", color=viz.INK, fontsize=9)
ax.set_yticks(range(len(names)))
ax.set_yticklabels([n.replace(": ", ":\n") for n in names], fontsize=8)
ax.invert_yaxis()
ax.grid(True, axis="x", color=viz.INK_SOFT, alpha=0.15, linewidth=0.8)
ax.grid(False, axis="y")
viz.finish(fig, ax, "The next model, by what its training set was allowed to see",
           "PR-AUC on the drift week", "", PLOTS / "risk_selective_labels.png")


# ------------------------------------------------------------- summary ----
summary = {
    "churn": {
        "randomised_test_months": list(CAMPAIGN_MONTHS),
        "targeting_month": TARGETING_MONTH,
        "eligible": int(eligible.sum()),
        "treated": int(treated.sum()),
        "control": int(control.sum()),
        "saved": int(saved.sum()),
        "woken": int(woken.sum()),
        "incremental_retention": float(lift),
        "incremental_retention_ci_half_width": float(1.96 * se),
        "pr_auc_next_model_contaminated": float(pr_contaminated),
        "pr_auc_next_model_clean": float(pr_clean),
        "persuadable_score_contaminated": float(score_contaminated.mean()),
        "persuadable_score_clean": float(score_clean.mean()),
        "targeting": uplift_table.to_dict("records"),
        "corr_score_with_true_effect": corr_score,
        "corr_uplift_with_true_effect": corr_uplift,
    },
    "risk": {
        "deployment_window": [RISK_OOT_START, RISK_DRIFT_DAY],
        "blocked": int(blocked.sum()),
        "fraud_censored_by_blocking": lost,
        "fraud_still_observable": kept,
        "exploration_share": EXPLORE,
        "reviewer_error_rate": REVIEWER_ERROR,
        "pr_auc": {name: float(pr) for name, (pr, _) in variants.items()},
    },
}
(METRICS / "feedback_metrics.json").write_text(json.dumps(summary, indent=2, default=float))
print(f"\nmetrics written: {METRICS.name}/feedback_metrics.json")
