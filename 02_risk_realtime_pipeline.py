"""Risk: the fast path (real time).

The article's claim is that the risk side is a different job, and the thing
that makes it different is not the algorithm. It is that the labels arrive
months late, the adversary changes tactics in days, and a false positive costs
more than the fraud it prevented.

This script builds the whole path against those three facts:

  1. Label maturation. What the pipeline can actually see today, and what
     training on the labels you happen to have does to the model.
  2. Point-in-time features in one pass, including network features, because
     fraud comes in rings and per-user features cannot see a ring.
  3. Out-of-time validation against the shuffled split that flatters it.
  4. Calibration, measured, because the threshold is arithmetic on probabilities.
  5. A threshold in dollars, from a versioned policy file, not 0.5.
  6. Four decision bands, plus plain rules running beside the model to cover
     the window between a tactic change and the retrain that learns it.
  7. Cold start, where the highest-fraud segment has the least history.

Run: python 02_risk_realtime_pipeline.py
"""

from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split

from churnrisk import metrics as mx
from churnrisk import registry, viz
from churnrisk import risk_data as rd
from churnrisk.config import (
    DAY_SECONDS,
    METRICS,
    MODELS,
    PLOTS,
    RISK_DAYS,
    RISK_DRIFT_DAY,
    RISK_MATURATION_DAYS,
    RISK_NEW_ACCOUNT_DAYS,
    RISK_OOT_START,
    SEED,
    TABLES,
    ensure_dirs,
)
from churnrisk.inference_log import InferenceLog
from churnrisk.policy import load_policy
from churnrisk.rules import RuleEngine, build_blocklist

MODEL_NAME = "ml.risk.scorer"
TODAY = RISK_DAYS
ensure_dirs()

policy = load_policy("risk_policy_v1")

print("=" * 74)
print("RISK PIPELINE (real time, the shield)")
print("=" * 74)
print(f"policy: {policy.summary()}")


def new_model():
    return CalibratedClassifierCV(
        HistGradientBoostingClassifier(random_state=SEED, max_iter=200),
        method="isotonic", cv=3,
    )


# ==================================================================== 1 =====
# The stream
# ===========================================================================
print("\n" + "-" * 74)
print("1. THE STREAM")
print("-" * 74)

stream = rd.generate_stream()
day = stream["ts"].values / DAY_SECONDS
truth = stream["is_fraud"].values                     # only the simulator knows this

print(f"transactions: {len(stream):,} over {RISK_DAYS} days   "
      f"true fraud rate: {truth.mean():.2%}")
print(stream.groupby("pattern").size().to_string())
print(f"\nthe card_testing rows start on day {RISK_DRIFT_DAY}: same accounts, "
      "completely different\nbehaviour. that is the adversary, not a data bug.")


# ==================================================================== 2 =====
# Label maturation: what you can actually see today
# ===========================================================================
print("\n" + "-" * 74)
print("2. LABEL MATURATION")
print("-" * 74)

curve = rd.maturation_curve(stream)
observed = rd.label_as_of(stream, TODAY)              # what the pipeline sees today
matured = rd.matured_mask(stream, TODAY, RISK_MATURATION_DAYS)

for checkpoint in (30, 60, 90, 120):
    share = curve.loc[curve["days_elapsed"] == checkpoint, "share_confirmed"].iloc[0]
    print(f"  {share:5.1%} of chargebacks have landed {checkpoint:>3} days after the transaction")

print(f"\ntoday is day {TODAY}. of {int(truth.sum()):,} transactions that really were fraud, "
      f"{int(observed.sum()):,}\nhave a chargeback on file. the other "
      f"{int(truth.sum() - observed.sum()):,} read as legitimate right now.")
print(f"\nrows old enough to trust ({RISK_MATURATION_DAYS}-day maturation): "
      f"{int(matured.sum()):,} of {len(stream):,} "
      f"({matured.mean():.0%}), everything up to day {RISK_OOT_START}.")
print(f"days {RISK_OOT_START} to {TODAY} are the blind window: partially labelled, and biased")
print("one way, because every label still in flight is a positive.")

fig, ax = viz.new_axes(width=7.2)
ax.plot(curve["days_elapsed"], curve["share_confirmed"], color=viz.BLUE, linewidth=2)
ax.axvline(RISK_MATURATION_DAYS, color=viz.ORANGE, linestyle="--", linewidth=2)
ax.text(RISK_MATURATION_DAYS - 2, 0.35, f"maturation cutoff\nday {RISK_MATURATION_DAYS}",
        ha="right", color=viz.INK, fontsize=8)
ax.axvspan(0, RISK_MATURATION_DAYS, color=viz.ORANGE, alpha=0.07)
ax.set_ylim(0, 1.02)
viz.finish(ax.figure, ax, "Share of fraud labels that have arrived",
           "days since the transaction", "chargebacks on file",
           PLOTS / "risk_label_maturation.png")

# --- three models, one honest question ------------------------------------
# A: only matured rows, true labels. The model you could actually build today.
# B: all pre-drift rows with the labels as they look today. What a pipeline
#    does when nobody thinks about maturation.
# C: the same rows with labels from the future. Impossible, and the only way
#    to separate "stale" from "mislabelled".
train_a = day < RISK_OOT_START
train_bc = day < RISK_DRIFT_DAY
eval_drift = day >= RISK_DRIFT_DAY
eval_oot = (day >= RISK_OOT_START) & (day < RISK_DRIFT_DAY)

print("\ncomputing point-in-time features (one pass over the stream)...")
feats = rd.build_features(stream)
X = feats[rd.FEATURES]

model = new_model().fit(X[train_a], truth[train_a])                  # A, the shipped model
naive = new_model().fit(X[train_bc], observed[train_bc])             # B
oracle = new_model().fit(X[train_bc], truth[train_bc])               # C

p = model.predict_proba(X)[:, 1]
p_naive = naive.predict_proba(X)[:, 1]
p_oracle = oracle.predict_proba(X)[:, 1]

pr_a = average_precision_score(truth[eval_drift], p[eval_drift])
pr_b = average_precision_score(truth[eval_drift], p_naive[eval_drift])
pr_c = average_precision_score(truth[eval_drift], p_oracle[eval_drift])

print(f"\nPR-AUC on days {RISK_DRIFT_DAY}-{RISK_DAYS}, scored against what really happened:")
print(f"  A  matured labels only, {int(train_a.sum()):>6,} rows to day {RISK_OOT_START}:  {pr_a:.3f}")
print(f"  B  all rows to day {RISK_DRIFT_DAY}, labels as they look today: {pr_b:.3f}")
print(f"  C  all rows to day {RISK_DRIFT_DAY}, labels from the future:    {pr_c:.3f}   (impossible)")
print(f"\nB minus C is the cost of the missing labels alone: {pr_b - pr_c:+.3f}. same rows,")
print("same recipe, same amount of data. the only difference is that B trained")
print(f"on {int((truth[train_bc] == 1).sum() - (observed[train_bc] == 1).sum()):,} frauds "
      "labelled as legitimate, because their chargebacks had not")
print("landed yet. that is the tax for ignoring maturation.")
print(f"\nA minus C is the cost of being stale: {pr_a - pr_c:+.3f}. A is the honest choice")
print(f"and it is still {RISK_MATURATION_DAYS} days behind the attack. no drift chart fixes that;")
print("only rules and fast signals cover the gap. section 6 measures them.")

fig, ax = viz.new_axes(width=7.6)
names = [f"A: matured only\n(to day {RISK_OOT_START})",
         "B: all rows,\nlabels as of today", "C: all rows,\nfuture labels"]
values = [pr_a, pr_b, pr_c]
bars = ax.bar(names, values, color=[viz.BLUE, viz.ORANGE, viz.INK_SOFT], width=0.55)
for bar, value in zip(bars, values):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01, f"{value:.2f}",
            ha="center", color=viz.INK, fontsize=9)
ax.set_ylim(0, max(values) * 1.25)
viz.finish(fig, ax, "Three training sets, scored on the drift week", "",
           "PR-AUC", PLOTS / "risk_label_maturation_cost.png")


# ==================================================================== 3 =====
# Out-of-time validation vs the shuffled split
# ===========================================================================
print("\n" + "-" * 74)
print("3. VALIDATION: OUT OF TIME, NEVER SHUFFLED")
print("-" * 74)

# To compare fairly, both models have to be asked about the same rows. So the
# shuffled model is trained on a random 60% of everything - which quietly
# includes most of the drift week - and then both are scored on the drift-week
# rows it was not trained on.
index = np.arange(len(X))
idx_train, idx_test = train_test_split(
    index, test_size=0.4, stratify=truth, random_state=SEED)
shuffled = new_model().fit(X.iloc[idx_train], truth[idx_train])

held_out_drift = idx_test[eval_drift[idx_test]]
pr_shuffled = average_precision_score(
    truth[held_out_drift], shuffled.predict_proba(X.iloc[held_out_drift])[:, 1])
pr_a_same_rows = average_precision_score(truth[held_out_drift], p[held_out_drift])
pr_oot = average_precision_score(truth[eval_oot], p[eval_oot])

print(f"out of time, days {RISK_OOT_START}-{RISK_DRIFT_DAY} (honest):   {pr_oot:.3f}")
print(f"out of time, drift week {RISK_DRIFT_DAY}-{RISK_DAYS}:     {pr_a:.3f}")
print(f"\nboth models, scored on the same {len(held_out_drift):,} held-out drift-week rows:")
print(f"  shuffled split, trained on a random 60% of everything: {pr_shuffled:.3f}")
print(f"  time split, trained on matured data to day {RISK_OOT_START}:      {pr_a_same_rows:.3f}")
print(f"\nthe shuffled model looks {pr_shuffled - pr_a_same_rows:+.3f} better and it has learned "
      "nothing extra.")
print("its training set contained 60% of the card-testing bursts, so it is")
print("being asked to recognise the second half of an attack whose first half")
print("it memorised. that is not generalisation, and there is no version of")
print("production where that data was available.")

fig, ax = viz.new_axes(width=8.0)
names = ["shuffled split\n(flattering)", f"days {RISK_OOT_START}-{RISK_DRIFT_DAY}\n(honest)",
         f"drift week {RISK_DRIFT_DAY}-{RISK_DAYS}\n(tactics changed)"]
values = [pr_shuffled, pr_oot, pr_a]
bars = ax.bar(names, values, color=[viz.INK_SOFT, viz.BLUE, viz.ORANGE], width=0.55)
for bar, value in zip(bars, values):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.015, f"{value:.2f}",
            ha="center", color=viz.INK, fontsize=9)
ax.set_ylim(0, 1)
viz.finish(fig, ax, "Fraud model: the same model, three views of reality", "",
           "PR-AUC", PLOTS / "risk_validation_gap.png")


# ==================================================================== 4 =====
# Calibration, measured
# ===========================================================================
print("\n" + "-" * 74)
print("4. CALIBRATION")
print("-" * 74)

cal_oot = mx.headline(truth[eval_oot], p[eval_oot])
cal_drift = mx.headline(truth[eval_drift], p[eval_drift])
print(f"days {RISK_OOT_START}-{RISK_DRIFT_DAY}:   Brier {cal_oot['brier']:.5f}   "
      f"ECE {cal_oot['ece']:.5f}")
print(f"drift week:  Brier {cal_drift['brier']:.5f}   ECE {cal_drift['ece']:.5f}")
print("\ncalibration is fitted on the training window and it decays with"
      "\neverything else. the block threshold is arithmetic on these"
      "\nprobabilities, so when calibration drifts the threshold quietly stops"
      "\nmeaning what the cost model says it means. this is the number to alarm"
      "\non, and it needs no labels at inference time, only at review time.")

fig, axes = viz.new_grid(2, width=11.0)
for ax, (name, mask) in zip(axes, [(f"days {RISK_OOT_START}-{RISK_DRIFT_DAY}", eval_oot),
                                   (f"drift week {RISK_DRIFT_DAY}-{RISK_DAYS}", eval_drift)]):
    predicted, observed_rate, _ = mx.reliability_curve(truth[mask], p[mask], n_bins=8)
    limit = max(predicted.max(), observed_rate.max()) * 1.15 if len(predicted) else 1.0
    ax.plot([0, limit], [0, limit], color=viz.INK_SOFT, linestyle=":", linewidth=1.2)
    ax.plot(predicted, observed_rate, marker="o", color=viz.BLUE, linewidth=2, markersize=5)
    ax.set_xlim(0, limit)
    ax.set_ylim(0, limit)
    viz.label(ax, f"{name}  (ECE {mx.expected_calibration_error(truth[mask], p[mask]):.4f})",
              "predicted probability", "observed fraud rate")
viz.save(fig, PLOTS / "risk_calibration.png")


# ==================================================================== 5 =====
# The threshold is a cost calculation
# ===========================================================================
print("\n" + "-" * 74)
print("5. THE THRESHOLD IS IN DOLLARS")
print("-" * 74)

fp_cost = policy.costs["false_positive"]
fn_cost = policy.costs["false_negative"]
block_at = policy.block_at

p_eval, y_eval = p[eval_oot], truth[eval_oot]
thresholds, costs, empirical_best = mx.cost_curve(y_eval, p_eval, fp_cost, fn_cost)

cost_derived = mx.total_cost(y_eval, p_eval, block_at, fp_cost, fn_cost)
cost_half = mx.total_cost(y_eval, p_eval, 0.50, fp_cost, fn_cost)
cost_best = float(costs.min())

print(f"from the policy file: block when p > {block_at:.2f}   "
      f"(FP ${fp_cost} vs FN ${fn_cost})")
print(f"  cost at the derived threshold {block_at:.2f}:  ${cost_derived:>10,.0f}")
print(f"  cost at the default           0.50:  ${cost_half:>10,.0f}")
print(f"  cost at the measured minimum  {empirical_best:.2f}:  ${cost_best:>10,.0f}")
print("\nthe measured minimum beats the default by "
      f"${cost_half - cost_best:,.0f} ({1 - cost_best / cost_half:.0%}). whatever else is")
print("true, 0.5 is not the answer to a question nobody asked it.")

# Where the derived threshold and the measured minimum disagree, the scores
# are the suspect, not the cost model. Global ECE will not show it: at this
# imbalance almost every row sits at p near zero and drowns the bins that
# actually decide anything.
region = p_eval >= 0.05
ece_all = mx.expected_calibration_error(y_eval, p_eval)
ece_region = mx.expected_calibration_error(y_eval[region], p_eval[region])
band = (p_eval >= min(empirical_best, block_at)) & (p_eval < max(empirical_best, block_at))

print(f"\nderived {block_at:.2f} vs measured {empirical_best:.2f}. the gap is a calibration "
      "story:")
print(f"  ECE over everything:            {ece_all:.4f}   <- looks perfect")
print(f"  ECE over the decision region (p >= 0.05, {int(region.sum()):,} rows): "
      f"{ece_region:.4f}")
if band.sum():
    print(f"  rows scored between {min(empirical_best, block_at):.2f} and "
          f"{max(empirical_best, block_at):.2f}: {int(band.sum()):,}, "
          f"and {y_eval[band].mean():.0%} of them really were fraud")
print("\nthat last line is the whole thing. the model says those rows are around")
print(f"{p_eval[band].mean():.0%} likely to be fraud and they are actually "
      f"{y_eval[band].mean():.0%} fraud, so the")
print("arithmetic that produced 0.77 was fed a number that was not true. the")
print("global ECE never saw it, because it is an average over a population that")
print("is 99% easy negatives.")
print("\nso: derive the threshold from costs to start, then recompute it from the")
print("inference log against measured outcomes, and re-check calibration inside")
print("the decision region rather than over the whole score range.")

fig, ax = viz.new_axes(width=7.6)
ax.plot(thresholds, costs / 1000, color=viz.BLUE, linewidth=2)
ax.axvline(0.5, color=viz.INK_SOFT, linewidth=1.5, linestyle=":")
ax.text(0.5, ax.get_ylim()[1] * 0.94, " default 0.5", color=viz.INK_SOFT, fontsize=8)
ax.axvline(block_at, color=viz.ORANGE, linewidth=2, linestyle="--")
ax.text(block_at, ax.get_ylim()[1] * 0.82, f" policy threshold {block_at:.2f}",
        color=viz.INK, fontsize=8)
viz.finish(fig, ax, "Total cost of mistakes by block threshold (days "
           f"{RISK_OOT_START}-{RISK_DRIFT_DAY})",
           "block threshold", "total cost ($ thousands)", PLOTS / "risk_cost_curve.png")


# ==================================================================== 6 =====
# Bands, and the rules that run beside the model
# ===========================================================================
print("\n" + "-" * 74)
print("6. FOUR BANDS, AND THE RULES BESIDE THEM")
print("-" * 74)

# The blocklist is built from the matured window only, so it carries no
# knowledge the pipeline could not have had on the day.
records_train = stream[train_a].to_dict("records")
blocked_devices = build_blocklist(records_train, truth[train_a])
engine = RuleEngine(policy, blocked_devices)
print(f"blocklist built from matured data only: {len(blocked_devices)} devices")
print("devices, not cards. blocklisting a compromised card punishes the victim")
print("for having been defrauded; the device stays with the attacker.")


def run_decisions(mask, probabilities):
    """Score a window with the model, then let the rules escalate."""
    window = stream[mask]
    features = feats[mask]
    model_actions, final_actions, rules_actions, rule_names = [], [], [], []

    columns = ["device", "card", "amount", "account_age_days"]
    base = window[columns].to_dict("records")
    extra = features[["txn_60s", "txn_10min", "device_accounts"]].to_dict("records")

    for probability, left, right in zip(probabilities, base, extra):
        record = {**left, **right}
        model_action = policy.decide(probability)
        final_action, hits = engine.apply(model_action, record)
        # The same rules with the model switched off, so their standalone
        # contribution can be measured rather than assumed.
        rules_only, _ = engine.apply("allow", record)
        model_actions.append(model_action)
        final_actions.append(final_action)
        rules_actions.append(rules_only)
        rule_names.append(",".join(hit.rule for hit in hits))
    return pd.DataFrame({
        "model_action": model_actions, "final_action": final_actions,
        "rules_action": rules_actions,
        "rules": rule_names, "probability": probabilities,
        "is_fraud": truth[mask], "day": day[mask],
        "account": window["account"].values, "pattern": window["pattern"].values,
    })


def stopped(frame, actions=("block", "review", "challenge")):
    fraud = frame[frame["is_fraud"] == 1]
    if len(fraud) == 0:
        return float("nan")
    return float(fraud["final_action"].isin(actions).mean())


def insult(frame):
    legit = frame[frame["is_fraud"] == 0]
    return float((legit["final_action"] == "block").mean())


windows = {f"days {RISK_OOT_START}-{RISK_DRIFT_DAY}": eval_oot,
           f"drift week {RISK_DRIFT_DAY}-{RISK_DAYS}": eval_drift}
band_rows = []
decisions = {}
for name, mask in windows.items():
    frame = run_decisions(mask, p[mask])
    decisions[name] = frame

    model_only = frame.assign(final_action=frame["model_action"])
    # What the rules would catch on their own, with the model switched off.
    rules_alone = frame.assign(final_action=frame["rules_action"])
    band_rows.append({
        "window": name,
        "fraud_blocked_by_model": float(
            (model_only[model_only["is_fraud"] == 1]["final_action"] == "block").mean()),
        "fraud_touched_model_only": stopped(model_only),
        "fraud_touched_rules_only": stopped(rules_alone),
        "fraud_touched_together": stopped(frame),
        "insult_model_only": insult(model_only),
        "insult_together": insult(frame),
    })

bands = pd.DataFrame(band_rows)
print("\n" + bands.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

for name in windows:
    frame = decisions[name]
    print(f"\ndecision mix, {name} (rows: final action, columns: 0 legit / 1 fraud)")
    print(pd.crosstab(frame["final_action"], frame["is_fraud"]).to_string())

drift_frame = decisions[f"drift week {RISK_DRIFT_DAY}-{RISK_DAYS}"]
drift_fraud = drift_frame[drift_frame["is_fraud"] == 1]
rules_alone_recall = float((drift_fraud["rules_action"] != "allow").mean())
incremental = drift_fraud[(drift_fraud["model_action"] == "allow")
                          & (drift_fraud["final_action"] != "allow")]

print("\non the drift week, with no model at all, the rules alone would touch "
      f"{rules_alone_recall:.0%}\nof the fraud. they add {len(incremental):,} catches "
      "on top of a working model.")
print(drift_fraud["rules"].replace("", "no rule fired").value_counts().to_string())
print("\nthe incremental number is small, and that is the honest result: a model")
print("that is working catches most of what a velocity limit catches. the case")
print("for the rules is the other two columns. they need no labels, so they do")
print(f"not care that training data stops on day {RISK_OOT_START}, and they need no endpoint,")
print("so they keep working when scoring does not. 03_live_scoring_demo.py runs")
print("that second case: the rules are what stands between a scoring outage and")
print("waving every transaction through.")

fig, ax = viz.new_axes(width=8.6)
labels = list(windows)
x = np.arange(len(labels))
for offset, column, colour, name in (
        (-0.26, "fraud_touched_model_only", viz.BLUE, "model alone"),
        (0.0, "fraud_touched_rules_only", viz.ORANGE, "rules alone"),
        (0.26, "fraud_touched_together", viz.AQUA, "both")):
    ax.bar(x + offset, bands[column], width=0.24, color=colour, label=name)
    for i, value in enumerate(bands[column]):
        ax.text(i + offset, value + 0.015, f"{value:.0%}", ha="center",
                color=viz.INK, fontsize=8)
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylim(0, 1.12)
ax.legend(frameon=False, fontsize=9, labelcolor=viz.INK_SOFT, ncol=3)
viz.finish(fig, ax, "Fraud challenged, reviewed or blocked", "",
           "share of fraud touched", PLOTS / "risk_rules_coverage.png")


# ==================================================================== 7 =====
# Cold start on the risk side
# ===========================================================================
print("\n" + "-" * 74)
print("7. COLD START")
print("-" * 74)

new_account = feats["account_age_days"].values <= RISK_NEW_ACCOUNT_DAYS
scope = eval_oot | eval_drift
new_scope = new_account & scope
old_scope = ~new_account & scope

print(f"transactions from accounts under {RISK_NEW_ACCOUNT_DAYS} days old: "
      f"{int(new_scope.sum()):,} ({new_scope.sum() / scope.sum():.1%} of the window)")
print(f"  fraud rate on new accounts:   {truth[new_scope].mean():.2%}")
print(f"  fraud rate on aged accounts:  {truth[old_scope].mean():.2%}")
print(f"  ratio: {truth[new_scope].mean() / max(truth[old_scope].mean(), 1e-9):.1f}x")
print("the opposite of the churn side. there, a new account is simply unknown;")
print("here it is the highest-risk segment there is, and it is the segment with")
print("the least history to score on. so the compensating controls in the policy")
print("do the work: an amount ceiling, device and network signals, stricter")
print("defaults, until the account has a past worth reading.")


# ==================================================================== 8 =====
# Register, promote, log
# ===========================================================================
print("\n" + "-" * 74)
print("8. TRACKING, PROMOTION AND THE INFERENCE LOG")
print("-" * 74)

version = registry.log_model(
    model, MODEL_NAME,
    params={"model": "HistGradientBoosting+isotonic", "trained_through_day": RISK_OOT_START,
            "maturation_days": RISK_MATURATION_DAYS, "n_train": int(train_a.sum()),
            "features": len(rd.FEATURES), "seed": SEED},
    metrics={"pr_auc_oot": pr_oot, "pr_auc_drift": pr_a, "pr_auc_shuffled": pr_shuffled,
             "brier_oot": cal_oot["brier"], "ece_oot": cal_oot["ece"],
             "insult_rate": float(bands.iloc[0]["insult_together"])},
    input_example=X[train_a].head(3),
    tags={"path": "realtime", "policy": policy.version},
)
promoted = registry.promote(MODEL_NAME, version, metric="pr_auc_oot", value=pr_oot)

log = InferenceLog("risk_serving", MODEL_NAME, version, policy.version)
for frame in decisions.values():
    for row in frame.itertuples():
        log.record(entity_id=row.account, event_ts=row.day, probability=row.probability,
                   model_action=row.model_action, final_action=row.final_action,
                   rules_fired=row.rules.split(",") if row.rules else ())
path = log.flush()
print(f"  inference log: {path.name} "
      f"({sum(len(f) for f in decisions.values()):,} rows, no outcomes attached)")
print("  every row is logged with no outcome, on purpose. the outcome arrives")
print("  months later and gets joined in then. 04_feedback_loops.py picks it up")
print("  from here, which is where the selective label problem starts.")


# ------------------------------------------------------------- artifacts ---
joblib.dump({"model": model, "features": rd.FEATURES, "model_version": version,
             "blocked_devices": blocked_devices},
            MODELS / "risk_model.joblib")

live = feats[eval_drift].copy()
live["is_fraud"] = truth[eval_drift]
live["device"] = stream.loc[eval_drift, "device"].values
live["card"] = stream.loc[eval_drift, "card"].values
live["pattern"] = stream.loc[eval_drift, "pattern"].values
# Sampled, not taken from the top: the first rows of the drift week are all
# from the same few bursts, which would make the serving demo a story about
# one attack rather than about the window.
live = pd.concat([
    live[live["is_fraud"] == 1].sample(10, random_state=SEED),
    live[live["is_fraud"] == 0].sample(22, random_state=SEED),
]).sample(frac=1, random_state=SEED)
live.to_csv(TABLES / "live_sample.csv", index=False)

stream.to_parquet(MODELS / "risk_stream.parquet", index=False)
feats.to_parquet(MODELS / "risk_features.parquet", index=False)
np.save(MODELS / "risk_scores.npy", p)

summary = {
    "transactions": int(len(stream)),
    "true_fraud_rate": float(truth.mean()),
    "labels_visible_today": {"true_frauds": int(truth.sum()),
                             "chargebacks_on_file": int(observed.sum()),
                             "still_in_flight": int(truth.sum() - observed.sum())},
    "maturation": {"days": RISK_MATURATION_DAYS,
                   "trainable_share": float(matured.mean()),
                   "pr_auc_matured_only": float(pr_a),
                   "pr_auc_naive_labels": float(pr_b),
                   "pr_auc_oracle_labels": float(pr_c)},
    "validation": {"oot": float(pr_oot), "drift_week": float(pr_a),
                   "same_rows_shuffled_model": float(pr_shuffled),
                   "same_rows_time_split_model": float(pr_a_same_rows)},
    "calibration": {"oot_brier": cal_oot["brier"], "oot_ece": cal_oot["ece"],
                    "drift_brier": cal_drift["brier"], "drift_ece": cal_drift["ece"]},
    "threshold": {"policy_version": policy.version, "derived": float(block_at),
                  "ece_overall": float(ece_all), "ece_decision_region": float(ece_region),
                  "empirical_minimum": float(empirical_best),
                  "cost_at_derived": mx.total_cost(y_eval, p_eval, block_at, fp_cost, fn_cost),
                  "cost_at_default_half": mx.total_cost(y_eval, p_eval, 0.5, fp_cost, fn_cost)},
    "bands": bands.to_dict("records"),
    "rules": {"blocklisted_devices": len(blocked_devices),
              "rules_only_recall_drift_week": rules_alone_recall,
              "incremental_catches_drift_week": int(len(incremental))},
    "cold_start": {"new_account_fraud_rate": float(truth[new_scope].mean()),
                   "aged_account_fraud_rate": float(truth[old_scope].mean())},
    "model_version": version,
    "promoted_to_champion": promoted,
}
(METRICS / "risk_metrics.json").write_text(json.dumps(summary, indent=2, default=float))
print(f"\nmetrics written: {METRICS.name}/risk_metrics.json")
print(f"model saved:     {MODELS.name}/risk_model.joblib  (also registered as "
      f"{MODEL_NAME}@champion)")
print(f"live sample:     {TABLES.name}/live_sample.csv  (for 03_live_scoring_demo.py)")
