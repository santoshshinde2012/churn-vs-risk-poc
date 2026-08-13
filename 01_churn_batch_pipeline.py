"""Churn: the slow path (batch).

The nightly job from the article, end to end:

  1. The label is a window you choose. Three windows, three different problems.
  2. Point-in-time features, and what it costs to get that wrong.
  3. Cold start: accounts with no history borrow a cohort prior.
  4. A calibrated model, validated on a later snapshot than it trained on.
  5. Metrics that survive imbalance, and a calibration check, because the
     decision layer's arithmetic is meaningless on uncalibrated scores.
  6. Score the book, protect the holdout, write the call list.
  7. Track the run, promote to @champion, log the batch to the inference table.
  8. Cross-check the whole story on a real open dataset (IBM Telco).

Run: python 01_churn_batch_pipeline.py
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score

from churnrisk import churn_data as cd
from churnrisk import holdout, registry, viz
from churnrisk import metrics as mx
from churnrisk.config import (
    CHURN_LABEL_WINDOW,
    CHURN_MIN_HISTORY,
    CHURN_SCORE_SNAPSHOT,
    CHURN_TRAIN_SNAPSHOTS,
    DATA,
    METRICS,
    MODELS,
    PLOTS,
    SEED,
    TABLES,
    ensure_dirs,
)
from churnrisk.inference_log import InferenceLog
from churnrisk.policy import load_policy

MODEL_NAME = "ml.churn.scorer"
ensure_dirs()

policy = load_policy("churn_policy_v1")
CALL_CAPACITY = int(policy.capacity["calls_per_night"])
HOLDOUT_SHARE = float(policy.holdout["share"])

print("=" * 74)
print("CHURN PIPELINE (batch, the olive branch)")
print("=" * 74)
print(f"policy: {policy.summary()}")


def new_model():
    """One recipe, used everywhere in this script so comparisons are fair.

    Isotonic calibration on top of the booster, because the decision layer
    reads the score as a probability and section 5 measures whether it is one.
    """
    return CalibratedClassifierCV(
        HistGradientBoostingClassifier(random_state=SEED, max_iter=200),
        method="isotonic", cv=3,
    )


# ==================================================================== 1 =====
# The label is a window you choose, not a fact
# ===========================================================================
print("\n" + "-" * 74)
print("1. THE LABEL IS A WINDOW YOU CHOOSE")
print("-" * 74)

eng = cd.generate_engagement()
print(f"simulated book: {len(eng):,} accounts over {eng.days} days of history")

window_rows = []
for window in (7, 14, 30, 60):
    labels = cd.build_labels(eng, CHURN_TRAIN_SNAPSHOTS[-1], window)
    came_back = cd.returned_after_window(eng, CHURN_TRAIN_SNAPSHOTS[-1], window).sum()
    churned = int(labels["churned"].sum())
    window_rows.append({
        "window_days": window,
        "labelled_churned": churned,
        "churn_rate": labels["churned"].mean(),
        "came_back_later": int(came_back),
        "false_alarm_share": came_back / max(churned, 1),
        "days_until_label_confirms": window,
    })

windows = pd.DataFrame(window_rows)
print(windows.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
print("\nread the false alarm column. a 7-day window calls "
      f"{windows.iloc[0]['false_alarm_share']:.0%} of its churners wrong: they were on")
print("holiday, not leaving. a 60-day window is far cleaner, but the label only")
print("confirms two months after the account went quiet, which is too late to")
print(f"call anyone. this pipeline uses {CHURN_LABEL_WINDOW} days. it is a trade, not a truth.")

fig, axes = viz.new_grid(2, width=11.0)
axes[0].bar(windows["window_days"].astype(str), windows["churn_rate"], color=viz.BLUE, width=0.55)
for i, v in enumerate(windows["churn_rate"]):
    axes[0].text(i, v + 0.002, f"{v:.1%}", ha="center", color=viz.INK, fontsize=9)
viz.label(axes[0], "The same book, four churn rates", "inactivity window (days)", "churn rate")
axes[1].bar(windows["window_days"].astype(str), windows["false_alarm_share"],
            color=viz.ORANGE, width=0.55)
for i, v in enumerate(windows["false_alarm_share"]):
    axes[1].text(i, v + 0.004, f"{v:.0%}", ha="center", color=viz.INK, fontsize=9)
viz.label(axes[1], "Share of 'churners' who came back", "inactivity window (days)",
          "mislabelled")
viz.save(fig, PLOTS / "churn_label_window.png")


# ==================================================================== 2 =====
# Point-in-time features, and the cost of getting it wrong
# ===========================================================================
print("\n" + "-" * 74)
print("2. POINT-IN-TIME FEATURES")
print("-" * 74)


def snapshot_frame(snapshot: int, window: int = CHURN_LABEL_WINDOW) -> pd.DataFrame:
    """Features as of `snapshot`, joined to labels that resolve after it."""
    features = cd.build_features(eng, snapshot, CHURN_MIN_HISTORY)
    labels = cd.build_labels(eng, snapshot, window)
    return features.merge(labels, on="account_id", how="inner")


train = pd.concat([snapshot_frame(s) for s in CHURN_TRAIN_SNAPSHOTS], ignore_index=True)
test = snapshot_frame(CHURN_SCORE_SNAPSHOT)

print(f"train snapshots {list(CHURN_TRAIN_SNAPSHOTS)}: {len(train):,} rows, "
      f"churn {train['churned'].mean():.1%}")
print(f"test snapshot   {CHURN_SCORE_SNAPSHOT}: {len(test):,} rows, "
      f"churn {test['churned'].mean():.1%}")
print("the split is by time, never at random. a random split would let the")
print("model learn from next month to predict last month.")

X_train, y_train = cd.design_matrix(train), train["churned"].values
X_test, y_test = cd.design_matrix(test), test["churned"].values

model = new_model().fit(X_train, y_train)
p_test = model.predict_proba(X_test)[:, 1]

# The leaky counterfactual. Not a strawman: this is what you get when the
# feature job runs on a lag and aggregates "the last 30 days" as of the day it
# happens to run, rather than as of the day each label was decided. Here it
# peeks half-way into the label window, and the population stays fixed at the
# label date so the two models answer the same question.
LEAK_DAYS = CHURN_LABEL_WINDOW // 2


def leaky_frame(snapshot: int) -> pd.DataFrame:
    features = cd.build_features(eng, snapshot + LEAK_DAYS, CHURN_MIN_HISTORY,
                                 book_snapshot=snapshot)
    return features.merge(cd.build_labels(eng, snapshot, CHURN_LABEL_WINDOW), on="account_id")


leaky_train = pd.concat([leaky_frame(s) for s in CHURN_TRAIN_SNAPSHOTS], ignore_index=True)
leaky_test = leaky_frame(CHURN_SCORE_SNAPSHOT)
leaky_model = new_model().fit(cd.design_matrix(leaky_train), leaky_train["churned"].values)

leaky_offline = average_precision_score(
    leaky_test["churned"].values, leaky_model.predict_proba(cd.design_matrix(leaky_test))[:, 1])
leaky_in_production = average_precision_score(
    y_test, leaky_model.predict_proba(X_test)[:, 1])
honest = average_precision_score(y_test, p_test)

print(f"\nPR-AUC of the leaky model, scored on leaky features:  {leaky_offline:.3f}  <- the demo")
print(f"PR-AUC of the leaky model, scored on real features:   {leaky_in_production:.3f}  <- production")
print(f"PR-AUC of the point-in-time model:                    {honest:.3f}")
print("the leaky model looks like the best thing anyone ever built, right up")
print("until it is asked a question about the future. that gap is the whole")
print("reason to use a feature store with as-of joins.")

fig, ax = viz.new_axes()
bars = ax.bar(["leaky model,\nleaky evaluation", "leaky model,\nreal features",
               "point-in-time model,\nreal features"],
              [leaky_offline, leaky_in_production, honest],
              color=[viz.INK_SOFT, viz.ORANGE, viz.BLUE], width=0.55)
for bar, value in zip(bars, (leaky_offline, leaky_in_production, honest)):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01, f"{value:.2f}",
            ha="center", color=viz.INK, fontsize=9)
ax.set_ylim(0, 1)
viz.finish(fig, ax, "What leakage buys you offline, and costs you in production",
           "", "PR-AUC", PLOTS / "churn_leakage.png")


# ==================================================================== 3 =====
# Cold start
# ===========================================================================
print("\n" + "-" * 74)
print("3. COLD START")
print("-" * 74)

cold = test["is_cold_start"] == 1
print(f"accounts under {CHURN_MIN_HISTORY} days old at scoring time: "
      f"{int(cold.sum()):,} of {len(test):,} ({cold.mean():.1%})")
if cold.any():
    cold_pr = (average_precision_score(y_test[cold], p_test[cold])
               if 0 < y_test[cold].sum() < cold.sum() else float("nan"))
    print(f"their churn rate:      {test.loc[cold, 'churned'].mean():.1%}")
    print(f"everyone else's:       {test.loc[~cold, 'churned'].mean():.1%}")
    print(f"PR-AUC on cold start:  {cold_pr:.3f}")
    print(f"PR-AUC on the rest:    {average_precision_score(y_test[~cold], p_test[~cold]):.3f}")
    print("the model is far weaker on the accounts it knows nothing about, and")
    print("that is the honest result. do not pretend a cold account has a")
    print("reliable score; route it by cohort rules until it has a history.")
print("their rolling windows are borrowed from their plan cohort, and a flag")
print("tells the model the numbers are not theirs. without the flag the model")
print("cannot tell a brand-new account from a departing one: both have zero")
print("actions in the last 90 days, for opposite reasons.")


# ==================================================================== 4 =====
# Metrics that survive imbalance, and a calibration check
# ===========================================================================
print("\n" + "-" * 74)
print("4. METRICS, AND WHETHER THE PROBABILITIES ARE REAL")
print("-" * 74)

head = mx.headline(y_test, p_test)
print(f"PR-AUC {head['pr_auc']:.3f}   (random = the base rate, {head['base_rate']:.3f}, "
      f"so {head['pr_auc_lift']:.1f}x)")
print(f"ROC-AUC {head['roc_auc']:.3f}   <- flattering: with a negative class this large, "
      "hundreds of\n            false positives barely move the false positive rate")
print(f"Brier {head['brier']:.4f}   ECE {head['ece']:.4f}")

ks = (100, CALL_CAPACITY, 400)
p_at_k = {k: mx.precision_at_k(y_test, p_test, k) for k in ks}
r_at_k = {k: mx.recall_at_k(y_test, p_test, k) for k in ks}
for k in ks:
    print(f"precision@{k:<4} {p_at_k[k]:.1%}   recall@{k:<4} {r_at_k[k]:.1%}   "
          f"lift {p_at_k[k] / head['base_rate']:.1f}x   (random: {head['base_rate']:.1%})")

# Calibration is asserted by everyone and measured by almost nobody. The
# decision layer divides dollars by probabilities, so it has to be measured.
raw = HistGradientBoostingClassifier(random_state=SEED, max_iter=200).fit(X_train, y_train)
p_raw = raw.predict_proba(X_test)[:, 1]
raw_ece = mx.expected_calibration_error(y_test, p_raw)
print(f"\nuncalibrated booster: Brier {mx.brier_score(y_test, p_raw):.4f}  ECE {raw_ece:.4f}")
print(f"isotonic calibration: Brier {head['brier']:.4f}  ECE {head['ece']:.4f}")
print("ECE is the average gap between a predicted probability and the observed")
print("frequency. the decision layer multiplies these numbers by dollars, so")
print("the gap is the error bar on every threshold in this repo.")

fig, axes = viz.new_grid(2, width=11.0)
for ax, (name, probs, colour) in zip(
        axes, [("isotonic calibration", p_test, viz.BLUE),
               ("raw booster", p_raw, viz.ORANGE)]):
    predicted, observed, counts = mx.reliability_curve(y_test, probs, n_bins=10)
    limit = max(predicted.max(), observed.max()) * 1.1 if len(predicted) else 1.0
    ax.plot([0, limit], [0, limit], color=viz.INK_SOFT, linestyle=":", linewidth=1.2)
    ax.plot(predicted, observed, marker="o", color=colour, linewidth=2, markersize=5)
    ax.set_xlim(0, limit)
    ax.set_ylim(0, limit)
    viz.label(ax, f"{name}  (ECE {mx.expected_calibration_error(y_test, probs):.3f})",
              "predicted probability", "observed churn rate")
viz.save(fig, PLOTS / "churn_calibration.png")

fig, ax = viz.new_axes()
bars = ax.bar([str(k) for k in ks], [p_at_k[k] for k in ks], color=viz.BLUE, width=0.55)
for bar, k in zip(bars, ks):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.008,
            f"{p_at_k[k]:.0%}", ha="center", color=viz.INK, fontsize=9)
ax.axhline(head["base_rate"], color=viz.ORANGE, linewidth=2)
ax.text(len(ks) - 0.55, head["base_rate"] + 0.008,
        f"calling at random {head['base_rate']:.1%}", ha="right",
        color=viz.INK_SOFT, fontsize=8)
viz.finish(fig, ax, "Churn model: share of real churners in the call list",
           "size of the call list (k)", "precision at k",
           PLOTS / "churn_precision_at_k.png")


# ==================================================================== 5 =====
# The nightly batch: score the book, protect the holdout, write the call list
# ===========================================================================
print("\n" + "-" * 74)
print("5. THE NIGHTLY BATCH")
print("-" * 74)


nightly = pd.DataFrame({
    "account_id": test["account_id"].values,
    "snapshot": CHURN_SCORE_SNAPSHOT,
    "churn_probability": p_test.round(4),
})
nightly["holdout"] = holdout.assign(nightly["account_id"], HOLDOUT_SHARE)
nightly["action"] = [policy.decide(p) for p in p_test]
nightly.loc[nightly["holdout"], "action"] = "holdout"
nightly = nightly.sort_values("churn_probability", ascending=False).reset_index(drop=True)

call_list = nightly[(~nightly["holdout"]) & (nightly["action"] == "call")].head(CALL_CAPACITY)
nightly.to_csv(TABLES / "churn_scores.csv", index=False)
call_list.to_csv(TABLES / "retention_call_list.csv", index=False)

held_high = int(nightly["holdout"].head(CALL_CAPACITY).sum())
print(f"scored {len(nightly):,} accounts on the day-{CHURN_SCORE_SNAPSHOT} book, "
      f"out of sample: the model never saw this snapshot")
print(f"holdout: {int(nightly['holdout'].sum()):,} accounts "
      f"({held_high} of them are in the top {CALL_CAPACITY} by score)")
print(f"call list: {len(call_list)} accounts -> {TABLES.name}/retention_call_list.csv")
print("the holdout is excluded on purpose, and it hurts on purpose. those are")
print("real churners nobody will call. it is the only way to find out later")
print("whether calling anyone helped, and it keeps the next round of labels")
print("clean. 04_feedback_loops.py spends it.")

# Capacity, not the threshold, is what binds on this side.
worth_contacting = int((p_test >= policy.act_threshold).sum())
print(f"\naccounts above the cost-derived contact threshold (p > "
      f"{policy.act_threshold:.3f}): {worth_contacting:,}")
print(f"calls the team can actually make:                        {CALL_CAPACITY:,}")
print("that is the churn side in one line. losing a customer costs so much more")
print("than a phone call that the economics say contact almost everyone, and")
print("capacity is the real constraint. hence precision at k, where k is the team.")


# ==================================================================== 6 =====
# Track, promote, log
# ===========================================================================
print("\n" + "-" * 74)
print("6. TRACKING, PROMOTION AND THE INFERENCE LOG")
print("-" * 74)

version = registry.log_model(
    model, MODEL_NAME,
    params={"model": "HistGradientBoosting+isotonic", "train_snapshots": str(CHURN_TRAIN_SNAPSHOTS),
            "label_window_days": CHURN_LABEL_WINDOW, "min_history_days": CHURN_MIN_HISTORY,
            "n_train": len(train), "seed": SEED},
    metrics={"pr_auc": head["pr_auc"], "roc_auc": head["roc_auc"], "brier": head["brier"],
             "ece": head["ece"], f"precision_at_{CALL_CAPACITY}": p_at_k[CALL_CAPACITY]},
    input_example=X_train.head(3),
    tags={"path": "batch", "policy": policy.version},
)
promoted = registry.promote(MODEL_NAME, version, metric="pr_auc", value=head["pr_auc"])
print(f"  batch jobs load this as models:/{MODEL_NAME}@champion, so a promotion")
print("  changes what runs tonight without editing the job.")

log = InferenceLog("churn_batch", MODEL_NAME, version, policy.version)
for row in nightly.itertuples():
    log.record(entity_id=row.account_id, event_ts=CHURN_SCORE_SNAPSHOT,
               probability=row.churn_probability,
               model_action=policy.decide(row.churn_probability),
               final_action=row.action)
path = log.flush()
print(f"  inference log: {path.name} ({len(nightly):,} rows, model v{version}, "
      f"policy v{policy.version})")


# ==================================================================== 7 =====
# The same story on a real open dataset
# ===========================================================================
print("\n" + "-" * 74)
print("7. CROSS-CHECK ON REAL DATA (IBM Telco Customer Churn)")
print("-" * 74)

telco_path = DATA / "telco_churn.csv"
telco_summary = None
if telco_path.exists():
    from sklearn.model_selection import train_test_split

    telco = pd.read_csv(telco_path)
    telco["TotalCharges"] = pd.to_numeric(telco["TotalCharges"], errors="coerce").fillna(0.0)
    y_telco = (telco["Churn"] == "Yes").astype(int).values
    X_telco = pd.get_dummies(telco.drop(columns=["customerID", "Churn"]), drop_first=True)

    Xa, Xb, ya, yb = train_test_split(
        X_telco, y_telco, test_size=0.3, stratify=y_telco, random_state=SEED)
    telco_model = new_model().fit(Xa, ya)
    p_telco = telco_model.predict_proba(Xb)[:, 1]
    telco_head = mx.headline(yb, p_telco)
    telco_p200 = mx.precision_at_k(yb, p_telco, 200)
    telco_summary = {"pr_auc": telco_head["pr_auc"], "base_rate": telco_head["base_rate"],
                     "ece": telco_head["ece"], "precision_at_200": telco_p200}

    print(f"{len(telco):,} customers, churn rate {y_telco.mean():.1%}")
    print(f"PR-AUC {telco_head['pr_auc']:.3f} vs base rate {telco_head['base_rate']:.3f} "
          f"({telco_head['pr_auc'] / telco_head['base_rate']:.1f}x)   ECE {telco_head['ece']:.4f}")
    print(f"precision@200 {telco_p200:.1%} vs {telco_head['base_rate']:.1%} at random")
    print("same shape of result on data nobody simulated. what Telco cannot show")
    print("is anything above: it is a snapshot, so it has no label window, no")
    print("as-of join, no cold start and no holdout to measure against.")
else:
    print(f"skipped: {telco_path} not found (see README for the download link)")


# ---------------------------------------------------------------- metrics --
summary = {
    "label_windows": windows.to_dict("records"),
    "leakage": {"leaky_model_leaky_eval": float(leaky_offline),
                "leaky_model_real_features": float(leaky_in_production),
                "point_in_time_model": float(honest)},
    "test": {k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
             for k, v in head.items()},
    "precision_at_k": {str(k): float(v) for k, v in p_at_k.items()},
    "recall_at_k": {str(k): float(v) for k, v in r_at_k.items()},
    "calibration": {"isotonic_ece": float(head["ece"]), "raw_booster_ece": float(raw_ece)},
    "cold_start_share": float(cold.mean()),
    "holdout_accounts": int(nightly["holdout"].sum()),
    "call_list": len(call_list),
    "model_version": version,
    "promoted_to_champion": promoted,
    "policy_version": policy.version,
    "telco_cross_check": telco_summary,
}
(METRICS / "churn_metrics.json").write_text(json.dumps(summary, indent=2, default=float))
print(f"\nmetrics written: {METRICS.name}/churn_metrics.json")
print(f"tables written:  {TABLES.name}/churn_scores.csv, {TABLES.name}/retention_call_list.csv")

# The engagement history is reused by scripts 04 and 05, and regenerating it is
# cheap but not free. Cache the pieces they need.
np.savez_compressed(MODELS / "churn_engagement.npz",
                    actions=eng.actions, tickets=eng.tickets, negative=eng.negative)
eng.accounts.to_parquet(MODELS / "churn_accounts.parquet", index=False)
test.to_parquet(MODELS / "churn_test_snapshot.parquet", index=False)
nightly.to_parquet(MODELS / "churn_nightly.parquet", index=False)
print(f"state saved:     {MODELS.name}/churn_*.{{npz,parquet}} (scripts 04 and 05 reuse it)")
