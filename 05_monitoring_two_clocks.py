"""Monitoring, on two different clocks.

Both pipelines get monitored. They do not get monitored at the same speed, and
the reason is not that fraud is more important. It is that fraud has an
adversary on the other side, and adversaries move faster than customers lose
interest.

There is a second reason, specific to the risk side and larger than the first:
a chargeback takes 30 to 90 days to arrive, so any monitor that waits for
labels reports on an attack a quarter after it started. The only monitoring
that can run daily is monitoring that needs no outcome at all - the shape of
the features, and the shape of the scores.

This script runs both:

  churn   monthly snapshots, PSI on features and scores. Slow, boring, and
          boring is the correct answer.
  risk    daily, same statistics, against the training window. Then it
          measures how many days earlier that fires than a label-based
          performance chart could have.

Run 01 and 02 first, then: python 05_monitoring_two_clocks.py
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from churnrisk import churn_data as cd
from churnrisk import drift, registry, viz
from churnrisk import risk_data as rd
from churnrisk.config import (
    CHURN_MIN_HISTORY,
    DAY_SECONDS,
    METRICS,
    MODELS,
    PLOTS,
    RISK_DAYS,
    RISK_DRIFT_DAY,
    RISK_MATURATION_DAYS,
    RISK_OOT_START,
    TABLES,
    ensure_dirs,
)
from churnrisk.policy import load_policy

ensure_dirs()
churn_policy = load_policy("churn_policy_v1")
risk_policy = load_policy("risk_policy_v1")

print("=" * 78)
print("MONITORING ON TWO CLOCKS")
print("=" * 78)


# ###########################################################################
# THE FAST CLOCK: RISK
# ###########################################################################
print("\n" + "-" * 78)
print("RISK: DAILY, NO LABELS REQUIRED  "
      f"(alert at PSI {risk_policy.monitoring['psi_alert']}, "
      f"cadence {risk_policy.monitoring['cadence_days']}d)")
print("-" * 78)

stream = pd.read_parquet(MODELS / "risk_stream.parquet")
feats = pd.read_parquet(MODELS / "risk_features.parquet")
scores = np.load(MODELS / "risk_scores.npy")
day = stream["ts"].values / DAY_SECONDS

# The reference is the window the model was trained on. Everything after is
# compared against it, which is the question a monitor is really asking:
# does today still look like the world this model was fitted to?
reference = day < RISK_OOT_START
monitor = drift.Monitor(feats.loc[reference, rd.FEATURES], rd.FEATURES,
                        score_ref=scores[reference])

alert_at = float(risk_policy.monitoring["psi_alert"])
watch_at = float(risk_policy.monitoring["psi_watch"])

# Two signals, deliberately. Population PSI answers "does the traffic still
# look like the training set". The flag rate answers "is the model saying yes
# more often than it used to". A card-testing wave is a fraction of a percent
# of the day's transactions, so it can move the second without touching the
# first, and the second is the one worth paging on.
# The two signals want two different baselines, and using one for both is the
# mistake that makes a flag-rate alarm useless.
#
# PSI asks "does the traffic still look like what this model was fitted to", so
# its reference is the training window.
#
# The flag rate asks "is the model saying yes more often than it normally
# does", and normal means recent production, not training. Scored against the
# training window instead, the alarm inherits every legitimate change since
# then as a permanent positive offset, and sits above the line forever.
challenge_at = risk_policy.bands["challenge"]
WINDOW_DAYS = 7             # a trailing week, because one day is mostly noise
BASELINE_DAYS = 28          # the four weeks before that week
MONITOR_FROM = RISK_OOT_START + WINDOW_DAYS + BASELINE_DAYS
Z_ALERT = 3.0               # standard errors above the rolling baseline

training_flag_rate = drift.flag_rate(scores[reference], challenge_at)
print(f"flag rate on the training window: {training_flag_rate:.3%}")
print(f"alarm: last {WINDOW_DAYS} days against the {BASELINE_DAYS} days before them, "
      f"{Z_ALERT:.0f} standard errors")

flagged_today = scores >= challenge_at
daily_rows = []
for d in range(MONITOR_FROM, RISK_DAYS):
    today = (day >= d) & (day < d + 1)
    if today.sum() < 200:                     # too little traffic to say anything
        continue
    window = (day > d - WINDOW_DAYS) & (day < d + 1)
    # A rolling reference rather than a fixed one. The pre-attack flag rate on
    # this stream is not stationary - fraud rings arrive in bursts - so any
    # fixed baseline is a guess about which quiet fortnight was the real one.
    # Comparing a week against the four weeks before it asks the question a
    # monitor should ask: has something changed recently, not has anything
    # changed since a date somebody picked.
    baseline = (day > d - WINDOW_DAYS - BASELINE_DAYS) & (day <= d - WINDOW_DAYS)
    baseline_rate = drift.flag_rate(scores[baseline], challenge_at)

    report = monitor.check(feats.loc[today, rd.FEATURES], scores[today])
    name, worst_psi = drift.worst(report)
    daily_rows.append({
        "day": d,
        "n": int(today.sum()),
        "worst_signal": "score" if name == "__score__" else name,
        "worst_psi": worst_psi,
        "score_psi": report["__score__"]["psi"],
        "flag_rate_1d": drift.flag_rate(scores[today], challenge_at),
        "flag_rate_7d": drift.flag_rate(scores[window], challenge_at),
        "z_1d": drift.rate_z_score(int(flagged_today[today].sum()),
                                   int(today.sum()), baseline_rate),
        "baseline_rate": baseline_rate,
        "z_7d": drift.rate_z_score(int(flagged_today[window].sum()),
                                   int(window.sum()), baseline_rate),
        # The same statistic against the training window, to show what that
        # choice costs. It is not a straw man; it is the obvious thing to do.
        "z_7d_vs_training": drift.rate_z_score(int(flagged_today[window].sum()),
                                               int(window.sum()), training_flag_rate),
        "psi_verdict": drift.verdict(report, alert_at, watch_at),
    })

daily = pd.DataFrame(daily_rows)
daily["flag_alert"] = daily["z_7d"] >= Z_ALERT
daily["naive_alert"] = daily["flag_rate_1d"] >= 2 * daily["baseline_rate"]
daily["training_baseline_alert"] = daily["z_7d_vs_training"] >= Z_ALERT
daily["verdict"] = np.where(
    daily["flag_alert"] | (daily["psi_verdict"] == "alert"), "alert", daily["psi_verdict"])

# The trailing window needs to be full before it means anything.
warm = daily
before, after = warm[warm["day"] < RISK_DRIFT_DAY], warm[warm["day"] >= RISK_DRIFT_DAY]

psi_alerts = after[after["psi_verdict"] == "alert"]["day"]
flag_alerts = after[after["flag_alert"]]["day"]
first_psi = int(psi_alerts.min()) if len(psi_alerts) else None
first_flag = int(flag_alerts.min()) if len(flag_alerts) else None
first_alert = min([d for d in (first_psi, first_flag) if d is not None], default=None)

false_alarms_naive = int(before["naive_alert"].sum())
false_alarms_z = int(before["flag_alert"].sum())
false_alarms_training = int(before["training_baseline_alert"].sum())

print(f"\ndays monitored: {len(warm)}   (day {warm['day'].min()} to {warm['day'].max()})")
print(f"{'':30} {'before the change':>18} {'after':>10}")
print(f"{'median worst PSI':30} {before['worst_psi'].median():>18.3f} "
      f"{after['worst_psi'].median():>10.3f}")
print(f"{'median 7-day flag rate':30} {before['flag_rate_7d'].median():>18.3%} "
      f"{after['flag_rate_7d'].median():>10.3%}")
print(f"{'alerting days, PSI':30} "
      f"{int((before['psi_verdict'] == 'alert').sum()):>18} "
      f"{int((after['psi_verdict'] == 'alert').sum()):>10}")
print(f"{'alerting days, 7-day z-score':30} {false_alarms_z:>18} "
      f"{int(after['flag_alert'].sum()):>10}")
print(f"{'alerting days, naive 2x rule':30} {false_alarms_naive:>18} "
      f"{int(after['naive_alert'].sum()):>10}")
print(f"{'  same z, training baseline':30} {false_alarms_training:>18} "
      f"{int(after['training_baseline_alert'].sum()):>10}")

print("\nthe last row is the identical statistic against a fixed reference - the")
print(f"training window's {training_flag_rate:.3%} - rather than a rolling one. it alerts on more")
print(f"than twice as many days after the change ({int(after['training_baseline_alert'].sum())} against "
      f"{int(after['flag_alert'].sum())}), and with only")
print(f"{len(before)} quiet days here this window cannot tell you whether that is more")
print("sensitivity or more noise. it is worth knowing which before shipping it.")
print("\nthe rolling reference is the safer default anyway, because it does not")
print("assume the flag rate is stationary, and on this stream it plainly is not:")
print(f"it ranges from {warm['flag_rate_7d'].min():.3%} to {warm['flag_rate_7d'].max():.3%} across the monitoring window, with fraud")
print("rings arriving in bursts. a fixed reference bakes one arbitrary fortnight")
print("in as the definition of normal.")
print(f"\nas configured, the alarm fires on {false_alarms_z} of {len(before)} quiet days and on "
      f"{int(after['flag_alert'].sum())} of the {len(after)} after.")

print("\nfirst alert from population PSI:      "
      f"{'day ' + str(first_psi) if first_psi else 'never fired'}")
print("first alert from the 7-day flag rate: "
      f"{'day ' + str(first_flag) if first_flag else 'never fired'}")

if first_psi is None:
    wave_share = float((stream.loc[day >= RISK_DRIFT_DAY, "pattern"] == "card_testing").mean())
    print("\nthe population PSI never fires, and that is not a bug in the monitor.")
    print(f"the card-testing wave is {wave_share:.2%} of the transactions in that window. a")
    print("statistic computed over the whole population cannot see a change that")
    print("small, and lowering the threshold until it does would produce an alert")
    print("every week for reasons that are never attacks.")
    print("\nthe flag rate sees it because the attack is concentrated exactly where")
    print("the flag rate looks: the top of the score distribution.")
if first_alert is not None:
    print(f"\nfirst alert of any kind: day {first_alert}, "
          f"{first_alert - RISK_DRIFT_DAY} day(s) after the tactics changed.")
    print("\nthe first alerting week:")
    print(after[after["verdict"] == "alert"].head(7)
          [["day", "n", "worst_psi", "flag_rate_7d", "z_7d"]]
          .to_string(index=False, float_format=lambda v: f"{v:.3f}"))

# The comparison that matters: how long a label-based chart would have waited.
label_based_day = RISK_DRIFT_DAY + RISK_MATURATION_DAYS
lag = label_based_day - (first_alert if first_alert is not None else RISK_DAYS)
print("\nthe same change, seen through labels:")
print(f"  tactic changes                      day {RISK_DRIFT_DAY}")
print(f"  distribution monitor alerts         day {first_alert}")
print(f"  chargebacks mature enough to score  day {label_based_day}")
print(f"\nthat is a {lag}-day head start, and it is the whole argument for monitoring")
print("inputs rather than outcomes on this side. a PR-AUC chart is the right")
print("way to grade a model and the wrong way to notice an attack: by the time")
print("it moves, the attack is a quarter old and has already been paid for.")
print("\nthe distribution monitor cannot tell you the model got worse. it can only")
print("tell you the traffic stopped looking like the training set, which is a")
print("weaker claim and an earlier one. on this side, earlier wins.")


# ###########################################################################
# THE SLOW CLOCK: CHURN
# ###########################################################################
print("\n" + "-" * 78)
print(f"CHURN: MONTHLY  (alert at PSI {churn_policy.monitoring['psi_alert']}, "
      f"cadence {churn_policy.monitoring['cadence_days']}d)")
print("-" * 78)

accounts = pd.read_parquet(MODELS / "churn_accounts.parquet")
cached = np.load(MODELS / "churn_engagement.npz")
eng = cd.Engagement(accounts=accounts, actions=cached["actions"], tickets=cached["tickets"],
                    negative=cached["negative"], days=cached["actions"].shape[1])
churn_model = registry.load("ml.churn.scorer")

churn_alert = float(churn_policy.monitoring["psi_alert"])
churn_watch = float(churn_policy.monitoring["psi_watch"])
cadence = int(churn_policy.monitoring["cadence_days"])

# The baseline starts at day 90 so its 90-day rolling windows are complete.
# A baseline taken while the windows are still filling drifts against every
# later snapshot for a reason that has nothing to do with customers.
snapshots = list(range(90, eng.days - 20, cadence))
panels = {}
for snapshot in snapshots:
    panel = cd.build_features(eng, snapshot, CHURN_MIN_HISTORY)
    panels[snapshot] = (panel, churn_model.predict_proba(cd.design_matrix(panel))[:, 1])

baseline_snapshot = snapshots[0]
base_panel, base_scores = panels[baseline_snapshot]
churn_monitor = drift.Monitor(base_panel[cd.FEATURES], cd.FEATURES, score_ref=base_scores)

monthly_rows = []
for snapshot in snapshots[1:]:
    panel, panel_scores = panels[snapshot]
    report = churn_monitor.check(panel, panel_scores)
    name, worst_psi = drift.worst(report)
    monthly_rows.append({
        "snapshot": snapshot,
        "months_after_baseline": (snapshot - baseline_snapshot) // cadence,
        "book": len(panel),
        "worst_signal": "score" if name == "__score__" else name,
        "worst_psi": worst_psi,
        "score_psi": report["__score__"]["psi"],
        "verdict": drift.verdict(report, churn_alert, churn_watch),
    })

monthly = pd.DataFrame(monthly_rows)
print(f"baseline snapshot: day {baseline_snapshot}\n")
print(monthly.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
crossings = monthly[monthly["verdict"] == "alert"]
print(f"\n{cadence} days between checks. the worst signal peaks at "
      f"{monthly['worst_psi'].max():.3f} against a {churn_alert} threshold,")
if len(crossings):
    peak = crossings.iloc[0]
    print(f"crossing it once, at month {peak['months_after_baseline']}, on "
          f"{peak['worst_signal']}, and settling back")
    print(f"to {monthly['worst_psi'].iloc[-1]:.3f} by month "
          f"{monthly['months_after_baseline'].iloc[-1]} with nobody doing anything about it.")
    print("\nthat is the shape of drift rather than an incident: one threshold")
    print("crossing that decays on its own. on a daily clock it would have been")
    print("thirty pages. on a monthly clock it is a line in a review.")
else:
    print("and never crossing it.")
print("\nnothing on this side needs a daily job, an on-call rota or a fast retrain.")
print("checking it weekly would produce the same decisions and four times the noise.")

drifting = monthly.nlargest(3, "worst_psi")["worst_signal"].tolist()
print(f"\nwhat moves most slowly and steadily: {', '.join(dict.fromkeys(drifting))}.")
print("the book ages, engagement distributions shift with it, and that is a")
print("retraining schedule, not an incident.")


# ###########################################################################
# THE PICTURE
# ###########################################################################
fig, axes = viz.new_grid(2, width=12.0, height=4.2)

# Both signals on one scale: each divided by its own alert threshold, so the
# alert line is 1.0 for both and the panel shows which one actually fires.
# Plotting only the PSI here would have been a chart of the signal that does
# not work, under a title claiming it does.
axes[0].plot(warm["day"], warm["worst_psi"] / alert_at, color=viz.INK_SOFT,
             linewidth=1.3, label="worst feature / score PSI")
axes[0].plot(warm["day"], warm["z_1d"] / Z_ALERT, color=viz.ORANGE, linewidth=1.0,
             alpha=0.55, label="flag rate, single day")
axes[0].plot(warm["day"], warm["z_7d"] / Z_ALERT, color=viz.BLUE, linewidth=1.8,
             label="flag rate, 7-day trailing")
axes[0].axhline(1.0, color=viz.ORANGE, linestyle="--", linewidth=1.5)
axes[0].axvline(RISK_DRIFT_DAY, color=viz.INK, linewidth=1.4, linestyle=":")

top = max(2.0, (warm["z_7d"] / Z_ALERT).max() * 1.15)
axes[0].set_ylim(-0.75, top)
axes[0].set_xlim(warm["day"].min() - 2, label_based_day + 6)
axes[0].text(warm["day"].min(), 1.06, "alert", color=viz.ORANGE, fontsize=8)
axes[0].text(RISK_DRIFT_DAY - 2, top * 0.42, "tactics change", ha="right", va="center",
             rotation=90, color=viz.INK, fontsize=8)
axes[0].axvline(label_based_day, color=viz.INK_SOFT, linewidth=1.2, linestyle="-.")
axes[0].text(label_based_day - 3, top * 0.42, "labels usable", ha="right", va="center",
             rotation=90, color=viz.INK_SOFT, fontsize=8)

arrow_y = -0.35
axes[0].annotate("", xy=(label_based_day, arrow_y),
                 xytext=(first_alert or RISK_DRIFT_DAY, arrow_y),
                 arrowprops={"arrowstyle": "<->", "color": viz.INK_SOFT, "linewidth": 1.0})
axes[0].text((label_based_day + (first_alert or RISK_DRIFT_DAY)) / 2, arrow_y + 0.08,
             f"{lag} days of head start", ha="center", color=viz.INK_SOFT, fontsize=8)
axes[0].legend(frameon=False, fontsize=8, labelcolor=viz.INK_SOFT,
               loc="upper left", bbox_to_anchor=(0.16, 1.0))
viz.label(axes[0], "Risk: daily, and no labels needed", "day",
          "multiples of the alert threshold")

axes[1].plot(monthly["months_after_baseline"], monthly["worst_psi"],
             color=viz.BLUE, linewidth=1.8, marker="o", markersize=5)
axes[1].axhline(churn_alert, color=viz.ORANGE, linestyle="--", linewidth=1.5)
axes[1].text(monthly["months_after_baseline"].max(), churn_alert * 1.05,
             f"alert {churn_alert}", ha="right", color=viz.ORANGE, fontsize=8)
axes[1].set_xticks(monthly["months_after_baseline"].tolist())
axes[1].set_ylim(0, max(churn_alert * 1.5, monthly["worst_psi"].max() * 1.3))
viz.label(axes[1], f"Churn: every {cadence} days, and one crossing that decays",
          "months after baseline", "worst PSI vs the baseline snapshot")
viz.save(fig, PLOTS / "monitoring_two_clocks.png")

daily.to_csv(TABLES / "risk_drift_daily.csv", index=False)
monthly.to_csv(TABLES / "churn_drift_monthly.csv", index=False)

summary = {
    "risk": {
        "cadence_days": int(risk_policy.monitoring["cadence_days"]),
        "psi_alert": alert_at,
        "days_monitored": int(len(warm)),
        "median_psi_before_drift": float(before["worst_psi"].median()),
        "median_psi_after_drift": float(after["worst_psi"].median()),
        "drift_day": RISK_DRIFT_DAY,
        "first_alert_day": first_alert,
        "first_psi_alert_day": first_psi,
        "first_flag_rate_alert_day": first_flag,
        "window_days": WINDOW_DAYS,
        "baseline_days": BASELINE_DAYS,
        "training_flag_rate": float(training_flag_rate),
        "z_alert": Z_ALERT,
        "quiet_days_monitored": int(len(before)),
        "false_alarms_before_change_naive_2x": false_alarms_naive,
        "false_alarms_before_change_fixed_baseline": false_alarms_training,
        "alerting_days_after_change": int(after["flag_alert"].sum()),
        "false_alarms_before_change_z_score": false_alarms_z,
        "label_based_detection_day": label_based_day,
        "head_start_days": int(lag),
    },
    "churn": {
        "cadence_days": cadence,
        "psi_alert": churn_alert,
        "baseline_snapshot": baseline_snapshot,
        "checks": len(monthly),
        "max_worst_psi": float(monthly["worst_psi"].max()),
        "alerts": int((monthly["verdict"] == "alert").sum()),
    },
}
(METRICS / "monitoring_metrics.json").write_text(json.dumps(summary, indent=2, default=float))
print(f"\nmetrics written: {METRICS.name}/monitoring_metrics.json")
print(f"tables written:  {TABLES.name}/risk_drift_daily.csv, "
      f"{TABLES.name}/churn_drift_monthly.csv")
