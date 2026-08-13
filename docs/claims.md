<!-- GENERATED FILE. Edit docs/claims.yaml, then run:
     python scripts/verify_claims.py --render -->

# The claims ledger

Every substantive claim in **[Why Your Churn Model Is Not a Risk Model (And Why It Matters)](https://medium.com/ai-that-ships/why-your-churn-model-is-not-a-risk-model-and-why-it-matters-963e1f9bd2e9)**, with a verdict and a pointer to the
metric that settles it.

This page is generated from [`claims.yaml`](claims.yaml) by
[`scripts/verify_claims.py`](https://github.com/santoshshinde2012/churn-vs-risk-poc/blob/main/scripts/verify_claims.py), which resolves every
pointer against `outputs/metrics/*.json` from a real run. CI fails if any of them
stops holding, so a claim here cannot quietly become untrue.

**28 claims, 67 verified measurements.**

| Verdict | | Count | Meaning |
| :--- | :-- | ---: | :--- |
| 🟢 | **Confirmed** | 20 | The run supports the claim as stated. |
| 🟡 | **Qualified** | 3 | Supported, with a caveat that changes what you do. |
| 🔴 | **Contradicted** | 2 | The run does not support the claim as stated. |
| 🔵 | **Demonstrated** | 3 | An engineering practice shown working. |

The three claims that are not confirmed are the reason this ledger exists. A
companion repo that agrees with its article about everything has not been run.

## At a glance

| ID | Theme | Claim | Verdict | Tested by |
| :--- | :--- | :--- | :-- | :--- |
| [LBL-01](#lbl-01) | Labels | The churn label is a window you choose, not a fact. Too short and people on holiday get l... | 🟢 Confirmed | `01_churn_batch_pipeline.py` |
| [LBL-02](#lbl-02) | Labels | Chargebacks land 30 to 90 days after the transaction, so the newest data has no labels ye... | 🟢 Confirmed | `02_risk_realtime_pipeline.py` |
| [LBL-03](#lbl-03) | Labels | Train only on matured labels. Training on the labels you happen to have today means train... | 🟢 Confirmed | `02_risk_realtime_pipeline.py` |
| [LBL-04](#lbl-04) | Labels | Validate on a time-based split, never a random one. A fraud model that shines on a shuffl... | 🟢 Confirmed | `02_risk_realtime_pipeline.py` |
| [FTR-01](#ftr-01) | Features | Features must be built point-in-time. If future data leaks in, offline AUC looks great an... | 🟢 Confirmed | `01_churn_batch_pipeline.py` |
| [FTR-02](#ftr-02) | Features | New users break both models in opposite ways. Churn features need 30 to 90 days of histor... | 🟢 Confirmed | `01_churn_batch_pipeline.py`, `02_risk_realtime_pipeline.py` |
| [MET-01](#met-01) | Metrics | At these base rates accuracy means nothing and even ROC-AUC flatters the model. Use PR-AUC. | 🟢 Confirmed | `01_churn_batch_pipeline.py`, `part_a_metrics.ipynb` |
| [MET-02](#met-02) | Metrics | Use precision at k, where k is real capacity: how many accounts the retention team can ac... | 🟢 Confirmed | `01_churn_batch_pipeline.py` |
| [MET-03](#met-03) | Metrics | Calibrate the probabilities. A cost-based threshold only works if a score of 0.9 really m... | 🟡 Qualified | `02_risk_realtime_pipeline.py` |
| [CST-01](#cst-01) | Cost | Set the threshold in dollars. Blocking a good customer costs $500 and a missed fraud cost... | 🔴 Contradicted | `02_risk_realtime_pipeline.py` |
| [CST-02](#cst-02) | Cost | A risk false positive is expensive, so the insult rate has to stay tiny. | 🟢 Confirmed | `02_risk_realtime_pipeline.py` |
| [CST-03](#cst-03) | Cost | Four decision bands, not two. The bands catch fraud the block threshold must let through... | 🟢 Confirmed | `02_risk_realtime_pipeline.py` |
| [FBK-01](#fbk-01) | Feedback loops | Risk has a selective label problem. When a transaction is blocked, nobody learns whether... | 🟢 Confirmed | `04_feedback_loops.py` |
| [FBK-02](#fbk-02) | Feedback loops | Fix selective labels by letting a small random slice of borderline traffic through, and b... | 🟡 Qualified | `04_feedback_loops.py` |
| [FBK-03](#fbk-03) | Feedback loops | Churn has the opposite problem. A retention offer that works makes the saved customer loo... | 🟢 Confirmed | `04_feedback_loops.py` |
| [FBK-04](#fbk-04) | Feedback loops | A holdout group that gets no intervention keeps the labels clean, and is the only honest... | 🟢 Confirmed | `04_feedback_loops.py` |
| [FBK-05](#fbk-05) | Feedback loops | A high churn score says who might leave, not who can be convinced to stay. The top decile... | 🟢 Confirmed | `04_feedback_loops.py` |
| [FBK-06](#fbk-06) | Feedback loops | With holdouts in place, the next step is uplift modelling: target the people who can actu... | 🔴 Contradicted | `04_feedback_loops.py` |
| [ARC-01](#arc-01) | Architecture | The model returns a probability and nothing else. A decision layer decides what it means,... | 🔵 Demonstrated | `03_live_scoring_demo.py` |
| [ARC-02](#arc-02) | Architecture | The full latency budget is 50 to 100 ms, including network, feature lookup and inference. | 🟢 Confirmed | `03_live_scoring_demo.py` |
| [ARC-03](#arc-03) | Architecture | Decide the timeout behaviour before launch. Most commerce flows fail open; irreversible a... | 🔵 Demonstrated | `03_live_scoring_demo.py` |
| [ARC-04](#arc-04) | Architecture | Plain rules run next to the model as detectors in their own right: velocity limits, block... | 🟢 Confirmed | `02_risk_realtime_pipeline.py`, `03_live_scoring_demo.py` |
| [ARC-05](#arc-05) | Architecture | Track training with MLflow and promote through a Unity Catalog alias, so batch jobs simpl... | 🔵 Demonstrated | `01_churn_batch_pipeline.py`, `03_live_scoring_demo.py` |
| [MON-01](#mon-01) | Monitoring | Churn drifts over months, so a monthly look at distributions is enough. | 🟢 Confirmed | `05_monitoring_two_clocks.py` |
| [MON-02](#mon-02) | Monitoring | Fraud goes stale in days. That side needs tight alerts and fast retraining, because a sud... | 🟡 Qualified | `05_monitoring_two_clocks.py` |
| [MON-03](#mon-03) | Monitoring | Label maturation controls the retraining schedule more than any drift chart does. | 🟢 Confirmed | `05_monitoring_two_clocks.py` |
| [DAT-01](#dat-01) | Data | Monthly churn might be 2 to 5% positive; fraud is often 0.1 to 1% of transactions. | 🟢 Confirmed | `01_churn_batch_pipeline.py`, `02_risk_realtime_pipeline.py` |
| [DAT-02](#dat-02) | Data | The metric story is not an artefact of simulated data. | 🟢 Confirmed | `01_churn_batch_pipeline.py` |

## The evidence

### LBL-01

**🟢 Confirmed** &nbsp;·&nbsp; Labels &nbsp;·&nbsp; Fig 1a

> The churn label is a window you choose, not a fact. Too short and people on holiday get labelled as churned.

| Measurement | Value | |
| :--- | :--- | :-- |
| a 7-day window mislabels 53% of its churners: they came back | `0.532348 vs 0.532 +/- 0.06` | ✅ |
| a 60-day window mislabels only 2%, and confirms two months too late | `0.0244216 vs 0.024 +/- 0.02` | ✅ |

Runs in `01_churn_batch_pipeline.py`, section 1. Written up in [docs/pipelines.md](pipelines.md#the-label-is-a-window-you-choose).

### LBL-02

**🟢 Confirmed** &nbsp;·&nbsp; Labels &nbsp;·&nbsp; Fig 1b

> Chargebacks land 30 to 90 days after the transaction, so the newest data has no labels yet. This is label maturation.

| Measurement | Value | |
| :--- | :--- | :-- |
| 1,420 transactions really were fraud | `1420 vs 1420 +/- 60` | ✅ |
| only 959 have a chargeback on file today | `959 vs 959 +/- 60` | ✅ |
| 59% of the stream is old enough to trust | `0.586209 vs 0.586 +/- 0.05` | ✅ |

Runs in `02_risk_realtime_pipeline.py`, section 2. Written up in [docs/pipelines.md](pipelines.md#label-maturation-is-the-binding-constraint).

### LBL-03

**🟢 Confirmed** &nbsp;·&nbsp; Labels

> Train only on matured labels. Training on the labels you happen to have today means training on frauds marked legitimate.

| Measurement | Value | |
| :--- | :--- | :-- |
| training on today's labels scores 0.885 on the drift week | `0.884675 vs 0.885 +/- 0.04` | ✅ |
| the same rows with true labels score 0.909; the gap is the tax | `0.909346 vs 0.909 +/- 0.04` | ✅ |
| the honest matured-only model scores 0.855 and is 90 days stale | `0.855222 vs 0.855 +/- 0.04` | ✅ |

Runs in `02_risk_realtime_pipeline.py`, section 2. Written up in [docs/pipelines.md](pipelines.md#label-maturation-is-the-binding-constraint).

### LBL-04

**🟢 Confirmed** &nbsp;·&nbsp; Labels

> Validate on a time-based split, never a random one. A fraud model that shines on a shuffled test set is usually memorising last quarter's attacks.

| Measurement | Value | |
| :--- | :--- | :-- |
| the shuffled-split model scores 0.988 on held-out drift-week rows | `0.988397 vs 0.988 +/- 0.03` | ✅ |
| the time-split model scores 0.877 on the identical rows | `0.876823 vs 0.877 +/- 0.04` | ✅ |

Runs in `02_risk_realtime_pipeline.py`, section 3. Written up in [docs/pipelines.md](pipelines.md#out-of-time-never-shuffled).

### FTR-01

**🟢 Confirmed** &nbsp;·&nbsp; Features &nbsp;·&nbsp; Fig 4

> Features must be built point-in-time. If future data leaks in, offline AUC looks great and production disappoints.

| Measurement | Value | |
| :--- | :--- | :-- |
| the leaky model scores 0.904 when evaluated the same leaky way | `0.904027 vs 0.904 +/- 0.04` | ✅ |
| the same model scores 0.470 once asked about the future | `0.469747 vs 0.47 +/- 0.06` | ✅ |
| the point-in-time model scores 0.742, and keeps scoring it | `0.74194 vs 0.742 +/- 0.04` | ✅ |

Runs in `01_churn_batch_pipeline.py`, section 2. Written up in [docs/pipelines.md](pipelines.md#point-in-time-features).

### FTR-02

**🟢 Confirmed** &nbsp;·&nbsp; Features &nbsp;·&nbsp; Fig 5a, 5b

> New users break both models in opposite ways. Churn features need 30 to 90 days of history; new accounts are the highest-fraud segment.

| Measurement | Value | |
| :--- | :--- | :-- |
| 3.4% of the churn book is younger than the feature windows | `0.0342499 vs 0.034 +/- 0.02` | ✅ |
| new accounts carry a 7.0% fraud rate | `0.0701583 vs 0.0702 +/- 0.02` | ✅ |
| aged accounts carry 0.44%, so new accounts are ~16x riskier | `0.00442728 vs 0.0044 +/- 0.003` | ✅ |

Runs in `01_churn_batch_pipeline.py`, `02_risk_realtime_pipeline.py`, 01 section 3, 02 section 7. Written up in [docs/pipelines.md](pipelines.md#cold-start).

### MET-01

**🟢 Confirmed** &nbsp;·&nbsp; Metrics &nbsp;·&nbsp; Fig 7a, 7b

> At these base rates accuracy means nothing and even ROC-AUC flatters the model. Use PR-AUC.

| Measurement | Value | |
| :--- | :--- | :-- |
| ROC-AUC reads 0.979 | `0.97918 >= 0.95` | ✅ |
| PR-AUC reads 0.742 on the same model and the same rows | `0.74194 vs 0.742 +/- 0.04` | ✅ |

Runs in `01_churn_batch_pipeline.py`, `part_a_metrics.ipynb`, 01 section 4. Written up in [docs/pipelines.md](pipelines.md#metrics-and-whether-the-probabilities-are-real).

### MET-02

**🟢 Confirmed** &nbsp;·&nbsp; Metrics

> Use precision at k, where k is real capacity: how many accounts the retention team can actually call.

| Measurement | Value | |
| :--- | :--- | :-- |
| 88% of the top 200 are real churners | `0.88 vs 0.88 +/- 0.05` | ✅ |
| against a 5.4% base rate, so calling at random | `0.0540347 vs 0.054 +/- 0.01` | ✅ |

Runs in `01_churn_batch_pipeline.py`, section 4. Written up in [docs/pipelines.md](pipelines.md#metrics-and-whether-the-probabilities-are-real).

### MET-03

**🟡 Qualified** &nbsp;·&nbsp; Metrics

> Calibrate the probabilities. A cost-based threshold only works if a score of 0.9 really means 90%.

**Why qualified.** Correct, and a global calibration metric will not tell you whether it holds. Overall ECE reads 0.001 while ECE inside the decision region reads 0.12, because the average is dominated by a population that is 99% easy negatives. Measure calibration where the decisions are made.

| Measurement | Value | |
| :--- | :--- | :-- |
| ECE over everything is 0.001, which looks perfect | `0.000978937 <= 0.01` | ✅ |
| ECE over the decision region is 0.12, which is not | `0.122642 >= 0.05` | ✅ |

Runs in `02_risk_realtime_pipeline.py`, section 5. Written up in [docs/findings.md](findings.md#1-the-dollar-threshold-is-a-starting-point-not-an-answer).

### CST-01

**🔴 Contradicted** &nbsp;·&nbsp; Cost &nbsp;·&nbsp; Fig 8

> Set the threshold in dollars. Blocking a good customer costs $500 and a missed fraud costs $150, so block when p > 0.77. Not 0.5.

**Why contradicted.** The arithmetic is right and the derived threshold is still the correct starting point, but on measured costs 0.77 is worse than the 0.5 it was meant to replace. The cost curve bottoms out at 0.38. The cause is MET-03: the formula was fed probabilities that are not calibrated in the band where blocking happens. Derive from costs, then recompute from the inference log.

| Measurement | Value | |
| :--- | :--- | :-- |
| the cost model gives 0.77, exactly as the article says | `0.769231 vs 0.769 +/- 0.001` | ✅ |
| blocking at 0.77 costs $19,100 over the test window | `19100 >= 15000` | ✅ |
| blocking at the 0.5 default costs $14,950, which is less | `14950 <= 20000` | ✅ |

Runs in `02_risk_realtime_pipeline.py`, section 5. Written up in [docs/findings.md](findings.md#1-the-dollar-threshold-is-a-starting-point-not-an-answer).

### CST-02

**🟢 Confirmed** &nbsp;·&nbsp; Cost

> A risk false positive is expensive, so the insult rate has to stay tiny.

| Measurement | Value | |
| :--- | :--- | :-- |
| 0.006% of legitimate transactions are blocked | `6.42333e-05 <= 0.001` | ✅ |

Runs in `02_risk_realtime_pipeline.py`, section 6. Written up in [docs/pipelines.md](pipelines.md#bands-and-the-rules-beside-them).

### CST-03

**🟢 Confirmed** &nbsp;·&nbsp; Cost

> Four decision bands, not two. The bands catch fraud the block threshold must let through to keep the insult rate low.

| Measurement | Value | |
| :--- | :--- | :-- |
| the block band alone stops 54% of drift-week fraud | `0.541176 vs 0.541 +/- 0.08` | ✅ |
| all four bands together touch 80% of it | `0.797647 vs 0.798 +/- 0.08` | ✅ |

Runs in `02_risk_realtime_pipeline.py`, section 6. Written up in [docs/pipelines.md](pipelines.md#bands-and-the-rules-beside-them).

### FBK-01

**🟢 Confirmed** &nbsp;·&nbsp; Feedback loops &nbsp;·&nbsp; Fig 9a

> Risk has a selective label problem. When a transaction is blocked, nobody learns whether it was fraud, so the model goes blind where it acts.

| Measurement | Value | |
| :--- | :--- | :-- |
| 242 confirmed frauds are blocked and therefore never labelled | `242 vs 242 +/- 60` | ✅ |
| only 114 remain observable, 32% of the window's fraud | `114 vs 114 +/- 40` | ✅ |
| the naive retrain scores 0.656 | `0.655717 vs 0.656 +/- 0.06` | ✅ |
| against 0.904 if every outcome had been observed | `0.904121 vs 0.904 +/- 0.04` | ✅ |

Runs in `04_feedback_loops.py`, part B. Written up in [docs/pipelines.md](pipelines.md#risk-blocking-erases-the-outcome).

### FBK-02

**🟡 Qualified** &nbsp;·&nbsp; Feedback loops

> Fix selective labels by letting a small random slice of borderline traffic through, and by using review-queue decisions as labels.

**Why qualified.** The exploration slice works and is cheap: 2% of blocked traffic recovers two thirds of the damage. Review-queue labels made things worse here, not better, because the simulated reviewers are wrong 12% of the time and a wrong label in the band the model is least sure about does more harm than a missing one. The article's "carefully" is doing a lot of work.

| Measurement | Value | |
| :--- | :--- | :-- |
| a 2% exploration slice recovers to 0.825 | `0.825007 vs 0.825 +/- 0.06` | ✅ |
| adding noisy review-queue labels drops it back to 0.806 | `0.806201 vs 0.806 +/- 0.06` | ✅ |

Runs in `04_feedback_loops.py`, part B. Written up in [docs/pipelines.md](pipelines.md#risk-blocking-erases-the-outcome).

### FBK-03

**🟢 Confirmed** &nbsp;·&nbsp; Feedback loops &nbsp;·&nbsp; Fig 9b

> Churn has the opposite problem. A retention offer that works makes the saved customer look like someone who was never going to leave.

| Measurement | Value | |
| :--- | :--- | :-- |
| 96 accounts are saved and relabelled 'did not churn' | `96 vs 96 +/- 30` | ✅ |
| next month's model trained on those labels scores 0.717 | `0.716612 vs 0.717 +/- 0.05` | ✅ |
| against 0.729 trained on labels no campaign touched | `0.728572 vs 0.729 +/- 0.05` | ✅ |

Runs in `04_feedback_loops.py`, part A. Written up in [docs/pipelines.md](pipelines.md#churn-the-intervention-erases-the-label).

### FBK-04

**🟢 Confirmed** &nbsp;·&nbsp; Feedback loops

> A holdout group that gets no intervention keeps the labels clean, and is the only honest way to know whether the campaigns work at all.

| Measurement | Value | |
| :--- | :--- | :-- |
| incremental retention measures +6.7% against the holdout | `0.0670733 vs 0.067 +/- 0.03` | ✅ |
| with a 95% interval of +/- 2.2%, so the effect is real | `0.0224377 <= 0.04` | ✅ |

Runs in `04_feedback_loops.py`, section A1. Written up in [docs/pipelines.md](pipelines.md#churn-the-intervention-erases-the-label).

### FBK-05

**🟢 Confirmed** &nbsp;·&nbsp; Feedback loops

> A high churn score says who might leave, not who can be convinced to stay. The top decile contains lost causes and sleeping dogs.

| Measurement | Value | |
| :--- | :--- | :-- |
| 73 of the 200 highest-risk accounts cannot be saved | `73 >= 40` | ✅ |
| 13 more would be pushed out the door by the call itself | `13 >= 5` | ✅ |

Runs in `04_feedback_loops.py`, section A3. Written up in [docs/findings.md](findings.md#2-uplift-modelling-works-as-a-screen-not-as-a-ranker).

### FBK-06

**🔴 Contradicted** &nbsp;·&nbsp; Feedback loops

> With holdouts in place, the next step is uplift modelling: target the people who can actually be persuaded (Ascarza 2018).

**Why contradicted.** As a ranking rule an uplift model loses to the plain churn score, by a wide margin, on a quarter of randomised campaign data. Its ranking correlates 0.01 with the true effect against the risk score's 0.26. Used as a screen instead - drop non-positive estimated uplift, then rank by risk - it beats risk targeting by 14%. The sign of a treatment effect is estimable at this sample size; the per-account magnitude is not.

| Measurement | Value | |
| :--- | :--- | :-- |
| risk targeting keeps 19.2 customers per 200 calls | `19.244 vs 19.24 +/- 2.5` | ✅ |
| uplift as a ranker keeps 16.4, which is worse | `16.3744 vs 16.37 +/- 2.5` | ✅ |
| uplift as a screen keeps 21.9, which is better than both | `21.8776 vs 21.88 +/- 2.5` | ✅ |
| the uplift ranking correlates 0.01 with the true effect | `0.0108548 <= 0.1` | ✅ |

Runs in `04_feedback_loops.py`, section A3. Written up in [docs/findings.md](findings.md#2-uplift-modelling-works-as-a-screen-not-as-a-ranker).

### ARC-01

**🔵 Demonstrated** &nbsp;·&nbsp; Architecture &nbsp;·&nbsp; Fig 10a, 10b

> The model returns a probability and nothing else. A decision layer decides what it means, and business rules change without retraining the model.

| Measurement | Value | |
| :--- | :--- | :-- |
| policy v1 blocks above 0.77 | `0.769231 vs 0.769 +/- 0.001` | ✅ |
| policy v2 blocks above 0.64, from a cost change alone | `0.64 vs 0.64 +/- 0.001` | ✅ |
| 258 decisions move across 279,030 replayed scores, no retraining | `258 >= 50` | ✅ |

Runs in `03_live_scoring_demo.py`, policy swap. Written up in [docs/architecture.md](architecture.md#the-decision-layer-is-not-the-model).

### ARC-02

**🟢 Confirmed** &nbsp;·&nbsp; Architecture &nbsp;·&nbsp; Fig 2b

> The full latency budget is 50 to 100 ms, including network, feature lookup and inference.

| Measurement | Value | |
| :--- | :--- | :-- |
| p95 is 18 ms against an 80 ms budget | `18.6683 <= 80` | ✅ |
| the budget is enforced by the caller, not the library default | `80 vs 80 +/- 0` | ✅ |

Runs in `03_live_scoring_demo.py`, latency. Written up in [docs/pipelines.md](pipelines.md#03-serving-with-an-outage).

### ARC-03

**🔵 Demonstrated** &nbsp;·&nbsp; Architecture &nbsp;·&nbsp; Fig 11

> Decide the timeout behaviour before launch. Most commerce flows fail open; irreversible actions fail closed.

| Measurement | Value | |
| :--- | :--- | :-- |
| 3 irreversible actions demanded verification during the outage | `3 >= 1` | ✅ |
| no call outlives the budget by more than a few milliseconds | `85.334 <= 200` | ✅ |

Runs in `03_live_scoring_demo.py`, outage. Written up in [docs/pipelines.md](pipelines.md#03-serving-with-an-outage).

### ARC-04

**🟢 Confirmed** &nbsp;·&nbsp; Architecture

> Plain rules run next to the model as detectors in their own right: velocity limits, blocklists.

| Measurement | Value | |
| :--- | :--- | :-- |
| with no model at all the rules touch 17% of drift-week fraud | `0.167059 >= 0.05` | ✅ |
| and they caught fraud during the outage, with no model in the loop | `1 >= 1` | ✅ |

Runs in `02_risk_realtime_pipeline.py`, `03_live_scoring_demo.py`, 02 section 6, 03 outage. Written up in [docs/architecture.md](architecture.md#rules-beside-the-model).

### ARC-05

**🔵 Demonstrated** &nbsp;·&nbsp; Architecture &nbsp;·&nbsp; Fig 10a

> Track training with MLflow and promote through a Unity Catalog alias, so batch jobs simply load models:/name@champion.

| Measurement | Value | |
| :--- | :--- | :-- |
| the churn model takes the @champion alias after beating the incumbent | `1 vs 1 +/- 0` | ✅ |
| so does the risk model | `1 vs 1 +/- 0` | ✅ |

Runs in `01_churn_batch_pipeline.py`, `03_live_scoring_demo.py`, 01 section 6. Written up in [docs/architecture.md](architecture.md#mlflow-is-real-mlflow).

### MON-01

**🟢 Confirmed** &nbsp;·&nbsp; Monitoring &nbsp;·&nbsp; Fig 12a

> Churn drifts over months, so a monthly look at distributions is enough.

| Measurement | Value | |
| :--- | :--- | :-- |
| the worst signal peaks at 0.276 and decays without intervention | `0.276155 vs 0.276 +/- 0.06` | ✅ |
| one threshold crossing in four monthly checks | `1 <= 2` | ✅ |

Runs in `05_monitoring_two_clocks.py`, churn section. Written up in [docs/pipelines.md](pipelines.md#05-monitoring-on-two-clocks).

### MON-02

**🟡 Qualified** &nbsp;·&nbsp; Monitoring &nbsp;·&nbsp; Fig 12b

> Fraud goes stale in days. That side needs tight alerts and fast retraining, because a sudden drop is usually an adversary.

**Why qualified.** True, but distribution monitoring is the wrong instrument. A 0.72%-of- traffic attack never moves a population PSI at any usable threshold. What catches it in three days is the flag rate - the share of traffic scoring above the challenge band - over a trailing week, against a rolling reference rather than a fixed one.

| Measurement | Value | |
| :--- | :--- | :-- |
| population PSI never alerts, across all 55 monitored days | `null` | ✅ |
| the flag rate alerts on day 203, three days after the change | `203 vs 203 +/- 5` | ✅ |
| with no false alarms across the quiet days before it | `0 vs 0 +/- 0` | ✅ |

Runs in `05_monitoring_two_clocks.py`, risk section. Written up in [docs/findings.md](findings.md#3-monitor-the-flag-rate-not-only-the-feature-distributions).

### MON-03

**🟢 Confirmed** &nbsp;·&nbsp; Monitoring

> Label maturation controls the retraining schedule more than any drift chart does.

| Measurement | Value | |
| :--- | :--- | :-- |
| a label-based chart cannot move until day 290 | `290 vs 290 +/- 0` | ✅ |
| the label-free alarm buys 87 days of head start | `87 >= 60` | ✅ |

Runs in `05_monitoring_two_clocks.py`, risk section. Written up in [docs/findings.md](findings.md#3-monitor-the-flag-rate-not-only-the-feature-distributions).

### DAT-01

**🟢 Confirmed** &nbsp;·&nbsp; Data

> Monthly churn might be 2 to 5% positive; fraud is often 0.1 to 1% of transactions.

| Measurement | Value | |
| :--- | :--- | :-- |
| the simulated book churns at 5.4% per 30-day window | `0.0540347 vs 0.054 +/- 0.02` | ✅ |
| the simulated stream is 0.51% fraud | `0.00508906 vs 0.0051 +/- 0.004` | ✅ |

Runs in `01_churn_batch_pipeline.py`, `02_risk_realtime_pipeline.py`, 01 section 4, 02 section 1. Written up in [docs/pipelines.md](pipelines.md#01-churn-the-batch-path).

### DAT-02

**🟢 Confirmed** &nbsp;·&nbsp; Data

> The metric story is not an artefact of simulated data.

| Measurement | Value | |
| :--- | :--- | :-- |
| on the public IBM Telco dataset the same recipe scores 0.648 | `0.647799 vs 0.648 +/- 0.05` | ✅ |
| with precision@200 of 76% against a 26.5% base rate | `0.76 vs 0.76 +/- 0.08` | ✅ |

Runs in `01_churn_batch_pipeline.py`, section 7. Written up in [docs/pipelines.md](pipelines.md#01-churn-the-batch-path).

