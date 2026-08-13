# Where the results argue with the article

Three of the article's claims did not survive contact with the data. The scripts print these
findings rather than burying them, [the claims ledger](claims.md) records them as
`contradicted` or `qualified`, and CI fails if they stop being true.

This page exists because a companion repo that agrees with its article about everything has
not been run.

| | Claim | Verdict | What actually happens |
| :-- | :--- | :--- | :--- |
| 1 | The threshold is 0.77 from the costs, not 0.5 | 🔴 Contradicted | 0.77 costs more than the 0.5 it replaces. The scores are miscalibrated exactly where the decision is made. |
| 2 | Uplift modelling is the next step after holdouts | 🔴 Contradicted | As a ranker it loses to the plain risk score. As a screen it beats it by 14%. |
| 3 | Fraud needs tight drift alerts | 🟡 Qualified | Distribution drift never fires on a real attack. The flag rate does, in three days. |

---

## 1. The dollar threshold is a starting point, not an answer

**Claim** ([CST-01](claims.md#cst-01)) — blocking a good customer costs $500 and a missed
fraud costs $150, so block when `p > 500/(500+150) = 0.77`. Not 0.5.

**What happens.** The arithmetic is right. The threshold it produces is worse than the
default it was meant to replace:

```
  cost at the derived threshold 0.77:  $    19,100
  cost at the default           0.50:  $    14,950
  cost at the measured minimum  0.38:  $    12,800
```

**Why.** The formula treats `p` as a real probability. Measured globally, it looks like one:

```
  ECE over everything:                       0.0010   <- looks perfect
  ECE over the decision region (p >= 0.05):  0.1226
  rows scored between 0.38 and 0.77: 68, and 91% of them really were fraud
```

The model says those rows are about 55% likely to be fraud. They are 91% fraud. The
arithmetic that produced 0.77 was fed a number that was not true, so it under-blocks: every
transaction it waves through in that band is far likelier to be fraud than the threshold
assumed.

A global calibration metric never shows this, because it averages over a population that is
99% easy negatives. **Overall ECE is not a calibration check for a decision system.** It is a
check on the part of the score distribution where no decisions happen.

### What to do instead

1. **Derive the threshold from costs.** It is still the right starting point, and it is
   still not 0.5. Keep it in a versioned policy file, derived rather than typed in, so a
   cost change moves it automatically.
2. **Recompute it from the inference log** against measured outcomes, on a schedule. That is
   what the log is for.
3. **Check calibration inside the decision region**, not over the whole score range. On this
   run that is a factor of 100 difference in the number you get.

What never happens is that 0.5 turns out to be right. The measured optimum here is 0.38, and
0.38 beats 0.5 by 14%.

**Where it runs** — [`02_risk_realtime_pipeline.py`](https://github.com/santoshshinde2012/churn-vs-risk-poc/blob/main/02_risk_realtime_pipeline.py),
section 5. Related claims: [CST-01](claims.md#cst-01), [MET-03](claims.md#met-03).

---

## 2. Uplift modelling works as a screen, not as a ranker

**Claim** ([FBK-06](claims.md#fbk-06)) — with holdout groups in place, the next step is
uplift modelling: target the people who can actually be persuaded (Ascarza 2018).

**The premise is confirmed.** 40% of the top risk decile cannot be saved, and 6% are
sleeping dogs who leave *because* you called. That composition problem is real and it is
exactly as described.

**The prescription is not.** A treatment-effect model fitted on a quarter of randomised
campaigns — roughly 3,000 contacted and 3,000 control — ranks *worse* than the plain churn
score:

| Strategy for spending 200 calls | Customers kept | Lost causes called | Sleeping dogs called |
| :--- | ---: | ---: | ---: |
| highest risk score | 19.2 | 73 | 13 |
| highest uplift (T-learner) | 14.5 | 49 | 25 |
| highest uplift (transformed outcome) | 16.4 | 52 | 17 |
| **highest risk, screened by uplift sign** | **21.9** | 66 | 12 |
| at random | 14.4 | 30 | 24 |
| *clairvoyant ceiling* | *72.7* | *0* | *0* |

**Why.** Estimating *whether* anyone responds is a much easier statistical problem than
estimating *who* responds:

```
  correlation of the risk score with the true effect: +0.263
  correlation of the uplift estimate with it:         +0.011
```

An uplift estimate is a difference between two noisy quantities, and here the noise is
larger than the thing being measured. Ranking by it throws away the one strong signal
available — the churn probability — and replaces it with noise.

### What to do instead

Use the uplift model for what the data can support: **the sign, not the magnitude**. Drop
the accounts whose estimated uplift is not positive, then rank what survives by churn score.
Same model, same data, one line of difference:

```python
screen = uplift > 0                      # only the sign is used
chosen  = rank_by(score, where=screen)[:budget]
```

That keeps 21.9 customers per 200 calls against 19.2 for risk alone — a 14% improvement, and
it drops 633 of 1,682 eligible accounts before ranking.

Two smaller lessons in the same table:

- **The estimator matters.** A T-learner (one outcome model per arm) is what everyone writes
  first and it is a variance trap: each arm carries its own error and the difference carries
  both. The transformed-outcome estimator — rewrite the outcome so its conditional mean *is*
  the treatment effect, then regress on it once — beats it here, and it is barely more code.
- **The headroom is real but expensive.** A clairvoyant list keeps 72.7. Most of the gap
  between 21.9 and 72.7 buys more experiments, not a better model.

Treating an uplift score as a league table is how uplift projects get cancelled.

**Where it runs** — [`04_feedback_loops.py`](https://github.com/santoshshinde2012/churn-vs-risk-poc/blob/main/04_feedback_loops.py), section A3. Related
claims: [FBK-05](claims.md#fbk-05), [FBK-06](claims.md#fbk-06).

---

## 3. Monitor the flag rate, not only the feature distributions

**Claim** ([MON-02](claims.md#mon-02)) — fraud goes stale in days, so that side needs tight
alerts and fast retraining.

**The conclusion is right and the instrument is wrong.** Population PSI never crosses its
threshold during a real attack, across all 55 monitored days, because the card-testing wave
is **0.72% of traffic**. A statistic computed over the whole population cannot see a change
that small, and lowering the threshold until it does produces an alert every week for
reasons that are never attacks.

What catches it is the **flag rate** — the share of traffic scoring above the challenge band.
The attack is concentrated exactly where the flag rate looks, so a signal that is invisible
in aggregate is obvious in the tail. It fires on day 203, three days after the tactics
changed and 87 days before a chargeback-based chart could move.

### Two details that took several attempts

Both of these were wrong in the first working version of this script, and both produced a
monitor that looked fine and would have been muted within a fortnight.

**A single day is too noisy to page on.** At a 0.4% baseline a day sees about five flagged
transactions, and five events double by chance roughly twice a month. "The flag rate has
doubled" fired on 30% of quiet days. The fix is a trailing week and a binomial z-score,
which prices the count against the volume that produced it — so the same threshold means the
same thing on a Sunday and on Black Friday.

**The alarm needs a rolling reference, not a fixed one.** The pre-attack flag rate on this
stream ranges from 0.14% to 1.10%, because fraud rings arrive in bursts. A fixed baseline
bakes one arbitrary fortnight in as the definition of normal; scored against the training
window, the identical statistic alerts on more than twice as many days.

```python
# last 7 days, against the 28 days before them
z = rate_z_score(flagged_in_window, total_in_window, rate_over_preceding_28_days)
alert = z >= 3.0
```

A drift metric and an alarm want different references, and this is the distinction:

| | Question it answers | Reference |
| :--- | :--- | :--- |
| Feature / score PSI | Does the traffic still look like what this model was fitted to? | The training window |
| Flag rate | Is the model saying yes more often than it normally does? | Recent production, rolling |

Run both. Alarm on the second.

**Where it runs** — [`05_monitoring_two_clocks.py`](https://github.com/santoshshinde2012/churn-vs-risk-poc/blob/main/05_monitoring_two_clocks.py). Related
claims: [MON-02](claims.md#mon-02), [MON-03](claims.md#mon-03).

---

## How these were found

None of them came from reading the code. All three came from **plotting the result and
looking at it**:

- The threshold contradiction surfaced because the cost curve was plotted with both the
  derived threshold and the empirical minimum marked, and they were nowhere near each other.
- The uplift result surfaced because the targeting comparison included a random baseline,
  and the uplift model was barely beating it.
- The monitoring problem surfaced because the first version of the two-clocks figure plotted
  only the signal that never fires, under a title claiming it worked.

If a figure in this repo looks boring, that is a result. If it looks wrong, look harder.
