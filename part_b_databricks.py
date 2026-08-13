# Databricks notebook source
# MAGIC %md
# MAGIC # Part B — the same two pipelines, on Databricks
# MAGIC
# MAGIC Companion to **"Why Your Churn Model Is Not a Risk Model (And Why It Matters)"**.
# MAGIC
# MAGIC Every local stand-in in this repo exists because something on the platform does the
# MAGIC job properly. This notebook is the mapping, and it runs on the
# MAGIC [Free Edition](https://docs.databricks.com/aws/en/getting-started/free-edition).
# MAGIC
# MAGIC | In the repo | On Databricks |
# MAGIC | :--- | :--- |
# MAGIC | `churnrisk/churn_data.build_features` (as-of slices) | time series feature table + `create_training_set` |
# MAGIC | `churnrisk/risk_data.build_features` (single pass) | Structured Streaming + Online Feature Store |
# MAGIC | `outputs/mlruns` | MLflow tracking + Unity Catalog registered models |
# MAGIC | `churnrisk/registry.promote` | UC model aliases (`@champion`) |
# MAGIC | `churnrisk/inference_log.py` | inference tables on a serving endpoint |
# MAGIC | `policies/*.yaml` | a versioned Delta policy table |
# MAGIC | `churnrisk/drift.py` | Lakehouse Monitoring |
# MAGIC | `03_live_scoring_demo.py`'s HTTP server | Mosaic AI Model Serving, route-optimized |
# MAGIC
# MAGIC Set the catalog and schema below, then run top to bottom. Cells that cost money or
# MAGIC take a long time (online stores, serving endpoints) are marked and left commented.

# COMMAND ----------

CATALOG = "ml"
CHURN_SCHEMA = "churn"
RISK_SCHEMA = "risk"

spark.sql(f"CREATE CATALOG IF NOT EXISTS {CATALOG}")
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{CHURN_SCHEMA}")
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{RISK_SCHEMA}")
spark.sql(f"USE CATALOG {CATALOG}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Churn features, as a time series feature table
# MAGIC
# MAGIC The local code guarantees point-in-time correctness structurally: every slice stops at
# MAGIC the snapshot, so there is no future column available to leak. On the platform,
# MAGIC declaring `timeseries_columns` is what buys the same guarantee, and
# MAGIC `create_training_set` then does the as-of join for you.
# MAGIC
# MAGIC The guarantee is the point. Getting this wrong is not a subtle statistical issue; it
# MAGIC is the difference between the two bars in `outputs/plots/churn_leakage.png`.

# COMMAND ----------

from databricks.feature_engineering import FeatureEngineeringClient, FeatureLookup

fe = FeatureEngineeringClient()

# engagement_features_df: one row per (account_id, feature_ts) with the rolling
# aggregates computed in Spark. 30/60/90-day windows, week-over-week change,
# days since the last action, support ticket volume and tone.
fe.create_table(
    name=f"{CATALOG}.{CHURN_SCHEMA}.account_engagement",
    primary_keys=["account_id", "feature_ts"],
    timeseries_columns="feature_ts",          # this is what enables as-of lookups
    df=engagement_features_df,                # noqa: F821 - built in the cell above in a real run
    description="Rolling engagement aggregates per account. Point-in-time safe.",
)

# COMMAND ----------

# labels_df: account_id, label_ts, churned. The label_ts is the day the label was
# decided, not the day the row was written - that distinction is the whole
# mechanism. See 01_churn_batch_pipeline.py section 1 for how the window itself
# is a choice with consequences.
training_set = fe.create_training_set(
    df=labels_df,                             # noqa: F821
    feature_lookups=[
        FeatureLookup(
            table_name=f"{CATALOG}.{CHURN_SCHEMA}.account_engagement",
            lookup_key="account_id",
            timestamp_lookup_key="label_ts",  # as-of: the latest row strictly before this
        )
    ],
    label="churned",
    exclude_columns=["label_ts"],
)

training_df = training_set.load_df().toPandas()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Train, track, register, promote
# MAGIC
# MAGIC `registry.promote` in this repo refuses to move the `@champion` alias unless the
# MAGIC candidate beats the incumbent on the metric that matters. Same idea here, and the
# MAGIC guard matters more than it looks: "we retrained, therefore we shipped" is how a
# MAGIC pipeline quietly regresses for a quarter.

# COMMAND ----------

import mlflow
from mlflow import MlflowClient
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score

mlflow.set_registry_uri("databricks-uc")      # register into Unity Catalog, not the workspace
MODEL_NAME = f"{CATALOG}.{CHURN_SCHEMA}.scorer"

X = training_df.drop(columns=["churned", "account_id"])
y = training_df["churned"]

with mlflow.start_run(run_name="churn-batch") as run:
    model = CalibratedClassifierCV(
        HistGradientBoostingClassifier(random_state=7, max_iter=200),
        method="isotonic", cv=3,
    ).fit(X, y)

    # Validate on a LATER snapshot than the one trained on, never a random split.
    p_holdout = model.predict_proba(X_holdout)[:, 1]                    # noqa: F821
    pr_auc = average_precision_score(y_holdout, p_holdout)              # noqa: F821
    mlflow.log_metric("pr_auc", pr_auc)

    # log_model through the FE client keeps the feature spec attached, so scoring
    # later needs only the keys - the lookups happen automatically.
    fe.log_model(
        model=model,
        artifact_path="model",
        flavor=mlflow.sklearn,
        training_set=training_set,
        registered_model_name=MODEL_NAME,
    )

# COMMAND ----------

client = MlflowClient()
candidate = max(int(v.version) for v in client.search_model_versions(f"name='{MODEL_NAME}'"))

try:
    incumbent = client.get_model_version_by_alias(MODEL_NAME, "champion")
    incumbent_pr_auc = client.get_run(incumbent.run_id).data.metrics["pr_auc"]
except Exception:
    incumbent_pr_auc = None

if incumbent_pr_auc is None or pr_auc > incumbent_pr_auc:
    client.set_registered_model_alias(MODEL_NAME, "champion", candidate)
    print(f"promoted v{candidate} to @champion (pr_auc {pr_auc:.4f})")
else:
    client.set_registered_model_alias(MODEL_NAME, "challenger", candidate)
    print(f"v{candidate} parked as @challenger ({pr_auc:.4f} vs {incumbent_pr_auc:.4f})")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. The nightly batch job
# MAGIC
# MAGIC `score_batch` resolves the alias and re-runs the feature lookups, so the job never
# MAGIC names a version. Promotion in the registry changes what runs tonight.
# MAGIC
# MAGIC Schedule this with a Lakeflow Job. Nothing about it is latency sensitive, which is
# MAGIC the entire reason it can afford 90-day aggregates.

# COMMAND ----------

scores = fe.score_batch(
    model_uri=f"models:/{MODEL_NAME}@champion",
    df=spark.table(f"{CATALOG}.{CHURN_SCHEMA}.accounts_to_score"),     # keys + label_ts only
)

# The holdout is assigned by hashing the account id, never by a random draw. A
# holdout that moves between runs measures nothing: see churnrisk/holdout.py.
scores.createOrReplaceTempView("scores")
spark.sql(f"""
CREATE OR REPLACE TABLE {CATALOG}.{CHURN_SCHEMA}.nightly_scores AS
SELECT
  account_id,
  prediction AS churn_probability,
  current_date() AS scored_on,
  crc32(concat('retention-holdout-v1:', account_id)) % 10000 < 1500 AS in_holdout
FROM scores
""")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. The decision layer, as a versioned table
# MAGIC
# MAGIC The model returns a probability and nothing else. Where the bands sit is a business
# MAGIC input, so it lives in a table with a version on it rather than inside the model
# MAGIC artifact. Changing a threshold is then a row insert, and every decision can be traced
# MAGIC back to the policy version that produced it.

# COMMAND ----------

spark.sql(f"""
CREATE TABLE IF NOT EXISTS {CATALOG}.{RISK_SCHEMA}.decision_policy (
  policy_version STRING,
  effective_from TIMESTAMP,
  action         STRING,
  min_probability DOUBLE,
  fp_cost DOUBLE,
  fn_cost DOUBLE
) USING DELTA
""")

# block_at is derived from the costs, never typed in:
#     block when p * fn_cost > (1 - p) * fp_cost   =>   p > fp / (fp + fn)
spark.sql(f"""
INSERT INTO {CATALOG}.{RISK_SCHEMA}.decision_policy VALUES
  ('1.0.0', current_timestamp(), 'challenge', 0.30, 500, 150),
  ('1.0.0', current_timestamp(), 'review',    0.60, 500, 150),
  ('1.0.0', current_timestamp(), 'block',     500.0/(500+150), 500, 150)
""")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Risk features: streaming in, online store out
# MAGIC
# MAGIC The local version computes velocity and network features in a single pass, updating
# MAGIC its state only after each row has been scored. That ordering is the point-in-time
# MAGIC guarantee. Here, Structured Streaming maintains the same aggregates and the online
# MAGIC store serves them back in milliseconds.
# MAGIC
# MAGIC Note the network features. Per-account velocity cannot see a fraud ring; how many
# MAGIC accounts share a device can. See section 6 of `02_risk_realtime_pipeline.py`.

# COMMAND ----------

from pyspark.sql import functions as F

events = (
    spark.readStream
    .format("kafka")
    .option("subscribe", "transactions")
    .load()
    .select(F.from_json(F.col("value").cast("string"), TRANSACTION_SCHEMA).alias("e"))  # noqa: F821
    .select("e.*")
    .withWatermark("event_ts", "10 minutes")
)

velocity = (
    events
    .groupBy(F.window("event_ts", "10 minutes", "1 minute"), "account_id")
    .agg(F.count("*").alias("txn_10min"),
         F.avg("amount").alias("avg_amount_10min"))
)

# The ring signal: distinct accounts seen on a device inside a rolling hour.
network = (
    events
    .groupBy(F.window("event_ts", "1 hour", "5 minutes"), "device_id")
    .agg(F.approx_count_distinct("account_id").alias("device_accounts"),
         F.count("*").alias("device_txn_1h"))
)

(velocity.writeStream
    .option("checkpointLocation", f"/Volumes/{CATALOG}/{RISK_SCHEMA}/chk/velocity")
    .trigger(processingTime="30 seconds")
    .toTable(f"{CATALOG}.{RISK_SCHEMA}.txn_velocity"))

# COMMAND ----------

# MAGIC %md
# MAGIC **Costs money.** An online store is a provisioned resource. Leave it commented until
# MAGIC you want it, and delete it when you are done.

# COMMAND ----------

# fe.create_online_store(name="risk-online-store", capacity="CU_2")
#
# fe.publish_table(
#     online_store=fe.get_online_store(name="risk-online-store"),
#     source_table_name=f"{CATALOG}.{RISK_SCHEMA}.txn_velocity",
#     online_table_name=f"{CATALOG}.{RISK_SCHEMA}.txn_velocity_online",
# )

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Label maturation, which sets the retraining schedule
# MAGIC
# MAGIC This is the constraint the article says matters more than any drift chart, and it is
# MAGIC the one most pipelines get wrong silently. A chargeback lands 30 to 90 days after the
# MAGIC transaction, so `is_fraud = 0` on a recent row does not mean "legitimate". It means
# MAGIC "not yet".
# MAGIC
# MAGIC Train on that and the recent, most relevant rows are systematically mislabelled in one
# MAGIC direction, because every label still in flight is a positive.
# MAGIC `02_risk_realtime_pipeline.py` section 2 measures what it costs.

# COMMAND ----------

spark.sql(f"""
CREATE OR REPLACE VIEW {CATALOG}.{RISK_SCHEMA}.matured_training_set AS
SELECT
  t.*,
  COALESCE(c.is_chargeback, FALSE) AS is_fraud
FROM {CATALOG}.{RISK_SCHEMA}.transactions t
LEFT JOIN {CATALOG}.{RISK_SCHEMA}.chargebacks c USING (transaction_id)
-- Only rows old enough that a chargeback would have arrived by now. Everything
-- newer is partially labelled and biased, and belongs to the rules, not the model.
WHERE t.event_ts < current_timestamp() - INTERVAL 90 DAYS
""")

# The same query, as a monitoring metric: how much of your recent traffic is
# untrainable right now. If this is not on a dashboard somewhere, the retraining
# cadence is being set by a cron expression rather than by the data.
spark.sql(f"""
SELECT
  date_trunc('WEEK', event_ts) AS week,
  count(*) AS transactions,
  sum(CASE WHEN event_ts < current_timestamp() - INTERVAL 90 DAYS THEN 1 ELSE 0 END)
    / count(*) AS trainable_share
FROM {CATALOG}.{RISK_SCHEMA}.transactions
GROUP BY 1 ORDER BY 1 DESC
""").display()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Serving, with inference tables on
# MAGIC
# MAGIC Route-optimized, because the budget is 50 to 100 ms end to end including the feature
# MAGIC lookup. Inference tables give you the request log for free, which is what makes the
# MAGIC threshold recomputable from measured costs later, and what makes the selective label
# MAGIC problem in `04_feedback_loops.py` visible at all.

# COMMAND ----------

# **Costs money.** A serving endpoint bills while it is up.
#
# from databricks.sdk import WorkspaceClient
# from databricks.sdk.service.serving import (
#     EndpointCoreConfigInput, ServedEntityInput, AutoCaptureConfigInput)
#
# w = WorkspaceClient()
# w.serving_endpoints.create(
#     name="risk-scorer",
#     config=EndpointCoreConfigInput(
#         served_entities=[ServedEntityInput(
#             entity_name=f"{CATALOG}.{RISK_SCHEMA}.scorer",
#             entity_version=None,
#             entity_alias="champion",          # promotion changes what serves
#             workload_size="Small",
#             scale_to_zero_enabled=False,      # a cold start does not fit in 100 ms
#         )],
#         auto_capture_config=AutoCaptureConfigInput(
#             catalog_name=CATALOG,
#             schema_name=RISK_SCHEMA,
#             table_name_prefix="risk_scorer",  # -> risk_scorer_payload
#             enabled=True,
#         ),
#     ),
# )

# COMMAND ----------

# MAGIC %md
# MAGIC ### Calling it, with the budget enforced by the caller
# MAGIC
# MAGIC The timeout is your latency budget, not the library default. Without one, the client
# MAGIC waits as long as the endpoint wants, and a scoring outage becomes a checkout outage.
# MAGIC
# MAGIC What happens on timeout is a policy decision and it is not the same for every action.
# MAGIC Most commerce flows fail open. Withdrawals and password changes fail closed, because
# MAGIC the downside is not symmetric. `03_live_scoring_demo.py` runs both.

# COMMAND ----------

import requests

FAIL_MODES = {"checkout": "open", "login": "open",
              "withdrawal": "closed", "password_change": "closed"}


def score(transaction, action_type="checkout", host=HOST, token=TOKEN):    # noqa: F821
    try:
        response = requests.post(
            f"{host}/serving-endpoints/risk-scorer/invocations",
            headers={"Authorization": f"Bearer {token}"},
            json={"dataframe_records": [transaction]},
            timeout=0.08,                     # the budget, not the library default
        )
        return response.json()["predictions"][0], None
    except requests.RequestException:
        if FAIL_MODES.get(action_type, "open") == "closed":
            return None, "fail closed: step-up verification required"
        return None, "fail open: allowed, flagged for async review"

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Selective labels: buy back what blocking destroys
# MAGIC
# MAGIC A blocked transaction never completes, so no chargeback can ever arrive against it,
# MAGIC so no label will ever exist. Retrain on what was allowed and the model goes blind
# MAGIC exactly where it acts.
# MAGIC
# MAGIC In `04_feedback_loops.py` this costs 0.25 PR-AUC, and letting 2% of would-be-blocked
# MAGIC traffic through - with an inverse-propensity weight, so the sample still stands for
# MAGIC the population - buys back roughly two thirds of it.
# MAGIC
# MAGIC The exploration slice belongs in the decision layer, not in the model, so that the
# MAGIC rate is a versioned business decision with a number attached.

# COMMAND ----------

EXPLORATION_RATE = 0.02

spark.sql(f"""
CREATE OR REPLACE VIEW {CATALOG}.{RISK_SCHEMA}.next_training_set AS
SELECT
  f.*,
  o.is_fraud,
  -- Rows that only exist because we deliberately let them through stand for all
  -- the ones we did not. Without the weight, the exploration slice is a biased
  -- sample of the block band rather than a fix for one.
  CASE WHEN l.final_action = 'explore' THEN 1.0 / {EXPLORATION_RATE} ELSE 1.0 END
    AS sample_weight
FROM {CATALOG}.{RISK_SCHEMA}.risk_scorer_payload l
JOIN {CATALOG}.{RISK_SCHEMA}.features f USING (transaction_id)
JOIN {CATALOG}.{RISK_SCHEMA}.outcomes o USING (transaction_id)
WHERE l.final_action IN ('allow', 'challenge', 'explore')     -- outcomes exist
   OR l.final_action = 'review'                               -- analyst decided
""")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Monitoring, on two clocks
# MAGIC
# MAGIC Same statistics on both sides, wired to different cadences and different thresholds.
# MAGIC Churn drifts over months. Fraud has an adversary, and its labels are 90 days behind,
# MAGIC so the only monitoring that can run daily is monitoring that needs no outcome.
# MAGIC
# MAGIC One finding from `05_monitoring_two_clocks.py` worth carrying over: a card-testing
# MAGIC wave was 0.7% of traffic, which never moved a population-level PSI at all. What
# MAGIC caught it the next day was the **flag rate** - the share of traffic scoring above the
# MAGIC challenge band. Alert on that, not only on distribution drift.

# COMMAND ----------

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.catalog import MonitorInferenceLog, MonitorInferenceLogProblemType

w = WorkspaceClient()

# Risk: daily, tight.
w.quality_monitors.create(
    table_name=f"{CATALOG}.{RISK_SCHEMA}.risk_scorer_payload",
    assets_dir=f"/Workspace/Users/{w.current_user.me().user_name}/monitors/risk",
    output_schema_name=f"{CATALOG}.{RISK_SCHEMA}",
    inference_log=MonitorInferenceLog(
        problem_type=MonitorInferenceLogProblemType.PROBLEM_TYPE_CLASSIFICATION,
        prediction_col="probability",
        label_col="is_fraud",                 # joined in 90 days late, and that is fine
        timestamp_col="event_ts",
        model_id_col="model_version",
        granularities=["1 day"],
    ),
    schedule=None,                            # or a daily cron
)

# Churn: monthly is enough, and weekly would produce the same decisions with more pages.
w.quality_monitors.create(
    table_name=f"{CATALOG}.{CHURN_SCHEMA}.nightly_scores",
    assets_dir=f"/Workspace/Users/{w.current_user.me().user_name}/monitors/churn",
    output_schema_name=f"{CATALOG}.{CHURN_SCHEMA}",
    inference_log=MonitorInferenceLog(
        problem_type=MonitorInferenceLogProblemType.PROBLEM_TYPE_CLASSIFICATION,
        prediction_col="churn_probability",
        label_col="churned",
        timestamp_col="scored_on",
        model_id_col="model_version",
        granularities=["1 month"],
    ),
    schedule=None,
)

# COMMAND ----------

# The flag rate: the label-free alarm that actually catches a small-volume attack.
spark.sql(f"""
SELECT
  date_trunc('DAY', event_ts) AS day,
  count(*) AS scored,
  avg(CASE WHEN probability >= 0.30 THEN 1.0 ELSE 0.0 END) AS flag_rate
FROM {CATALOG}.{RISK_SCHEMA}.risk_scorer_payload
GROUP BY 1 ORDER BY 1 DESC
""").display()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Keep the paths separate
# MAGIC
# MAGIC One governance layer, two pipelines, and no shared serving path:
# MAGIC
# MAGIC - **churn** — offline time series feature tables, a Lakeflow Job, batch scoring, a
# MAGIC   holdout nobody touches, monthly monitoring.
# MAGIC - **risk** — Structured Streaming, an online store, a route-optimized endpoint,
# MAGIC   inference tables, rules beside the model, daily monitoring.
# MAGIC
# MAGIC Mixing them adds latency to the side that does not need it and gives the side that
# MAGIC does a jumpy, untrusted score. Unity Catalog is what lets them stay separate without
# MAGIC becoming two platforms.
