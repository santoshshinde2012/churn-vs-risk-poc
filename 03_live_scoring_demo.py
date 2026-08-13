"""The shield, live: a scoring endpoint, a decision layer and an outage.

This starts a local HTTP scoring service holding the model that
02_risk_realtime_pipeline.py promoted to @champion, then sends transactions
through it the way an application would:

  - every call has a hard budget, read from the policy file, not from the
    library default
  - the model returns a probability and nothing else; the decision layer maps
    it to allow / challenge / review / block
  - rules run beside the model, and they do not need the endpoint
  - half way through, the endpoint stops answering
  - checkout and login fail open. Withdrawals and password changes fail
    closed, because an irreversible action has an asymmetric downside
  - every request is logged with its latency and the policy version

Then the same recorded probabilities are replayed under a second policy file,
to show the decision mix moving with no retraining and no redeploy.

Run 02 first, then: python 03_live_scoring_demo.py
"""

from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import joblib
import numpy as np
import pandas as pd
import requests

from churnrisk import registry, viz
from churnrisk.config import METRICS, MODELS, PLOTS, SEED, TABLES, ensure_dirs
from churnrisk.inference_log import InferenceLog, decision_mix
from churnrisk.policy import load_policy
from churnrisk.rules import RuleEngine

PORT = 8091
MODEL_NAME = "ml.risk.scorer"
FEATURE_LOOKUP_MS = 3.0        # the online feature store round trip, simulated

ensure_dirs()
policy = load_policy("risk_policy_v1")

bundle = joblib.load(MODELS / "risk_model.joblib")
FEATURES = bundle["features"]
BLOCKED_DEVICES = bundle["blocked_devices"]

# Load the model the way a serving endpoint would: by alias, not by file path.
# A promotion in the registry changes what serves, with no code change here.
try:
    MODEL = registry.load(MODEL_NAME)
    SOURCE = f"models:/{MODEL_NAME}@champion"
except Exception as error:                     # registry wiped or never populated
    MODEL = bundle["model"]
    SOURCE = f"{MODELS.name}/risk_model.joblib (registry unavailable: {type(error).__name__})"

engine = RuleEngine(policy, BLOCKED_DEVICES)
OUTAGE = {"on": False}


class ScoreHandler(BaseHTTPRequestHandler):
    """Stands in for a Mosaic AI Model Serving endpoint."""

    def do_POST(self):
        if OUTAGE["on"]:
            time.sleep(1.0)                    # hangs rather than erroring: the worse case
        try:
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            time.sleep(FEATURE_LOOKUP_MS / 1000.0)      # online feature store lookup
            row = pd.DataFrame([body["record"]], columns=FEATURES)
            probability = float(MODEL.predict_proba(row)[0, 1])
            payload = json.dumps({"probability": round(probability, 6)}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
        except (BrokenPipeError, ConnectionResetError):
            pass                               # the client already gave up on us

    def log_message(self, *args):
        pass


def handle_error(self, request, client_address):
    """The client abandoning a hung request is expected here, not an incident."""


ThreadingHTTPServer.handle_error = handle_error


def score(record: dict, action_type: str):
    """One scored decision, with the budget enforced by the caller.

    Returns (action, probability, latency_ms, rules that fired, note).
    """
    started = time.perf_counter()
    try:
        response = requests.post(
            f"http://127.0.0.1:{PORT}/score",
            json={"record": {k: float(record[k]) for k in FEATURES}},
            timeout=policy.timeout_seconds,     # the budget, not the library default
        )
        probability = response.json()["probability"]
        model_action = policy.decide(probability)
        note = ""
    except requests.RequestException:
        # The model did not answer. The policy decides what that means, and it
        # is not the same answer for a checkout and for a withdrawal.
        probability = None
        model_action, note = policy.fail_action(action_type)

    elapsed_ms = (time.perf_counter() - started) * 1000.0

    # The rules do not need the endpoint. They are the reason an outage
    # degrades the defence instead of removing it.
    final_action, hits = engine.apply(model_action, record)
    if hits and probability is None:
        note = f"model down; {hits[0].rule} rule fired: {hits[0].detail}"
    return final_action, probability, elapsed_ms, [hit.rule for hit in hits], note


def main():
    server = ThreadingHTTPServer(("127.0.0.1", PORT), ScoreHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    time.sleep(0.4)

    live = pd.read_csv(TABLES / "live_sample.csv").reset_index(drop=True)
    rng = np.random.default_rng(SEED)
    # A realistic mix, so both fail modes get exercised.
    action_types = rng.choice(
        ["checkout", "login", "withdrawal", "password_change"],
        size=len(live), p=[0.55, 0.25, 0.12, 0.08])

    log = InferenceLog("risk_live", MODEL_NAME, bundle["model_version"], policy.version)

    print("=" * 92)
    print("LIVE SCORING DEMO")
    print("=" * 92)
    print(f"model:  {SOURCE}")
    print(f"policy: {policy.summary()}")
    print(f"budget: {policy.timeout_ms} ms end to end, including the feature lookup")
    print("fail modes: " + ", ".join(f"{k}={v}" for k, v in policy.fail_modes.items()))
    print("-" * 92)
    print(f"{'#':>3} {'action_type':<16} {'decision':<10} {'p(fraud)':>9} "
          f"{'latency':>9} {'truth':<6} note")
    print("-" * 92)

    outage_at = len(live) // 2
    rows = []
    for i, row in live.iterrows():
        if i == outage_at:
            OUTAGE["on"] = True
            print("-" * 92)
            print("... the scoring endpoint stops answering here (simulated outage) ...")
            print("-" * 92)

        record = row.to_dict()
        action_type = action_types[i]
        action, probability, elapsed_ms, hits, note = score(record, action_type)

        truth = "FRAUD" if row["is_fraud"] == 1 else "legit"
        shown = f"{probability:.3f}" if probability is not None else "--"
        print(f"{i + 1:>3} {action_type:<16} {action.upper():<10} {shown:>9} "
              f"{elapsed_ms:>7.1f}ms {truth:<6} {note}")

        log.record(entity_id=f"live-{i}", probability=probability,
                   model_action=action if probability is not None else "unavailable",
                   final_action=action, rules_fired=hits, latency_ms=elapsed_ms,
                   outcome=int(row["is_fraud"]), outcome_available=True)
        rows.append({"i": i, "action_type": action_type, "action": action,
                     "probability": probability, "latency_ms": elapsed_ms,
                     "is_fraud": int(row["is_fraud"]), "outage": OUTAGE["on"],
                     "rules": ",".join(hits)})

    server.shutdown()
    frame = pd.DataFrame(rows)

    # ------------------------------------------------------------ latency --
    healthy = frame[~frame["outage"]]["latency_ms"]
    down = frame[frame["outage"]]["latency_ms"]
    print("-" * 92)
    print("LATENCY")
    print(f"  healthy: p50 {healthy.quantile(0.5):.1f} ms   p95 {healthy.quantile(0.95):.1f} ms   "
          f"max {healthy.max():.1f} ms   budget {policy.timeout_ms} ms")
    print(f"  outage:  every call gives up at the budget, "
          f"p50 {down.quantile(0.5):.1f} ms   max {down.max():.1f} ms")
    print("  the timeout is the point. without it the library would wait for as")
    print("  long as the endpoint wanted, and a scoring outage would become a")
    print("  checkout outage.")

    # ------------------------------------------------------- what happened --
    print("\nWHAT THE OUTAGE COST")
    touched = ("BLOCK", "REVIEW", "CHALLENGE")
    for label, part in (("healthy", frame[~frame["outage"]]), ("outage", frame[frame["outage"]])):
        fraud = part[part["is_fraud"] == 1]
        caught = int(fraud["action"].str.upper().isin(touched).sum())
        print(f"  {label:<8} {len(part):>2} requests, {len(fraud)} of them fraud, "
              f"{caught} stopped or flagged")

    outage_frame = frame[frame["outage"]]
    failed_closed = outage_frame[outage_frame["action_type"].isin(
        [k for k, v in policy.fail_modes.items() if v == "closed"])]
    rules_saves = outage_frame[(outage_frame["is_fraud"] == 1) & (outage_frame["rules"] != "")]
    print(f"\n  {len(failed_closed)} of the {len(outage_frame)} requests during the outage were "
          "irreversible actions.")
    print("  those failed closed: verification required rather than waved through.")
    print(f"  {len(rules_saves)} fraudulent request(s) during the outage were caught by a rule")
    print("  with no model in the loop at all.")
    print("\n  everything else was allowed and flagged for asynchronous review.")
    print("  that is the honest cost of failing open, and it is the reason the")
    print("  flag queue exists rather than being a nice-to-have.")

    # ------------------------------------------- the policy is not the model --
    print("\n" + "-" * 92)
    print("CHANGING THE DECISION LAYER WITHOUT TOUCHING THE MODEL")
    print("-" * 92)

    v2 = load_policy("risk_policy_v2")
    print(f"v{policy.version}: block >= {policy.block_at:.2f}   "
          f"v{v2.version}: block >= {v2.block_at:.2f}   "
          f"(finance re-measured the cost of a false decline)")

    # Replayed over a real day's worth of traffic rather than the handful of
    # requests above, because a decision mix is the thing that shifts and
    # sixteen requests cannot show one.
    replay = np.load(MODELS / "risk_scores.npy")
    comparison = pd.DataFrame({
        f"v{policy.version}": [policy.decide(p) for p in replay],
        f"v{v2.version}": [v2.decide(p) for p in replay],
    })
    moved = int((comparison[f"v{policy.version}"] != comparison[f"v{v2.version}"]).sum())

    print(f"\nreplaying {len(replay):,} stored scores from 02 through both policy files:")
    print(pd.crosstab(comparison[f"v{policy.version}"], comparison[f"v{v2.version}"]).to_string())
    print(f"\n{moved:,} of {len(replay):,} decisions move ({moved / len(replay):.2%}). no retraining,")
    print("no redeploy, no data science involved. a threshold is a business input,")
    print("and the model artifact is the wrong place to keep one.")
    print("\nthat is also why the policy version is on every inference log row: when")
    print("the block rate jumps overnight, the first question is which of the two")
    print("things changed, and the log answers it without anyone guessing.")

    path = log.flush()
    print(f"\ninference log: {path.name} ({len(frame)} rows with latency and outcome attached)")
    print(decision_mix(pd.read_parquet(path)).to_string(index=False))

    # ------------------------------------------------------------- plot ----
    fig, ax = viz.new_axes(width=8.4)
    colours = [viz.ORANGE if outage else viz.BLUE for outage in frame["outage"]]
    ax.bar(frame["i"] + 1, frame["latency_ms"], color=colours, width=0.7)
    ax.axhline(policy.timeout_ms, color=viz.INK, linewidth=1.5, linestyle="--")
    ax.text(len(frame), policy.timeout_ms + 2, f" budget {policy.timeout_ms} ms",
            ha="right", color=viz.INK, fontsize=8)
    viz.finish(fig, ax, "Latency against the budget, outage included",
               "request", "latency (ms)", PLOTS / "risk_latency.png")

    summary = {
        "model_source": SOURCE,
        "policy_version": policy.version,
        "budget_ms": policy.timeout_ms,
        "latency_healthy": {"p50": float(healthy.quantile(0.5)),
                            "p95": float(healthy.quantile(0.95)),
                            "max": float(healthy.max())},
        "latency_outage_max": float(down.max()),
        "requests": len(frame),
        "outage_requests": int(frame["outage"].sum()),
        "failed_closed": int(len(failed_closed)),
        "rule_catches_during_outage": int(len(rules_saves)),
        "policy_swap": {"from": policy.version, "to": v2.version,
                        "block_at_from": policy.block_at, "block_at_to": v2.block_at,
                        "decisions_moved": moved, "of": int(len(replay))},
    }
    (METRICS / "live_metrics.json").write_text(json.dumps(summary, indent=2, default=float))
    print(f"metrics written: {METRICS.name}/live_metrics.json")


if __name__ == "__main__":
    main()
