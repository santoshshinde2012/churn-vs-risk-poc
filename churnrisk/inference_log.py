"""The local stand-in for Databricks inference tables.

On Databricks, a Model Serving endpoint with inference tables enabled writes
every request, response and timestamp to a Delta table for free. Locally that
is a Parquet file, appended to in batches.

It matters more than it looks. The inference log is what lets you:

  - recompute the decision threshold from measured costs instead of guesses
  - reconstruct the training set for the next round, with the decisions the
    model actually took attached (which is where selective labels show up)
  - attribute a shift in the decision mix to a policy version
  - monitor score drift without waiting for labels

Every row carries the model version and the policy version, so a change in
behaviour can always be traced to one or the other.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .config import INFERENCE

SCHEMA = [
    "request_id", "logged_at", "event_ts", "entity_id",
    "model_name", "model_version", "policy_version",
    "probability", "model_action", "final_action", "rules_fired",
    "latency_ms", "outcome_available", "outcome",
]


class InferenceLog:
    """Append-only request log, buffered in memory and flushed to Parquet."""

    def __init__(self, name: str, model_name: str, model_version: str, policy_version: str,
                 directory: Path | None = None):
        self.path = Path(directory or INFERENCE) / f"{name}.parquet"
        self.model_name = model_name
        self.model_version = str(model_version)
        self.policy_version = str(policy_version)
        self._buffer: list[dict] = []
        self._next_id = 0

    def record(self, *, entity_id, probability, model_action, final_action,
               event_ts=None, rules_fired=(), latency_ms=None,
               outcome=None, outcome_available=False) -> None:
        """Log one scored request.

        `outcome_available=False` is the normal case at request time: the label
        does not exist yet, and on the risk side it will not exist for months.
        """
        self._next_id += 1
        self._buffer.append({
            "request_id": f"{self.model_name}-{self._next_id:09d}",
            "logged_at": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "event_ts": float(event_ts) if event_ts is not None else None,
            "entity_id": str(entity_id),
            "model_name": self.model_name,
            "model_version": self.model_version,
            "policy_version": self.policy_version,
            "probability": None if probability is None else float(probability),
            "model_action": model_action,
            "final_action": final_action,
            "rules_fired": ",".join(rules_fired) if rules_fired else "",
            "latency_ms": None if latency_ms is None else float(latency_ms),
            "outcome_available": bool(outcome_available),
            "outcome": None if outcome is None else int(outcome),
        })

    def record_many(self, frame: pd.DataFrame) -> None:
        """Bulk path for the batch pipelines, which score millions of rows."""
        for row in frame.to_dict("records"):
            self.record(**row)

    def flush(self) -> Path:
        """Write the buffer to Parquet. Appends if the file already exists."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        new = pd.DataFrame(self._buffer, columns=SCHEMA)
        if self.path.exists():
            new = pd.concat([pd.read_parquet(self.path), new], ignore_index=True)
        new.to_parquet(self.path, index=False)
        self._buffer.clear()
        return self.path

    def __len__(self) -> int:
        return len(self._buffer)


def read_log(name: str, directory: Path | None = None) -> pd.DataFrame:
    path = Path(directory or INFERENCE) / f"{name}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"no inference log at {path}; run the pipeline first")
    return pd.read_parquet(path)


def decision_mix(log: pd.DataFrame) -> pd.DataFrame:
    """Share of each final action, by policy version.

    This is the table you look at the morning after a policy change.
    """
    counts = log.groupby(["policy_version", "final_action"]).size().rename("n").reset_index()
    totals = counts.groupby("policy_version")["n"].transform("sum")
    counts["share"] = counts["n"] / totals
    return counts.sort_values(["policy_version", "n"], ascending=[True, False])
