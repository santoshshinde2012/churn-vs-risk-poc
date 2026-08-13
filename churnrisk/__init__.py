"""Shared building blocks for the churn-vs-risk POC.

The pipeline scripts (01 to 05) are meant to be read top to bottom, so anything
that would be scaffolding in them lives here instead:

    config          paths, seeds and the constants worth changing
    metrics         imbalance-safe metrics and the calibration checks
    drift           PSI and KS, for monitoring that does not wait for labels
    holdout         stable, hash-based control-group assignment
    policy          the versioned decision layer, loaded from policies/*.yaml
    rules           plain detectors that run beside the model
    registry        MLflow tracking and @champion promotion
    inference_log   the local stand-in for inference tables
    churn_data      simulated engagement history, with a real time axis
    risk_data       simulated card stream, with rings and late labels
    viz             one plot style for every figure
"""

from . import (  # noqa: F401
    churn_data,
    config,
    drift,
    holdout,
    inference_log,
    metrics,
    policy,
    registry,
    risk_data,
    rules,
    viz,
)

__all__ = [
    "churn_data", "config", "drift", "holdout", "inference_log", "metrics",
    "policy", "registry", "risk_data", "rules", "viz",
]
