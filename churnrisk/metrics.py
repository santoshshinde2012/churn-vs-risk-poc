"""Metrics that survive class imbalance, plus the calibration checks the
dollar threshold depends on.

The article's argument is that a cost-based threshold is only meaningful if a
score of 0.9 really means 90%. Asserting calibration by wrapping a model in
CalibratedClassifierCV is not the same as measuring it, so everything needed to
measure it lives here.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def precision_at_k(y_true, y_score, k: int) -> float:
    """Share of real positives in the k highest-scoring rows.

    k is your team's capacity: how many customers retention can call, how many
    cases the fraud queue can review.
    """
    y_true = np.asarray(y_true)
    k = min(int(k), len(y_true))
    if k == 0:
        return float("nan")
    top = np.argsort(y_score)[::-1][:k]
    return float(y_true[top].mean())


def recall_at_k(y_true, y_score, k: int) -> float:
    """Share of all positives captured in the k highest-scoring rows."""
    y_true = np.asarray(y_true)
    positives = y_true.sum()
    if positives == 0:
        return float("nan")
    k = min(int(k), len(y_true))
    top = np.argsort(y_score)[::-1][:k]
    return float(y_true[top].sum() / positives)


def lift_at_k(y_true, y_score, k: int) -> float:
    """precision@k divided by the base rate. 1.0 means no better than random."""
    base = float(np.asarray(y_true).mean())
    if base == 0:
        return float("nan")
    return precision_at_k(y_true, y_score, k) / base


def brier_score(y_true, y_prob) -> float:
    """Mean squared error of the probabilities. Lower is better."""
    y_true = np.asarray(y_true, dtype=float)
    return float(np.mean((np.asarray(y_prob, dtype=float) - y_true) ** 2))


def expected_calibration_error(y_true, y_prob, n_bins: int = 10) -> float:
    """Weighted mean gap between predicted probability and observed frequency.

    Equal-width bins over [0, 1]. ECE of 0.03 means that, on average, a
    predicted probability is off by 3 percentage points.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(y_prob, edges[1:-1], right=False), 0, n_bins - 1)

    total, error = len(y_prob), 0.0
    for b in range(n_bins):
        mask = idx == b
        if not mask.any():
            continue
        error += mask.sum() / total * abs(y_prob[mask].mean() - y_true[mask].mean())
    return float(error)


def reliability_curve(y_true, y_prob, n_bins: int = 10, strategy: str = "quantile"):
    """Points for a reliability diagram: (mean predicted, observed rate, count).

    Quantile bins are the default because at fraud-level imbalance almost every
    score lands in the first equal-width bin, which makes the diagram useless.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)

    if strategy == "quantile":
        edges = np.unique(np.quantile(y_prob, np.linspace(0, 1, n_bins + 1)))
        edges[0], edges[-1] = -np.inf, np.inf
    else:
        edges = np.linspace(0.0, 1.0, n_bins + 1)
        edges[0], edges[-1] = -np.inf, np.inf

    idx = np.digitize(y_prob, edges[1:-1], right=False)
    predicted, observed, counts = [], [], []
    for b in range(len(edges) - 1):
        mask = idx == b
        if mask.sum() < 5:                      # too few rows to say anything
            continue
        predicted.append(float(y_prob[mask].mean()))
        observed.append(float(y_true[mask].mean()))
        counts.append(int(mask.sum()))
    return np.array(predicted), np.array(observed), np.array(counts)


def cost_threshold(fp_cost: float, fn_cost: float) -> float:
    """The threshold implied by the two costs.

    Act when the expected cost of not acting exceeds the expected cost of
    acting:  p * fn_cost > (1 - p) * fp_cost, so p > fp / (fp + fn).
    With the article's $500 insult and $150 chargeback that is 0.77, not 0.5.
    """
    if fp_cost < 0 or fn_cost < 0 or (fp_cost + fn_cost) == 0:
        raise ValueError("costs must be non-negative and not both zero")
    return float(fp_cost) / float(fp_cost + fn_cost)


def total_cost(y_true, y_prob, threshold: float, fp_cost: float, fn_cost: float) -> float:
    """Dollar cost of acting at `threshold`: insults plus missed positives."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    acted = y_prob >= threshold
    false_positives = int((acted & (y_true == 0)).sum())
    false_negatives = int((~acted & (y_true == 1)).sum())
    return float(false_positives * fp_cost + false_negatives * fn_cost)


def cost_curve(y_true, y_prob, fp_cost: float, fn_cost: float, n: int = 99):
    """Total cost across candidate thresholds, plus the empirical minimum."""
    thresholds = np.linspace(0.01, 0.99, n)
    costs = np.array([total_cost(y_true, y_prob, t, fp_cost, fn_cost) for t in thresholds])
    return thresholds, costs, float(thresholds[int(np.argmin(costs))])


def headline(y_true, y_prob) -> dict:
    """The small block of numbers every model in this repo reports."""
    y_true = np.asarray(y_true)
    base = float(y_true.mean())
    out = {
        "n": int(len(y_true)),
        "base_rate": base,
        "pr_auc": float("nan"),
        "roc_auc": float("nan"),
        "brier": brier_score(y_true, y_prob),
        "ece": expected_calibration_error(y_true, y_prob),
    }
    if 0 < y_true.sum() < len(y_true):          # both classes present
        out["pr_auc"] = float(average_precision_score(y_true, y_prob))
        out["roc_auc"] = float(roc_auc_score(y_true, y_prob))
        out["pr_auc_lift"] = out["pr_auc"] / base if base else float("nan")
    return out
