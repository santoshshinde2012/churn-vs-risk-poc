"""Label-free drift detection.

The point of monitoring on the risk side is that you cannot wait for labels.
Chargebacks land 30 to 90 days late, so a label-based performance chart tells
you about an attack a quarter after it started. A monitor that only looks at
feature and score distributions can fire the next day.

PSI (population stability index) and the two-sample KS statistic are the two
standard ways to do that, and neither needs an outcome.
"""

from __future__ import annotations

import numpy as np

# Conventional PSI reading, used across the industry:
PSI_WATCH = 0.10      # something moved, look at it
PSI_ALERT = 0.25      # the population is materially different


def psi(reference, current, n_bins: int = 10) -> float:
    """Population stability index between a reference and a current sample.

    Bins come from the reference quantiles, so the reference is uniform across
    bins by construction and any imbalance is genuinely the current sample.
    """
    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)
    if len(reference) == 0 or len(current) == 0:
        return float("nan")

    edges = np.unique(np.quantile(reference, np.linspace(0, 1, n_bins + 1)))
    if len(edges) < 3:                       # near-constant feature, nothing to bin
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf

    ref_share = np.histogram(reference, bins=edges)[0] / len(reference)
    cur_share = np.histogram(current, bins=edges)[0] / len(current)

    # Floor the shares so an empty bin contributes a large but finite term.
    floor = 1e-4
    ref_share = np.clip(ref_share, floor, None)
    cur_share = np.clip(cur_share, floor, None)
    return float(np.sum((cur_share - ref_share) * np.log(cur_share / ref_share)))


def ks_statistic(reference, current) -> float:
    """Two-sample Kolmogorov-Smirnov statistic, the largest CDF gap.

    Implemented directly so the repo does not need scipy for one number.
    """
    reference = np.sort(np.asarray(reference, dtype=float))
    current = np.sort(np.asarray(current, dtype=float))
    if len(reference) == 0 or len(current) == 0:
        return float("nan")

    grid = np.sort(np.concatenate([reference, current]))
    cdf_ref = np.searchsorted(reference, grid, side="right") / len(reference)
    cdf_cur = np.searchsorted(current, grid, side="right") / len(current)
    return float(np.max(np.abs(cdf_ref - cdf_cur)))


class Monitor:
    """A reference window with its bin edges precomputed.

    Recomputing reference quantiles on every check is the obvious way to write
    this and it makes a daily monitor quadratic in the size of the reference.
    Fixing the edges once is also more correct: a monitor whose bins move with
    the reference is comparing against a moving target.
    """

    def __init__(self, reference_df, columns, score_ref=None, n_bins: int = 10,
                 max_rows: int = 25_000, seed: int = 0):
        if len(reference_df) > max_rows:      # a sample is plenty for bin edges
            reference_df = reference_df.sample(max_rows, random_state=seed)
            if score_ref is not None:
                score_ref = np.asarray(score_ref)[reference_df.index.values]

        self.columns = list(columns)
        self.n_bins = n_bins
        self._edges = {c: _edges(reference_df[c].values, n_bins) for c in self.columns}
        self._shares = {c: _shares(reference_df[c].values, self._edges[c])
                        for c in self.columns}
        self._reference = {c: reference_df[c].values for c in self.columns}

        self.score_ref = None if score_ref is None else np.asarray(score_ref)
        if self.score_ref is not None:
            self._edges["__score__"] = _edges(self.score_ref, n_bins)
            self._shares["__score__"] = _shares(self.score_ref, self._edges["__score__"])

    def check(self, current_df, score_cur=None) -> dict:
        report = {}
        for column in self.columns:
            values = current_df[column].values
            report[column] = {
                "psi": _psi_with_edges(self._shares[column], values, self._edges[column]),
                "ks": ks_statistic(self._reference[column], values),
            }
        if score_cur is not None and self.score_ref is not None:
            score_cur = np.asarray(score_cur)
            report["__score__"] = {
                "psi": _psi_with_edges(self._shares["__score__"], score_cur,
                                       self._edges["__score__"]),
                "ks": ks_statistic(self.score_ref, score_cur),
                "mean_shift": float(np.mean(score_cur) - np.mean(self.score_ref)),
            }
        return report


def _edges(values, n_bins: int):
    edges = np.unique(np.quantile(np.asarray(values, dtype=float),
                                  np.linspace(0, 1, n_bins + 1)))
    if len(edges) < 3:
        return None                            # near-constant, nothing to bin
    edges[0], edges[-1] = -np.inf, np.inf
    return edges


def _shares(values, edges, floor: float = 1e-4):
    if edges is None:
        return None
    values = np.asarray(values, dtype=float)
    return np.clip(np.histogram(values, bins=edges)[0] / len(values), floor, None)


def _psi_with_edges(ref_shares, current, edges, floor: float = 1e-4) -> float:
    if edges is None or ref_shares is None:
        return 0.0
    current = np.asarray(current, dtype=float)
    if len(current) == 0:
        return float("nan")
    cur_shares = np.clip(np.histogram(current, bins=edges)[0] / len(current), floor, None)
    return float(np.sum((cur_shares - ref_shares) * np.log(cur_shares / ref_shares)))


def flag_rate(scores, threshold: float) -> float:
    """Share of scored events at or above a decision threshold.

    The most sensitive label-free signal there is on the risk side, and the one
    that actually catches a small-volume attack. A card-testing wave might be
    1% of a day's traffic, which will not move a population-level PSI by a
    measurable amount - but if most of it lands above the challenge band, the
    flag rate can double while every distribution still looks normal.
    """
    scores = np.asarray(scores)
    if len(scores) == 0:
        return float("nan")
    return float((scores >= threshold).mean())


def rate_z_score(flagged: int, total: int, baseline_rate: float) -> float:
    """How many standard errors today's flag count is above the baseline rate.

    A raw ratio is the obvious way to alarm on a flag rate and it does not
    work. At a 0.4% baseline a day of 1,200 transactions expects about five
    flags, and five events double by chance roughly twice a month - so
    "flag rate has doubled" fires constantly on quiet traffic and tells you
    nothing on busy traffic.

    A binomial z-score prices the count against the volume that produced it,
    which is what makes the same threshold mean the same thing on a Sunday and
    on Black Friday. Pair it with a trailing window: a sustained attack shows up
    in a week of data far more clearly than in any single day.
    """
    if total <= 0 or not 0.0 < baseline_rate < 1.0:
        return float("nan")
    expected = total * baseline_rate
    standard_error = np.sqrt(total * baseline_rate * (1.0 - baseline_rate))
    if standard_error == 0:
        return float("nan")
    return float((flagged - expected) / standard_error)


def drift_report(reference_df, current_df, columns, score_ref=None, score_cur=None) -> dict:
    """PSI and KS for every column, plus the score distribution if given.

    The score distribution is usually the earliest warning: it aggregates every
    feature the model actually uses, weighted the way the model weights them.
    """
    report = {}
    for col in columns:
        report[col] = {
            "psi": psi(reference_df[col].values, current_df[col].values),
            "ks": ks_statistic(reference_df[col].values, current_df[col].values),
        }
    if score_ref is not None and score_cur is not None:
        report["__score__"] = {
            "psi": psi(score_ref, score_cur),
            "ks": ks_statistic(score_ref, score_cur),
            "mean_shift": float(np.mean(score_cur) - np.mean(score_ref)),
        }
    return report


def worst(report: dict) -> tuple[str, float]:
    """The single most-drifted signal in a report, by PSI."""
    ranked = sorted(
        ((name, stats["psi"]) for name, stats in report.items() if np.isfinite(stats["psi"])),
        key=lambda item: item[1],
        reverse=True,
    )
    return ranked[0] if ranked else ("", float("nan"))


def verdict(report: dict, alert: float = PSI_ALERT, watch: float = PSI_WATCH) -> str:
    """Turn a report into one of three words, using the worst signal in it."""
    _, worst_psi = worst(report)
    if not np.isfinite(worst_psi):
        return "unknown"
    if worst_psi >= alert:
        return "alert"
    if worst_psi >= watch:
        return "watch"
    return "stable"
