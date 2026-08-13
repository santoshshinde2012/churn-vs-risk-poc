"""Stable holdout assignment.

The holdout only works if membership never moves. A per-run random draw
reshuffles it whenever the row order or the population changes, and a holdout
that moves measures nothing: last month's control group is this month's treated
group, and the difference between them is noise plus a reshuffle.

Hashing the entity id makes membership a property of the customer. It is stable
across runs, machines, pipeline rewrites and language changes, it needs no
state anywhere, and it stays balanced as the book grows.
"""

from __future__ import annotations

import hashlib

import numpy as np

BUCKETS = 10_000


def bucket(entity_id: str, salt: str) -> int:
    """Deterministic bucket in [0, BUCKETS) for one entity."""
    digest = hashlib.md5(f"{salt}:{entity_id}".encode()).hexdigest()[:8]
    return int(digest, 16) % BUCKETS


def assign(entity_ids, share: float, salt: str = "retention-holdout-v1") -> np.ndarray:
    """Boolean mask: True where the entity is in the holdout.

    Changing `salt` re-draws the whole holdout, so treat it as a version
    number and change it only when you deliberately want a fresh control group.
    """
    if not 0.0 <= share < 1.0:
        raise ValueError(f"holdout share must be in [0, 1), got {share}")
    cutoff = int(share * BUCKETS)
    return np.array([bucket(str(entity_id), salt) < cutoff for entity_id in entity_ids])
