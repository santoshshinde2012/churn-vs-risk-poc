"""Plain rules that run beside the model, as detectors in their own right.

Two reasons they exist, both from the article:

  1. A confirmed-bad device is not a probability. Rules express certainty the
     model cannot, and they never need a training set.
  2. Fraud tactics change faster than labels mature. Between a tactic change
     and the retrain that learns it, rules are the only thing covering the gap
     - and on this data the gap is 90 days wide, because that is how long a
     chargeback takes to confirm.

A rule can only escalate. It never downgrades the model's decision, so a rule
misfiring can cost friction but can never wave a high-score event through.
"""

from __future__ import annotations

from dataclasses import dataclass

# Severity order, mild to severe. Escalation means moving right, never left.
SEVERITY = ("none", "allow", "nurture", "call", "challenge", "review", "block")


def rank(action: str) -> int:
    return SEVERITY.index(action) if action in SEVERITY else 0


def escalate(current: str, proposed: str) -> str:
    """Return whichever action is more severe."""
    return proposed if rank(proposed) > rank(current) else current


@dataclass
class RuleHit:
    rule: str
    action: str
    detail: str


class RuleEngine:
    """Evaluates a policy's `rules` block against one scored event."""

    def __init__(self, policy, blocked_devices=None):
        self.policy = policy
        self.config = policy.rules or {}
        self.blocked_devices = set(blocked_devices or ())

    # ------------------------------------------------------------------ api --
    def evaluate(self, record: dict) -> list[RuleHit]:
        """Every rule that fires for this record, in configured order."""
        hits: list[RuleHit] = []
        for check in (self._blocklist, self._velocity, self._new_account, self._network):
            hit = check(record)
            if hit is not None:
                hits.append(hit)
        return hits

    def apply(self, model_action: str, record: dict) -> tuple[str, list[RuleHit]]:
        """Combine the model's action with the rules. Rules can only escalate."""
        hits = self.evaluate(record)
        action = model_action
        for hit in hits:
            action = escalate(action, hit.action)
        return action, hits

    # ---------------------------------------------------------------- rules --
    def _blocklist(self, record):
        config = self.config.get("blocklist")
        if not config:
            return None
        device = record.get("device")
        if device is not None and device in self.blocked_devices:
            return RuleHit("blocklist", config.get("action", "block"),
                           f"device {device} is on the blocklist")
        return None

    def _velocity(self, record):
        config = self.config.get("velocity")
        if not config:
            return None
        per_minute = record.get("txn_60s", 0)
        per_ten = record.get("txn_10min", 0)
        if per_minute > config.get("max_txn_60s", float("inf")):
            return RuleHit("velocity", config.get("action", "review"),
                           f"{per_minute:.0f} txns in 60s (limit {config['max_txn_60s']})")
        if per_ten > config.get("max_txn_10min", float("inf")):
            return RuleHit("velocity", config.get("action", "review"),
                           f"{per_ten:.0f} txns in 10min (limit {config['max_txn_10min']})")
        return None

    def _new_account(self, record):
        config = self.config.get("new_account")
        if not config:
            return None
        age = record.get("account_age_days")
        amount = record.get("amount", 0.0)
        if age is None:
            return None
        if age <= config.get("max_age_days", 0) and amount > config.get("max_amount", float("inf")):
            return RuleHit("new_account", config.get("action", "challenge"),
                           f"${amount:,.0f} on an account {age:.0f} days old")
        return None

    def _network(self, record):
        config = self.config.get("network")
        if not config:
            return None
        shared = record.get("device_accounts", 0)
        if shared > config.get("max_accounts_per_device", float("inf")):
            return RuleHit("network", config.get("action", "review"),
                           f"{shared:.0f} accounts share this device "
                           f"(limit {config['max_accounts_per_device']})")
        return None


def build_blocklist(records, labels, min_confirmed: int = 2, min_fraud_share: float = 0.9):
    """Devices that are the attacker's, not the victim's.

    Two conditions, and both are load-bearing. Neither is obvious, and dropping
    either one wrecks the insult rate on this dataset:

    Devices, not cards. A compromised card gets reissued, so blocklisting it
    punishes the victim for having been defrauded and turns every one of their
    later transactions into a false decline. A device fingerprint stays with
    whoever is holding the device.

    And only devices that are almost entirely fraudulent. A session hijack
    happens on the customer's own laptop: that device has confirmed fraud on
    it and a long legitimate history underneath. Blocklisting it is the card
    mistake wearing a different hat. Requiring a high fraud share keeps the
    attacker's throwaway devices and the ring's shared handsets, and leaves the
    victim's own hardware alone.

    On this dataset, dropping the fraud-share condition takes the insult rate
    from roughly 0.01% to over 1%, which is the whole article in one line of
    configuration.

    In production this list is fed by the review queue and by confirmed
    chargebacks, so it lags exactly the way labels lag. Here it is built from
    the matured training window only, so it carries no future knowledge.
    """
    from collections import Counter

    fraud_hits, total_hits = Counter(), Counter()
    for record, label in zip(records, labels):
        device = record.get("device")
        if device is None:
            continue
        total_hits[device] += 1
        if label == 1:
            fraud_hits[device] += 1

    return {
        device for device, n in fraud_hits.items()
        if n >= min_confirmed and n / total_hits[device] >= min_fraud_share
    }
