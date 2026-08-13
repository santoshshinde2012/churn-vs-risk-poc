"""The decision layer, versioned separately from the model.

The article's rule: the model returns a probability and nothing else. A
decision layer decides what the probability means. If the bands are baked into
the model artifact then changing a business threshold means shipping a new
model, which is exactly what the separation is supposed to avoid.

So bands, costs, timeouts and fail behaviour live in policies/*.yaml, are
loaded at decision time, and carry a version that gets written into every
inference log row.

Two kinds of policy, because the two sides of the article decide differently:

  blocking  the risk side. Above a cost-derived threshold the action is
            refused. Bands underneath it hand off to cheaper interventions.
  tiered    the churn side. Nothing is ever refused; higher scores buy a more
            expensive contact, and capacity binds long before the threshold.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .config import POLICIES
from .metrics import cost_threshold

BLOCKING, TIERED = "blocking", "tiered"


@dataclass(frozen=True)
class Policy:
    """A loaded, validated decision policy."""

    version: str
    name: str
    kind: str
    costs: dict
    bands: dict                       # action -> lower bound on probability
    timeout_ms: int = 80
    fail_modes: dict = field(default_factory=dict)   # action_type -> open|closed
    rules: dict = field(default_factory=dict)
    capacity: dict = field(default_factory=dict)
    holdout: dict = field(default_factory=dict)
    label: dict = field(default_factory=dict)
    monitoring: dict = field(default_factory=dict)
    path: Path | None = None

    # --------------------------------------------------------- thresholds --
    @property
    def act_threshold(self) -> float:
        """The threshold implied by the costs, not typed into the file.

        Deriving it means a cost change moves the threshold automatically, and
        the number in the config can never drift from the arithmetic.
        On a blocking policy this is the block threshold. On a tiered policy it
        is the point where contacting someone is worth more than it costs,
        which is usually so low that capacity is the real constraint.
        """
        return cost_threshold(self.costs["false_positive"], self.costs["false_negative"])

    @property
    def block_at(self) -> float:
        if self.kind != BLOCKING:
            raise AttributeError(f"policy '{self.name}' is {self.kind}, it never blocks")
        return self.act_threshold

    @property
    def tiers(self) -> list[tuple[str, float]]:
        """Bands from most to least severe."""
        return sorted(self.bands.items(), key=lambda kv: kv[1], reverse=True)

    # ------------------------------------------------------------ decision --
    def decide(self, probability: float | None) -> str:
        """Map a probability to an action. None means the model did not answer."""
        if probability is None:
            raise ValueError("no probability: use fail_action() for the outage path")
        if self.kind == BLOCKING and probability >= self.block_at:
            return "block"
        for action, floor in self.tiers:
            if probability >= floor:
                return action
        return "allow" if self.kind == BLOCKING else "none"

    def fail_action(self, action_type: str) -> tuple[str, str]:
        """What to do when the model does not answer inside the budget.

        Most commerce flows fail open: allow, flag for review, page the on-call.
        Irreversible actions fail closed, because the downside is not symmetric.
        """
        mode = self.fail_modes.get(action_type, self.fail_modes.get("default", "open"))
        if mode == "closed":
            return "challenge", f"fail closed ({action_type}): step-up verification"
        return "allow", f"fail open ({action_type}): flagged for async review"

    @property
    def timeout_seconds(self) -> float:
        return self.timeout_ms / 1000.0

    def summary(self) -> str:
        bands = ", ".join(f"{a} >= {t:.2f}" for a, t in reversed(self.tiers))
        head = (f"block >= {self.block_at:.2f}" if self.kind == BLOCKING
                else f"worth contacting above p = {self.act_threshold:.3f}")
        return (f"{self.name} v{self.version} | {head} "
                f"(FP ${self.costs['false_positive']:,} vs FN ${self.costs['false_negative']:,})"
                f" | {bands}")


def load_policy(name: str = "risk_policy_v1", directory: Path | None = None) -> Policy:
    """Load and validate a policy file."""
    directory = Path(directory or POLICIES)
    path = directory / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"no policy at {path}")

    raw = yaml.safe_load(path.read_text())
    _validate(raw, path)
    return Policy(
        version=str(raw["version"]),
        name=raw["name"],
        kind=raw.get("kind", BLOCKING),
        costs=raw["costs"],
        bands={k: float(v) for k, v in raw["bands"].items()},
        timeout_ms=int(raw.get("timeout_ms", 80)),
        fail_modes=raw.get("fail_modes", {}),
        rules=raw.get("rules", {}),
        capacity=raw.get("capacity", {}),
        holdout=raw.get("holdout", {}),
        label=raw.get("label", {}),
        monitoring=raw.get("monitoring", {}),
        path=path,
    )


def _validate(raw: dict, path: Path) -> None:
    for key in ("version", "name", "costs", "bands"):
        if key not in raw:
            raise ValueError(f"{path}: missing required key '{key}'")
    for key in ("false_positive", "false_negative"):
        if key not in raw["costs"]:
            raise ValueError(f"{path}: costs.{key} is required")

    kind = raw.get("kind", BLOCKING)
    if kind not in (BLOCKING, TIERED):
        raise ValueError(f"{path}: kind must be '{BLOCKING}' or '{TIERED}', got '{kind}'")

    bands = raw["bands"]
    if not bands:
        raise ValueError(f"{path}: at least one band is required")
    for action, floor in bands.items():
        if not 0.0 <= float(floor) <= 1.0:
            raise ValueError(f"{path}: band '{action}' must be a probability, got {floor}")

    # On a blocking policy every band must sit below the derived block
    # threshold, otherwise the band can never fire and the config quietly lies.
    if kind == BLOCKING:
        block_at = cost_threshold(raw["costs"]["false_positive"], raw["costs"]["false_negative"])
        for action, floor in bands.items():
            if float(floor) >= block_at:
                raise ValueError(
                    f"{path}: band '{action}' at {floor} is at or above the derived block "
                    f"threshold {block_at:.3f}, so it can never fire")

    for action_type, mode in raw.get("fail_modes", {}).items():
        if mode not in ("open", "closed"):
            raise ValueError(f"{path}: fail_modes.{action_type} must be 'open' or 'closed'")
