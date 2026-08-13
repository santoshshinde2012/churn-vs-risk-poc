"""The decision layer: thresholds, bands, fail modes and the rules beside them."""

from __future__ import annotations

import textwrap

import pytest
import yaml

from churnrisk.metrics import cost_threshold
from churnrisk.policy import BLOCKING, TIERED, load_policy
from churnrisk.rules import RuleEngine, build_blocklist, escalate


# ------------------------------------------------------------- policies ----
def test_shipped_policies_load_and_validate():
    for name in ("risk_policy_v1", "risk_policy_v2", "churn_policy_v1"):
        policy = load_policy(name)
        assert policy.version
        assert policy.kind in (BLOCKING, TIERED)


def test_block_threshold_is_derived_from_costs_not_typed_in():
    """The article's arithmetic, and the reason the number is never in the file."""
    policy = load_policy("risk_policy_v1")
    assert policy.block_at == pytest.approx(500 / (500 + 150))
    assert policy.block_at == pytest.approx(0.769, abs=0.001)


def test_changing_costs_moves_the_threshold_on_its_own():
    v1, v2 = load_policy("risk_policy_v1"), load_policy("risk_policy_v2")
    assert v2.block_at < v1.block_at
    assert v2.block_at == pytest.approx(320 / (320 + 180))


def test_decisions_are_ordered_by_severity():
    policy = load_policy("risk_policy_v1")
    assert policy.decide(0.99) == "block"
    assert policy.decide(policy.block_at) == "block"
    assert policy.decide(policy.block_at - 1e-9) == "review"
    assert policy.decide(0.45) == "challenge"
    assert policy.decide(0.0) == "allow"


def test_a_tiered_policy_never_blocks():
    policy = load_policy("churn_policy_v1")
    assert policy.kind == TIERED
    assert policy.decide(1.0) == "call"
    assert policy.decide(0.0) == "none"
    with pytest.raises(AttributeError, match="never blocks"):
        _ = policy.block_at


def test_irreversible_actions_fail_closed():
    policy = load_policy("risk_policy_v1")
    assert policy.fail_action("checkout")[0] == "allow"
    assert policy.fail_action("login")[0] == "allow"
    assert policy.fail_action("withdrawal")[0] == "challenge"
    assert policy.fail_action("password_change")[0] == "challenge"
    assert policy.fail_action("something_new")[0] == "allow"      # the default


def test_no_probability_means_use_the_fail_path():
    policy = load_policy("risk_policy_v1")
    with pytest.raises(ValueError, match="fail_action"):
        policy.decide(None)


# -------------------------------------------------------- policy guards ----
def _write(tmp_path, body):
    path = tmp_path / "custom.yaml"
    path.write_text(textwrap.dedent(body))
    return path


VALID = """
    version: "9.9.9"
    name: test
    kind: blocking
    costs: {false_positive: 500, false_negative: 150}
    bands: {challenge: 0.3}
"""


def test_a_band_above_the_block_threshold_is_rejected(tmp_path):
    """A band that can never fire is a config that quietly lies."""
    _write(tmp_path, VALID.replace("challenge: 0.3", "challenge: 0.95"))
    with pytest.raises(ValueError, match="can never fire"):
        load_policy("custom", directory=tmp_path)


def test_out_of_range_bands_are_rejected(tmp_path):
    _write(tmp_path, VALID.replace("challenge: 0.3", "challenge: 1.7"))
    with pytest.raises(ValueError, match="must be a probability"):
        load_policy("custom", directory=tmp_path)


def test_unknown_fail_mode_is_rejected(tmp_path):
    _write(tmp_path, VALID + "    fail_modes: {checkout: maybe}\n")
    with pytest.raises(ValueError, match="'open' or 'closed'"):
        load_policy("custom", directory=tmp_path)


def test_missing_costs_are_rejected(tmp_path):
    body = yaml.safe_load(VALID)
    del body["costs"]["false_negative"]
    (tmp_path / "custom.yaml").write_text(yaml.safe_dump(body))
    with pytest.raises(ValueError, match="false_negative"):
        load_policy("custom", directory=tmp_path)


def test_cost_threshold_rejects_nonsense():
    with pytest.raises(ValueError):
        cost_threshold(0, 0)
    with pytest.raises(ValueError):
        cost_threshold(-1, 10)


# ----------------------------------------------------------------- rules ---
def test_rules_only_escalate():
    assert escalate("allow", "block") == "block"
    assert escalate("block", "allow") == "block"
    assert escalate("review", "challenge") == "review"


def test_velocity_rule_fires_above_the_limit():
    engine = RuleEngine(load_policy("risk_policy_v1"))
    quiet = {"txn_60s": 1, "txn_10min": 2, "account_age_days": 900,
             "amount": 20, "device_accounts": 1}
    fast = {**quiet, "txn_60s": 12}
    assert engine.evaluate(quiet) == []
    assert [hit.rule for hit in engine.evaluate(fast)] == ["velocity"]
    assert engine.apply("allow", fast)[0] == "review"


def test_new_account_rule_applies_a_ceiling():
    engine = RuleEngine(load_policy("risk_policy_v1"))
    record = {"txn_60s": 0, "txn_10min": 0, "account_age_days": 3,
              "amount": 900, "device_accounts": 1}
    assert [hit.rule for hit in engine.evaluate(record)] == ["new_account"]
    # The same amount on an aged account is unremarkable.
    assert engine.evaluate({**record, "account_age_days": 500}) == []


def test_network_rule_sees_the_ring():
    engine = RuleEngine(load_policy("risk_policy_v1"))
    record = {"txn_60s": 0, "txn_10min": 0, "account_age_days": 900,
              "amount": 40, "device_accounts": 9}
    assert [hit.rule for hit in engine.evaluate(record)] == ["network"]


def test_rules_never_downgrade_the_model():
    """A misfiring rule can cost friction. It can never wave a high score through."""
    engine = RuleEngine(load_policy("risk_policy_v1"))
    calm = {"txn_60s": 0, "txn_10min": 0, "account_age_days": 900,
            "amount": 10, "device_accounts": 1}
    assert engine.apply("block", calm)[0] == "block"


def test_blocklist_spares_devices_with_a_legitimate_history():
    """The session-hijack case: fraud on a device its real owner also uses.

    Blocklisting that device is the same mistake as blocklisting the victim's
    card, and it is the difference between a 0.01% insult rate and a 1% one.
    """
    attacker = [{"device": "throwaway"}] * 4
    victim = [{"device": "laptop"}] * 40
    records = attacker + victim
    labels = [1] * 4 + [1, 1] + [0] * 38          # two frauds on a busy real device

    blocked = build_blocklist(records, labels)
    assert "throwaway" in blocked
    assert "laptop" not in blocked


def test_blocklist_needs_more_than_one_confirmed_fraud():
    blocked = build_blocklist([{"device": "d"}], [1])
    assert blocked == set()
