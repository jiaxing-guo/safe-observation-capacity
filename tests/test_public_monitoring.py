""

import pytest

from safe_observation.agents import OnlineSafeExploitAgent
from safe_observation.confidence import (
    OpponentEvidenceStore,
    build_public_confidence_set,
    public_key,
)
from safe_observation.experiments.online import run_online_adaptation
from safe_observation.opponents import (
    leduc_private_state_leak_opponent,
    leduc_static_biased_opponent,
    static_biased_opponent,
)

KNOWN_KUHN_VALUE = -1.0 / 18.0
LEDUC_VALUE = -0.08560642


def test_public_key_kuhn_drops_private_card():

    assert public_key("kuhn", "0:pb") == "pb"
    assert public_key("kuhn", "1:pb") == "pb"
    assert public_key("kuhn", "2:") == ""


def test_public_key_leduc_drops_private_rank():

    assert public_key("leduc", "J|Q|cr|c") == "Q|cr|c"
    assert public_key("leduc", "K|Q|cr|c") == "Q|cr|c"


def test_public_key_unknown_game_raises():
    with pytest.raises(ValueError):
        public_key("chess", "anything")


def test_public_groups_partition_all_labels():
    store = OpponentEvidenceStore.for_game("leduc")
    groups = store.public_groups()

    flat = [label for labels in groups.values() for label in labels]
    assert sorted(flat) == sorted(store.labels)
    assert len(flat) == len(set(flat))

    for key, labels in groups.items():
        for label in labels:
            assert public_key("leduc", label) == key


def test_public_intervals_aggregate_counts_over_private_card():
    store = OpponentEvidenceStore.for_game("leduc")

    groups = store.public_groups()
    key, members = next((k, m) for k, m in groups.items() if len(m) >= 2)

    n_actions = store.interval(members[0], 0.1)
    per_action = len(n_actions)
    for label in members:
        store.record(label, [1] + [0] * (per_action - 1))
    pub = store.public_intervals(0.05)

    lo, hi = pub[key][0]
    assert lo <= 1.0 <= hi + 1e-12

    assert hi <= 1.0


def test_public_confidence_set_contains_consistent_opponent():
    store = OpponentEvidenceStore.for_game("leduc")

    groups = store.public_groups()
    intervals = store.public_intervals(0.05)
    cs = build_public_confidence_set("leduc", groups, intervals)
    assert cs.nrows == 0


def test_public_agent_initial_decision_is_safe_and_blueprint_like():
    agent = OnlineSafeExploitAgent(game="leduc", monitoring="public")
    decision = agent.select()
    assert decision.safety_value >= LEDUC_VALUE - 1e-6
    assert decision.robust_value == pytest.approx(LEDUC_VALUE, abs=1e-6)
    assert not decision.repaired


def test_unknown_monitoring_raises():
    with pytest.raises(ValueError):
        OnlineSafeExploitAgent(game="leduc", monitoring="telepathy")


def test_public_monitoring_preserves_safety_kuhn():
    results = run_online_adaptation(
        static_biased_opponent(bet_prob=0.05),
        rounds=40,
        episodes_per_round=200,
        monitoring="public",
        seed=2026,
        out_dir=None,
    )
    assert results["monitoring"] == "public"
    assert results["safety_preserved"]
    assert results["min_safety_value"] >= KNOWN_KUHN_VALUE - 1e-8


def test_public_monitoring_exploits_static_biased_leduc():
    results = run_online_adaptation(
        leduc_static_biased_opponent(),
        rounds=60,
        episodes_per_round=200,
        monitoring="public",
        seed=2026,
        out_dir=None,
    )

    assert results["safety_preserved"]
    assert results["exploitation_gain"] > 1e-2


def test_public_never_out_exploits_full():

    opp = leduc_private_state_leak_opponent("J", 0.9)
    full = run_online_adaptation(
        opp,
        rounds=50,
        episodes_per_round=200,
        monitoring="full",
        seed=2026,
        out_dir=None,
    )
    public = run_online_adaptation(
        opp,
        rounds=50,
        episodes_per_round=200,
        monitoring="public",
        seed=2026,
        out_dir=None,
    )
    assert full["safety_preserved"] and public["safety_preserved"]

    assert public["exploitation_gain"] <= full["exploitation_gain"] + 1e-6
