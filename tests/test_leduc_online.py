"""Regression tests for test Leduc online. See the corresponding implementation module and supplementary Reproducibility."""

import pytest

from safe_observation.agents import OnlineSafeExploitAgent
from safe_observation.experiments.online import run_online_adaptation
from safe_observation.opponents import (
    leduc_near_equilibrium_opponent,
    leduc_static_biased_opponent,
)

LEDUC_VALUE = -0.0856064240780


def test_leduc_agent_initial_decision_is_safe_and_blueprint_like():
    """Verify that Leduc agent initial decision is safe and blueprint like."""
    agent = OnlineSafeExploitAgent(game="leduc", delta=0.05, eps_safe=0.0)
    decision = agent.select()
    assert decision.safety_value >= LEDUC_VALUE - 1e-6
    assert decision.robust_value == pytest.approx(LEDUC_VALUE, abs=1e-6)
    assert not decision.repaired
    assert len(decision.realization) == 337


def test_leduc_static_biased_is_exploited_and_safe():
    """Verify that Leduc static biased is exploited and safe."""
    results = run_online_adaptation(
        leduc_static_biased_opponent(),
        rounds=25,
        episodes_per_round=200,
        seed=2026,
        out_dir=None,
    )
    assert results["game"] == "leduc"
    assert results["game_value"] == pytest.approx(LEDUC_VALUE, abs=1e-6)
    assert results["safety_preserved"]
    assert results["min_safety_value"] >= LEDUC_VALUE - 1e-6
    assert results["exploitation_gain"] > 1e-2


def test_leduc_near_equilibrium_stays_safe():
    """Verify that Leduc near equilibrium stays safe."""
    results = run_online_adaptation(
        leduc_near_equilibrium_opponent(eps=0.1),
        rounds=20,
        episodes_per_round=200,
        seed=2026,
        out_dir=None,
    )
    assert results["safety_preserved"]

    assert results["exploitation_gain"] < 0.5


def test_leduc_rounds_log_shape():
    """Verify that Leduc rounds log shape."""
    results = run_online_adaptation(
        leduc_static_biased_opponent(),
        rounds=4,
        episodes_per_round=50,
        seed=2026,
        out_dir=None,
    )
    row = results["rounds_log"][0]
    assert {
        "round",
        "actual_value",
        "robust_value",
        "safety_value",
        "empirical_br_value",
        "mean_ci_width",
        "avg_payoff",
    } <= set(row)

    assert len(row["ci_width_by_infoset"]) == 144
