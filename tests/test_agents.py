"""Regression tests for test agents. See the corresponding implementation module and supplementary Reproducibility."""

import pytest

from safe_observation.agents import OnlineSafeExploitAgent
from safe_observation.experiments.online import run_online_adaptation
from safe_observation.opponents import (
    always_fold_opponent,
    equilibrium_opponent,
    static_biased_opponent,
    trap_opponent,
)
from safe_observation.sequence_form import compile_kuhn
from safe_observation.solvers import (
    fallback_mixture_repair,
    safety_verifier,
    solve_blueprint,
)

KNOWN_KUHN_VALUE = -1.0 / 18.0


def test_fallback_mixture_repairs_unsafe_candidate():
    """Verify that fallback mixture repairs unsafe candidate."""
    blueprint = solve_blueprint("kuhn", method="lp")
    x_blue = blueprint.realization

    sf0 = compile_kuhn(0)
    x_bad = sf0.realization_from_behavior(
        {info.label: [1.0, 0.0] for info in sf0.info_sets}
    )
    assert safety_verifier(x_bad).value < KNOWN_KUHN_VALUE - 1e-6

    repaired = fallback_mixture_repair(x_blue, x_bad, v_ref=KNOWN_KUHN_VALUE)
    assert safety_verifier(repaired).value >= KNOWN_KUHN_VALUE - 1e-8
    assert sf0.constraint_residual(repaired) < 1e-9


def test_fallback_keeps_safe_candidate_unchanged():
    """Verify that fallback keeps safe candidate unchanged."""
    blueprint = solve_blueprint("kuhn", method="lp")
    x_blue = blueprint.realization

    repaired = fallback_mixture_repair(x_blue, x_blue, v_ref=KNOWN_KUHN_VALUE)
    assert repaired == pytest.approx(x_blue)


def test_agent_initial_decision_is_safe_and_blueprint_like():
    """Verify that agent initial decision is safe and blueprint like."""
    agent = OnlineSafeExploitAgent(delta=0.05, eps_safe=0.0)
    decision = agent.select()
    assert decision.safety_value >= KNOWN_KUHN_VALUE - 1e-8
    assert decision.robust_value == pytest.approx(KNOWN_KUHN_VALUE, abs=1e-7)
    assert not decision.repaired


def test_static_biased_opponent_exploited():
    """Verify that static biased opponent exploited."""
    results = run_online_adaptation(
        static_biased_opponent(bet_prob=0.05),
        rounds=60,
        episodes_per_round=200,
        seed=2026,
        out_dir=None,
    )

    assert results["safety_preserved"]
    assert results["exploitation_gain"] > 1e-3


def test_equilibrium_opponent_no_safety_violation():
    """Verify that equilibrium opponent no safety violation."""
    results = run_online_adaptation(
        equilibrium_opponent(),
        rounds=40,
        episodes_per_round=100,
        seed=2026,
        out_dir=None,
    )
    assert results["safety_preserved"]

    assert results["final_actual_value"] >= KNOWN_KUHN_VALUE - 1e-6
    assert results["final_actual_value"] <= KNOWN_KUHN_VALUE + 0.05


def test_trap_opponent_safety_preserved():
    """Verify that trap opponent safety preserved."""
    results = run_online_adaptation(
        trap_opponent(),
        rounds=60,
        episodes_per_round=200,
        seed=2026,
        out_dir=None,
    )

    assert results["safety_preserved"]
    assert results["min_safety_value"] >= KNOWN_KUHN_VALUE - 1e-8


def test_always_fold_is_strongly_exploited():
    """Verify that always fold is strongly exploited."""
    results = run_online_adaptation(
        always_fold_opponent(),
        rounds=60,
        episodes_per_round=200,
        seed=2026,
        out_dir=None,
    )
    assert results["safety_preserved"]

    assert results["exploitation_gain"] > 0.05


def test_results_payload_is_serializable_shape():
    """Verify that results payload is serializable shape."""
    results = run_online_adaptation(
        static_biased_opponent(),
        rounds=5,
        episodes_per_round=10,
        seed=2026,
        out_dir=None,
    )
    assert len(results["rounds_log"]) == 5
    assert {
        "round",
        "actual_value",
        "robust_value",
        "safety_value",
        "avg_payoff",
    } <= set(results["rounds_log"][0])
