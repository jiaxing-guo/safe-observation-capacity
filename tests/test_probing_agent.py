""

import pytest

from safe_observation import native
from safe_observation.agents import OnlineSafeExploitAgent
from safe_observation.experiments.online import (
    run_importance_comparison,
    run_probing_comparison,
)
from safe_observation.opponents import (
    _leduc_cap_facing_labels,
    leduc_low_reach_leak_opponent,
)
from safe_observation.payoff import build as build_payoff
from safe_observation.probe import ProbeBudget

LEDUC_VALUE = native.blueprint_lp("leduc")[0]


def test_probing_requires_full_monitoring():
    with pytest.raises(ValueError):
        OnlineSafeExploitAgent(game="leduc", probing=True, monitoring="public")


def test_unknown_importance_mode_raises():
    with pytest.raises(ValueError):
        OnlineSafeExploitAgent(game="leduc", probing=True, importance_mode="psychic")


def test_probing_decision_reports_probe_fields():
    agent = OnlineSafeExploitAgent(
        game="leduc", probing=True, beta=1.0, probe_budget=ProbeBudget(total=0.0)
    )
    decision = agent.select()

    assert decision.rho_granted == 0.0
    assert decision.rho_spent == pytest.approx(0.0, abs=1e-8)
    assert decision.info_gain >= 0.0


def _run_agent(opp, probing, beta, total, per_round, rounds=30, ep=150):
    budget = ProbeBudget(total=total, per_round=per_round)
    agent = OnlineSafeExploitAgent(
        game="leduc", probing=probing, beta=beta, probe_budget=budget
    )
    payoff = build_payoff("leduc")
    y_star = list(opp.realization())
    decision = agent.select()
    min_safety = float("inf")
    for t in range(rounds):
        decision = agent.select()
        min_safety = min(min_safety, decision.safety_value)
        _, counts = native.simulate(
            "leduc", decision.behavior, opp.behavior, ep, 5000 + t
        )
        agent.observe(counts)
    gain = payoff.bilinear(decision.realization, y_star) - LEDUC_VALUE
    return gain, min_safety, agent.probe_budget.spent, decision


def test_hard_safe_probing_preserves_theorem1():

    opp = leduc_low_reach_leak_opponent(0.9)
    _gain, min_safety, spent, _ = _run_agent(opp, True, 2.0, 0.0, 0.0)
    assert min_safety >= LEDUC_VALUE - 1e-8
    assert spent == pytest.approx(0.0, abs=1e-8)


def test_budgeted_probing_cracks_the_leak():

    opp = leduc_low_reach_leak_opponent(0.9)
    passive_gain, _, _, _ = _run_agent(opp, False, 0.0, 0.0, 0.0)
    budget_gain, min_safety, spent, _ = _run_agent(opp, True, 2.0, 1e9, 0.5)
    assert budget_gain > passive_gain + 1.0

    assert min_safety >= LEDUC_VALUE - 0.5 - 1e-8
    assert spent > 0.0


def test_budget_exhaustion_reverts_to_hard_safe():

    opp = leduc_low_reach_leak_opponent(0.9)
    budget = ProbeBudget(total=1.0, per_round=0.5)
    agent = OnlineSafeExploitAgent(
        game="leduc", probing=True, beta=2.0, probe_budget=budget
    )
    for t in range(20):
        decision = agent.select()
        _, counts = native.simulate(
            "leduc", decision.behavior, opp.behavior, 100, 9000 + t
        )
        agent.observe(counts)

    assert agent.probe_budget.spent <= 1.0 + 1e-8
    assert agent.probe_budget.allowance() == pytest.approx(0.0, abs=1e-8)


def test_probing_comparison_arms_and_metrics():
    opp = leduc_low_reach_leak_opponent(0.9)
    res = run_probing_comparison(
        opp, rounds=25, episodes_per_round=150, seeds=(42, 43), out_dir=None
    )
    assert res["n_target_labels"] == len(_leduc_cap_facing_labels())
    arms = res["arms"]
    assert set(arms) == {
        "passive",
        "probe_hard_safe",
        "probe_budget_only",
        "probe_budgeted",
    }

    assert arms["passive"]["budget_spent_mean"] == pytest.approx(0.0, abs=1e-8)
    assert arms["probe_hard_safe"]["budget_spent_mean"] == pytest.approx(0.0, abs=1e-8)
    assert arms["probe_budget_only"]["budget_spent_mean"] > 0.0
    assert arms["probe_budgeted"]["budget_spent_mean"] > 0.0

    for arm_name in ("probe_budget_only", "probe_budgeted"):
        assert (
            arms[arm_name]["exploitation_gain_mean"]
            > arms["passive"]["exploitation_gain_mean"] + 1.0
        )

    for arm in arms.values():
        assert arm["budget_respected"]
        assert arm["budget_floor_respected"]


def test_information_bonus_adds_identification_not_exploitation():

    opp = leduc_low_reach_leak_opponent(0.9)
    res = run_probing_comparison(
        opp, rounds=30, episodes_per_round=200, seeds=(42, 43, 44), out_dir=None
    )
    budget_only = res["arms"]["probe_budget_only"]
    budgeted = res["arms"]["probe_budgeted"]

    assert (
        budgeted["target_ci_by_round"][-1]["mean"]
        < budget_only["target_ci_by_round"][-1]["mean"] - 1e-3
    )

    assert (
        budgeted["exploitation_gain_mean"] < budget_only["exploitation_gain_mean"] + 0.5
    )


def test_probing_comparison_shrinks_target_intervals():

    opp = leduc_low_reach_leak_opponent(0.9)
    res = run_probing_comparison(
        opp, rounds=25, episodes_per_round=150, seeds=(42, 43), out_dir=None
    )
    passive_ci = res["arms"]["passive"]["target_ci_by_round"][-1]["mean"]
    probe_ci = res["arms"]["probe_budgeted"]["target_ci_by_round"][-1]["mean"]
    assert probe_ci < passive_ci


def test_sensitivity_importance_outperforms_uniform_at_small_budget():

    opp = leduc_low_reach_leak_opponent(0.9)
    res = run_importance_comparison(
        opp,
        budgets=(0.1,),
        rounds=30,
        episodes_per_round=200,
        seeds=(42, 43, 44),
        out_dir=None,
    )
    cell = res["cells"][0]
    assert cell["sensitivity"]["exploitation_gain_mean"] > (
        cell["uniform"]["exploitation_gain_mean"] + 0.05
    )

    assert cell["uniform"]["budget_floor_respected"]
    assert cell["sensitivity"]["budget_floor_respected"]
