""

from statistics import mean

import pytest

from safe_observation import native
from safe_observation.agents import OnlineSafeExploitAgent
from safe_observation.opponents import (
    _leduc_cap_facing_labels,
    best_response_value,
    leduc_equilibrium_opponent,
    leduc_low_reach_leak_opponent,
)
from safe_observation.payoff import build as build_payoff
from safe_observation.sequence_form import compile_leduc
from safe_observation.solvers import solve_blueprint

LEDUC_VALUE = -0.08560642


def test_cap_facing_sets_have_fold_call_actions():

    actions = {
        info.label: tuple(a for a, _ in info.children)
        for info in compile_leduc(1).info_sets
    }
    cap = _leduc_cap_facing_labels()
    assert cap, "expected some cap-facing info sets"
    assert all(actions[label] == ("f", "c") for label in cap)


def test_leak_only_perturbs_cap_facing_sets():
    eq = leduc_equilibrium_opponent().behavior
    opp = leduc_low_reach_leak_opponent(0.9)
    cap = set(_leduc_cap_facing_labels())
    increased = 0
    for label, dist in opp.behavior.items():
        if label in cap:
            assert dist[0] >= eq[label][0] - 1e-12
            increased += dist[0] > eq[label][0] + 1e-9
        else:
            assert dist == pytest.approx(eq[label])

    assert increased > 0


def test_low_reach_leak_is_well_formed_and_realizable():
    opp = leduc_low_reach_leak_opponent()
    assert opp.game == "leduc"
    sf1 = compile_leduc(1)
    assert sf1.constraint_residual(opp.realization()) < 1e-9


def test_leak_validates_input():
    with pytest.raises(ValueError):
        leduc_low_reach_leak_opponent(leak=1.5)


def test_leak_is_highly_exploitable_by_best_response():

    expl = best_response_value(leduc_low_reach_leak_opponent(0.9))
    assert expl > LEDUC_VALUE + 1.0


def test_passive_adaptation_fails_on_low_reach_leak():
    opp = leduc_low_reach_leak_opponent(0.9)
    cap = set(_leduc_cap_facing_labels())
    payoff = build_payoff("leduc")
    y_star = list(opp.realization())
    v = solve_blueprint("leduc", method="lp").value
    br_gain = best_response_value(opp) - v

    agent = OnlineSafeExploitAgent(game="leduc", delta=0.05)
    rounds, episodes = 40, 150
    decision = agent.select()
    for t in range(rounds):
        decision = agent.select()
        _, counts = native.simulate(
            "leduc", decision.behavior, opp.behavior, episodes, 5000 + t
        )
        agent.observe(counts)

    passive_gain = payoff.bilinear(decision.realization, y_star) - v

    assert decision.safety_value >= v - 1e-8

    assert br_gain > 1.0
    assert passive_gain < 0.25 * br_gain

    cap_visits = mean(
        agent.evidence.visits(label) for label in agent.evidence.labels if label in cap
    )
    non_visits = mean(
        agent.evidence.visits(label)
        for label in agent.evidence.labels
        if label not in cap
    )
    cap_ci = mean(decision.ci_width_by_infoset[label] for label in cap)
    non_ci = mean(
        decision.ci_width_by_infoset[label]
        for label in decision.ci_width_by_infoset
        if label not in cap
    )
    assert non_visits > 3 * cap_visits
    assert cap_ci > non_ci + 0.15
