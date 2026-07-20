""

import pytest

from safe_observation.confidence import (
    build_kuhn_confidence_set,
    empirical_bernstein_interval,
    hoeffding_interval,
)
from safe_observation.experiments.online import run_online_adaptation
from safe_observation.opponents import (
    always_fold_opponent,
    equilibrium_opponent,
    static_biased_opponent,
    trap_opponent,
)
from safe_observation.payoff import build_kuhn as build_kuhn_payoff
from safe_observation.sequence_form import compile_kuhn
from safe_observation.solvers import (
    robust_safe_response,
    safety_verifier,
    solve_blueprint,
)

KNOWN_KUHN_VALUE = -1.0 / 18.0


_OPPONENTS = {
    "always_fold": always_fold_opponent,
    "static_biased": lambda: static_biased_opponent(0.1),
    "trap": trap_opponent,
    "equilibrium": equilibrium_opponent,
}


def _intervals_from_opponent(opp, n=500, delta=0.05, method="hoeffding"):
    ""
    interval = (
        hoeffding_interval if method == "hoeffding" else empirical_bernstein_interval
    )
    sf1 = compile_kuhn(1)
    per = delta / (2 * len(sf1.info_sets))
    return {
        info.label: [interval(p, n, per) for p in opp.behavior[info.label]]
        for info in sf1.info_sets
    }


@pytest.mark.parametrize("player", [0, 1])
def test_sequence_constraints_root(player):

    sf = compile_kuhn(player)
    assert sf.sequences[0] == ""
    assert sf.e[0] == 1.0
    x = sf.realization_from_behavior()
    assert x[0] == pytest.approx(1.0)
    assert sf.constraint_residual(x) < 1e-9


@pytest.mark.parametrize("player", [0, 1])
def test_child_sum_equals_parent(player):

    sf = compile_kuhn(player)
    behavior = {info.label: [0.3, 0.7] for info in sf.info_sets}
    x = sf.realization_from_behavior(behavior)
    for info in sf.info_sets:
        child_sum = sum(x[child] for _action, child in info.children)
        assert child_sum == pytest.approx(x[info.parent_seq])


def test_terminal_payoff_matrix_entries():

    a = build_kuhn_payoff()
    sf0, sf1 = compile_kuhn(0), compile_kuhn(1)
    dense = a.dense()
    assert a.nnz == 30

    assert dense[sf0.seq_index["2:>b"]][sf1.seq_index["0:b>b"]] == pytest.approx(2 / 6)
    assert dense[sf0.seq_index["0:>b"]][sf1.seq_index["2:b>b"]] == pytest.approx(-2 / 6)

    assert dense[sf0.seq_index["2:>p"]][sf1.seq_index["0:p>p"]] == pytest.approx(1 / 6)


def test_blueprint_value_matches_known_kuhn_value():
    bp = solve_blueprint("kuhn", method="lp")
    assert bp.value == pytest.approx(KNOWN_KUHN_VALUE, abs=1e-9)
    assert compile_kuhn(0).constraint_residual(bp.realization) < 1e-9

    assert safety_verifier(bp.realization).value == pytest.approx(bp.value, abs=1e-9)


@pytest.mark.parametrize("name", sorted(_OPPONENTS))
def test_robust_lp_returns_safe_strategy(name):

    bp = solve_blueprint("kuhn", method="lp")
    opp = _OPPONENTS[name]()
    intervals = _intervals_from_opponent(opp)
    resp = robust_safe_response(intervals, v_ref=bp.value, eps_safe=0.0)
    assert resp.is_safe()
    assert safety_verifier(resp.realization).value >= bp.value - 1e-8
    assert compile_kuhn(0).constraint_residual(resp.realization) < 1e-9


def test_low_reach_leak_probe_improves_identification():
    ""
    from safe_observation.experiments.online import run_probing_comparison
    from safe_observation.opponents import leduc_low_reach_leak_opponent

    res = run_probing_comparison(
        leduc_low_reach_leak_opponent(0.9),
        rounds=25,
        episodes_per_round=150,
        seeds=(42, 43),
        out_dir=None,
    )
    passive = res["arms"]["passive"]
    probe = res["arms"]["probe_budgeted"]

    passive_ci = passive["target_ci_by_round"][-1]["mean"]
    probe_ci = probe["target_ci_by_round"][-1]["mean"]
    assert probe_ci < passive_ci

    assert probe["exploitation_gain_mean"] > passive["exploitation_gain_mean"] + 1.0
    assert probe["budget_respected"]
    assert probe["budget_floor_respected"]


@pytest.mark.parametrize("player", [0, 1])
def test_behavior_realization_roundtrip(player):

    sf = compile_kuhn(player)
    behavior = {info.label: [0.25, 0.75] for info in sf.info_sets}
    recovered = sf.behavior_from_realization(sf.realization_from_behavior(behavior))
    for info in sf.info_sets:
        assert recovered[info.label] == pytest.approx(behavior[info.label])


@pytest.mark.parametrize("name", sorted(_OPPONENTS))
@pytest.mark.parametrize("method", ["hoeffding", "empirical_bernstein"])
def test_confidence_covers_and_robust_is_safe_for_suite(name, method):

    bp = solve_blueprint("kuhn", method="lp")
    opp = _OPPONENTS[name]()
    y_star = opp.realization()
    assert compile_kuhn(1).constraint_residual(y_star) < 1e-9

    intervals = _intervals_from_opponent(opp, method=method)
    cs = build_kuhn_confidence_set(intervals)
    assert cs.contains(y_star)

    resp = robust_safe_response(intervals, v_ref=bp.value, eps_safe=0.0)
    assert resp.is_safe()


def test_end_to_end_pipeline_audit():
    ""
    sf0, sf1 = compile_kuhn(0), compile_kuhn(1)
    payoff = build_kuhn_payoff()

    assert (sf0.num_sequences, sf0.num_infosets) == (13, 6)
    assert sf0.constraint_residual(sf0.realization_from_behavior()) < 1e-9

    x_u = sf0.realization_from_behavior()
    y_u = sf1.realization_from_behavior()
    ay = payoff.matvec_a_y(y_u)
    assert payoff.bilinear(x_u, y_u) == pytest.approx(
        sum(xi * ai for xi, ai in zip(x_u, ay, strict=True))
    )

    bp = solve_blueprint("kuhn", method="lp")
    assert bp.value == pytest.approx(KNOWN_KUHN_VALUE, abs=1e-9)
    y_eq = equilibrium_opponent().realization()
    value_vs_eq = payoff.bilinear(bp.realization, y_eq)
    assert value_vs_eq >= bp.value - 1e-9
    assert value_vs_eq == pytest.approx(bp.value, abs=1e-2)

    opp = static_biased_opponent(0.1)
    intervals = _intervals_from_opponent(opp, n=1000)
    cs = build_kuhn_confidence_set(intervals)
    assert cs.contains(opp.realization())

    resp = robust_safe_response(intervals, v_ref=bp.value, eps_safe=0.0)
    assert resp.is_safe()
    actual_vs_opp = payoff.bilinear(resp.realization, opp.realization())
    assert actual_vs_opp > bp.value + 1e-3

    assert safety_verifier(resp.realization).value >= bp.value - 1e-8

    results = run_online_adaptation(
        opp, rounds=40, episodes_per_round=200, seed=2026, out_dir=None
    )
    assert results["safety_preserved"]
    assert results["min_safety_value"] >= KNOWN_KUHN_VALUE - 1e-8
    assert results["exploitation_gain"] > 0.0
