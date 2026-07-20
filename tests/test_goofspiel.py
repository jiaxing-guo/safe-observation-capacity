""

import math

from safe_observation import native
from safe_observation.experiments.online import run_online_adaptation
from safe_observation.opponents import (
    goofspiel_equilibrium_opponent,
    goofspiel_highball_opponent,
    goofspiel_lowball_opponent,
    goofspiel_opponent_suite,
    goofspiel_uniform_opponent,
)
from safe_observation.solvers import best_response, safety_verifier, solve_blueprint

GOOFSPIEL_SEQUENCES = 58
GOOFSPIEL_INFOSETS = 46


def test_goofspiel_sequence_form_sizes_are_symmetric():
    n_seq1, n_info1, n_seq2, n_info2 = native.sequence_form_sizes("goofspiel")
    assert (n_seq1, n_info1) == (GOOFSPIEL_SEQUENCES, GOOFSPIEL_INFOSETS)
    assert (n_seq2, n_info2) == (GOOFSPIEL_SEQUENCES, GOOFSPIEL_INFOSETS)


def test_goofspiel_value_is_zero_by_symmetry():
    value, realization = native.blueprint_lp("goofspiel")

    assert math.isclose(value, 0.0, abs_tol=1e-6)
    assert len(realization) == GOOFSPIEL_SEQUENCES


def test_goofspiel_blueprint_is_safe_at_the_floor():
    bp = solve_blueprint("goofspiel", method="lp")
    assert math.isclose(bp.value, 0.0, abs_tol=1e-6)

    sv = safety_verifier(bp.realization, game="goofspiel")
    assert sv.value >= -1e-6


def test_goofspiel_lowball_is_exploitable_highball_is_not():

    low = best_response(goofspiel_lowball_opponent().realization(), game="goofspiel")
    high = best_response(goofspiel_highball_opponent().realization(), game="goofspiel")
    assert low.value > 0.5
    assert high.value <= 1e-6

    eq = best_response(goofspiel_equilibrium_opponent().realization(), game="goofspiel")
    assert eq.value <= 1e-6


def test_goofspiel_online_exploits_lowball_safely():
    res = run_online_adaptation(
        goofspiel_lowball_opponent(),
        rounds=20,
        episodes_per_round=200,
        seed=2026,
        out_dir=None,
    )
    assert res["game"] == "goofspiel"
    assert math.isclose(res["game_value"], 0.0, abs_tol=1e-6)

    assert res["safety_preserved"]
    assert res["min_safety_value"] >= -1e-6
    assert res["exploitation_gain"] > 0.5


def test_goofspiel_online_against_equilibrium_stays_safe():
    res = run_online_adaptation(
        goofspiel_equilibrium_opponent(),
        rounds=20,
        episodes_per_round=200,
        seed=2026,
        out_dir=None,
    )
    assert res["safety_preserved"]
    assert res["min_safety_value"] >= -1e-6

    assert res["exploitation_gain"] < 0.25


def test_goofspiel_suite_is_well_formed():
    suite = goofspiel_opponent_suite()
    assert set(suite) == {"equilibrium", "lowball", "highball", "uniform"}
    for opp in suite.values():
        assert opp.game == "goofspiel"

        from safe_observation.sequence_form import compile_goofspiel

        sf1 = compile_goofspiel(1)
        x = sf1.realization_from_behavior(opp.behavior)
        assert sf1.constraint_residual(x) < 1e-9


def test_goofspiel_uniform_is_mildly_exploitable():

    uni = best_response(goofspiel_uniform_opponent().realization(), game="goofspiel")
    low = best_response(goofspiel_lowball_opponent().realization(), game="goofspiel")
    assert 0.0 < uni.value < low.value
