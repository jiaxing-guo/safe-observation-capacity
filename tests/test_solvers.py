""

import pytest

from safe_observation.sequence_form import compile_kuhn
from safe_observation.solvers import best_response, safety_verifier, solve_blueprint

KNOWN_KUHN_VALUE = -1.0 / 18.0


def test_blueprint_lp_matches_known_value():
    sol = solve_blueprint("kuhn", method="lp")
    assert sol.method == "lp"
    assert sol.value == pytest.approx(KNOWN_KUHN_VALUE, abs=1e-9)
    assert sol.realization is not None
    assert len(sol.realization) == 13

    sf0 = compile_kuhn(0)
    assert set(sol.strategy) == {info.label for info in sf0.info_sets}
    for probs in sol.strategy.values():
        assert sum(probs) == pytest.approx(1.0)


def test_blueprint_realization_is_feasible():
    sol = solve_blueprint("kuhn", method="lp")
    sf0 = compile_kuhn(0)
    assert sf0.constraint_residual(sol.realization) < 1e-9


def test_blueprint_is_exactly_safe():

    sol = solve_blueprint("kuhn", method="lp")
    safety = safety_verifier(sol.realization, game="kuhn")
    assert safety.value == pytest.approx(sol.value, abs=1e-9)
    assert safety.is_safe(sol.value)


def test_always_pass_is_exploited_to_minus_one():
    sf0 = compile_kuhn(0)
    x = sf0.realization_from_behavior(
        {info.label: [1.0, 0.0] for info in sf0.info_sets}
    )
    safety = safety_verifier(x, game="kuhn")
    assert safety.value == pytest.approx(-1.0, abs=1e-9)
    assert not safety.is_safe(KNOWN_KUHN_VALUE)
    sf1 = compile_kuhn(1)
    assert sf1.constraint_residual(safety.best_response) < 1e-9


def test_cfr_method_still_available():
    sol = solve_blueprint("kuhn", method="cfr", iterations=20_000)
    assert sol.method == "cfr"
    assert sol.realization is None
    assert sol.value == pytest.approx(KNOWN_KUHN_VALUE, abs=5e-3)


def test_mccfr_blueprint_backend_converges_on_kuhn():

    sol = solve_blueprint("kuhn", method="mccfr", iterations=20_000, seed=2026)
    assert sol.method == "mccfr"
    assert sol.realization is not None

    assert sol.value <= KNOWN_KUHN_VALUE + 1e-9
    assert sol.value == pytest.approx(KNOWN_KUHN_VALUE, abs=2e-2)

    assert compile_kuhn(0).constraint_residual(sol.realization) < 1e-9


def test_mccfr_backend_is_reproducible():
    a = solve_blueprint("kuhn", method="mccfr", iterations=2_000, seed=7)
    b = solve_blueprint("kuhn", method="mccfr", iterations=2_000, seed=7)
    assert a.value == b.value
    assert a.realization == b.realization


def test_unknown_method_raises():
    with pytest.raises(ValueError):
        solve_blueprint("kuhn", method="nope")


def test_safety_verifier_validates_length():
    with pytest.raises(ValueError):
        safety_verifier([0.0] * 5, game="kuhn")


def test_best_response_to_always_fold_is_plus_one():
    sf1 = compile_kuhn(1)
    y = sf1.realization_from_behavior(
        {info.label: [1.0, 0.0] for info in sf1.info_sets}
    )
    br = best_response(y, game="kuhn")
    assert br.value == pytest.approx(1.0, abs=1e-9)
    assert compile_kuhn(0).constraint_residual(br.realization) < 1e-9


def test_best_response_lower_bounds_game_value():

    sf1 = compile_kuhn(1)
    for probs in ([0.5, 0.5], [0.2, 0.8], [0.9, 0.1]):
        y = sf1.realization_from_behavior({info.label: probs for info in sf1.info_sets})
        assert best_response(y, game="kuhn").value >= KNOWN_KUHN_VALUE - 1e-9


def test_best_response_validates_length():
    with pytest.raises(ValueError):
        best_response([0.0] * 5, game="kuhn")
