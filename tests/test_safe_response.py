"""Regression tests for test safe response. See the corresponding implementation module and supplementary Reproducibility."""

import pytest

from safe_observation import native
from safe_observation.opponents import (
    leduc_equilibrium_opponent,
    leduc_low_reach_leak_opponent,
    leduc_static_biased_opponent,
)
from safe_observation.payoff import build as build_payoff
from safe_observation.solvers import (
    best_response,
    safety_constrained_best_response,
    safety_verifier,
)

LEDUC_VALUE = native.blueprint_lp("leduc")[0]


def _scbr(opp, eps_safe=0.0):
    """Compute scbr for the test safe response workflow."""
    return safety_constrained_best_response(
        opp.behavior, v_ref=LEDUC_VALUE, eps_safe=eps_safe, game="leduc"
    )


def test_scbr_is_safe_and_achieves_its_value():
    """Verify that scbr is safe and achieves its value."""
    opp = leduc_static_biased_opponent()
    scbr = _scbr(opp)
    payoff = build_payoff("leduc")
    y_star = list(opp.realization())
    assert payoff.bilinear(scbr.realization, y_star) == pytest.approx(
        scbr.value, abs=1e-6
    )
    assert safety_verifier(scbr.realization, game="leduc").value >= LEDUC_VALUE - 1e-8


def test_scbr_is_bracketed_by_game_value_and_best_response():
    """Verify that scbr is bracketed by game value and best response."""
    for opp in (
        leduc_equilibrium_opponent(),
        leduc_static_biased_opponent(),
        leduc_low_reach_leak_opponent(0.9),
    ):
        y_star = list(opp.realization())
        br = best_response(y_star, game="leduc").value
        scbr = _scbr(opp).value
        assert LEDUC_VALUE - 1e-9 <= scbr <= br + 1e-9


def test_scbr_equals_game_value_for_equilibrium():
    """Verify that scbr equals game value for equilibrium."""
    scbr = _scbr(leduc_equilibrium_opponent())
    assert scbr.value == pytest.approx(LEDUC_VALUE, abs=1e-6)


def test_scbr_relaxes_with_eps_safe():
    """Verify that scbr relaxes with eps safe."""
    opp = leduc_low_reach_leak_opponent(0.9)
    tight = _scbr(opp, eps_safe=0.0).value
    loose = _scbr(opp, eps_safe=0.5).value
    assert loose >= tight - 1e-9


def test_scbr_ceiling_is_low_for_low_reach_leak():
    """Verify that scbr ceiling is low for low reach leak."""
    opp = leduc_low_reach_leak_opponent(0.9)
    y_star = list(opp.realization())
    br_gain = best_response(y_star, game="leduc").value - LEDUC_VALUE
    scbr_gain = _scbr(opp).value - LEDUC_VALUE
    assert br_gain > 2.0
    assert scbr_gain < 0.25
