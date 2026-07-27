"""Regression tests for test robust. See the corresponding implementation module and supplementary Reproducibility."""

import pytest

from safe_observation.confidence import OpponentEvidenceStore
from safe_observation.sequence_form import compile_kuhn
from safe_observation.solvers import robust_safe_response, safety_verifier

KNOWN_KUHN_VALUE = -1.0 / 18.0


def _always_fold_intervals():
    """Compute the always fold intervals."""
    sf = compile_kuhn(1)

    return {info.label: [(1.0, 1.0), (0.0, 0.0)] for info in sf.info_sets}


def test_empty_confidence_set_recovers_game_value():
    """Verify that empty confidence set recovers game value."""
    resp = robust_safe_response({}, v_ref=KNOWN_KUHN_VALUE, eps_safe=0.0)
    assert resp.robust_value == pytest.approx(KNOWN_KUHN_VALUE, abs=1e-9)
    sf0 = compile_kuhn(0)
    assert sf0.constraint_residual(resp.realization) < 1e-9
    assert resp.is_safe()


def test_robust_response_is_always_safe():
    """Verify that robust response is always safe."""
    resp = robust_safe_response(_always_fold_intervals(), v_ref=KNOWN_KUHN_VALUE)
    assert resp.is_safe()
    assert resp.safety_value() >= KNOWN_KUHN_VALUE - 1e-9


def test_exploits_always_fold_when_safety_relaxed():
    """Verify that exploits always fold when safety relaxed."""
    resp = robust_safe_response(
        _always_fold_intervals(), v_ref=KNOWN_KUHN_VALUE, eps_safe=10.0
    )
    assert resp.robust_value == pytest.approx(1.0, abs=1e-9)
    sf0 = compile_kuhn(0)
    assert sf0.constraint_residual(resp.realization) < 1e-9


def test_safe_exploit_beats_equilibrium_but_not_unconstrained():
    """Verify that safe exploit beats equilibrium but not unconstrained."""
    safe = robust_safe_response(_always_fold_intervals(), v_ref=KNOWN_KUHN_VALUE)
    loose = robust_safe_response(
        _always_fold_intervals(), v_ref=KNOWN_KUHN_VALUE, eps_safe=10.0
    )
    assert safe.robust_value > KNOWN_KUHN_VALUE + 1e-6
    assert safe.robust_value <= loose.robust_value + 1e-9


def test_v_ref_defaults_to_blueprint_value():
    """Verify that v ref defaults to blueprint value."""
    resp = robust_safe_response(_always_fold_intervals())
    assert resp.v_ref == pytest.approx(KNOWN_KUHN_VALUE, abs=1e-9)
    assert resp.is_safe()


def test_pipeline_from_evidence_is_safe():
    """Verify that pipeline from evidence is safe."""
    sf = compile_kuhn(1)
    store = OpponentEvidenceStore.for_kuhn()
    for info in sf.info_sets:
        store.record(info.label, [90, 10])
    intervals = store.intervals(0.1, method="hoeffding")
    resp = robust_safe_response(intervals, v_ref=KNOWN_KUHN_VALUE)
    assert resp.is_safe()

    assert safety_verifier(resp.realization).value >= KNOWN_KUHN_VALUE - 1e-9


def test_unknown_game_raises():
    """Verify that unknown game raises."""
    with pytest.raises(ValueError):
        robust_safe_response({}, v_ref=0.0, game="chess")
