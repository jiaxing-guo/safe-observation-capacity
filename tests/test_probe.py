"""Regression tests for test probe. See the corresponding implementation module and supplementary Reproducibility."""

import pytest

from safe_observation import native
from safe_observation.confidence import OpponentEvidenceStore
from safe_observation.opponents import (
    _leduc_cap_facing_labels,
    leduc_low_reach_leak_opponent,
)
from safe_observation.probe import (
    ProbeBudget,
    ProbePlanner,
    SafetyBudgetLedger,
    information_gain,
    weights_from_intervals,
)
from safe_observation.sequence_form import compile_kuhn, compile_leduc
from safe_observation.solvers import (
    confidence_sensitivity,
    probe_coefficients,
    robust_safe_response,
    robust_safe_response_probe,
)

KUHN_VALUE = -1.0 / 18.0


LEDUC_VALUE = native.blueprint_lp("leduc")[0]


def test_weights_are_summed_interval_width():
    """Verify that weights are summed interval width."""
    intervals = {"a": [(0.1, 0.4), (0.0, 0.5)], "b": [(0.2, 0.2)]}
    w = weights_from_intervals(intervals)
    assert w["a"] == pytest.approx(0.3 + 0.5)
    assert w["b"] == pytest.approx(0.0)


def test_weights_apply_importance():
    """Verify that weights apply importance."""
    intervals = {"a": [(0.0, 0.4), (0.0, 0.2)]}
    importance = {"a": [2.0, 0.5]}
    w = weights_from_intervals(intervals, importance)
    assert w["a"] == pytest.approx(0.4 * 2.0 + 0.2 * 0.5)


def test_probe_coefficients_length_matches_player1_sequences():
    """Verify that probe coefficients length matches player1 sequences."""
    n_seq = len(compile_kuhn(0).sequences)
    coeffs = probe_coefficients(
        {}, {label: 1.0 for label in _kuhn_p2_labels()}, game="kuhn"
    )
    assert len(coeffs) == n_seq


def test_probe_coefficients_zero_weights_are_zero():
    """Verify that probe coefficients zero weights are zero."""
    coeffs = probe_coefficients({}, {}, game="kuhn")
    assert all(c == 0.0 for c in coeffs)


def test_information_gain_is_the_dot_product():
    """Verify that information gain is the dot product."""
    coeffs = (1.0, 2.0, 3.0)
    x = (0.5, 0.0, 2.0)
    assert information_gain(coeffs, x) == pytest.approx(0.5 + 0.0 + 6.0)


def _kuhn_p2_labels():
    """Compute Kuhn player-two labels for the test probe workflow."""
    return [info.label for info in compile_kuhn(1).info_sets]


def test_sensitivity_is_nonnegative_per_action():
    """Verify that sensitivity is nonnegative per action."""
    labels = _kuhn_p2_labels()
    intervals = {label: [(0.3, 0.7), (0.3, 0.7)] for label in labels}
    imp = confidence_sensitivity(intervals, v_ref=KUHN_VALUE, eps_safe=0.0, game="kuhn")
    assert set(imp) == set(labels)
    for row in imp.values():
        assert all(v >= -1e-9 for v in row)


def test_sensitivity_zero_when_unconstrained():
    """Verify that sensitivity zero when unconstrained."""
    labels = _kuhn_p2_labels()
    intervals = {label: [(0.0, 1.0), (0.0, 1.0)] for label in labels}
    imp = confidence_sensitivity(intervals, v_ref=KUHN_VALUE, eps_safe=0.0, game="kuhn")
    assert all(v == 0.0 for row in imp.values() for v in row)


def test_sensitivity_concentrates_on_binding_sites_leduc():
    """Verify that sensitivity concentrates on binding sites Leduc."""
    from safe_observation.opponents import leduc_static_biased_opponent

    store = OpponentEvidenceStore.for_game("leduc")
    beh0 = compile_leduc(0).behavior_from_realization(
        native.blueprint_realization("leduc", 0)
    )
    _, counts = native.simulate(
        "leduc", beh0, leduc_static_biased_opponent().behavior, 3000, 7
    )
    for label, row in counts.items():
        store.record(label, row)
    v = native.blueprint_lp("leduc")[0]
    imp = confidence_sensitivity(
        store.intervals(0.05), v_ref=v, eps_safe=0.0, game="leduc"
    )
    nonzero = [label for label, row in imp.items() if sum(row) > 1e-9]
    assert 0 < len(nonzero) < len(imp) // 2


def test_sensitivity_weighted_weights_differ_from_uniform():
    """Verify that sensitivity weighted weights differ from uniform."""
    store = OpponentEvidenceStore.for_game("leduc")
    beh0 = compile_leduc(0).behavior_from_realization(
        native.blueprint_realization("leduc", 0)
    )
    _, counts = native.simulate(
        "leduc", beh0, leduc_low_reach_leak_opponent(0.9).behavior, 3000, 7
    )
    for label, row in counts.items():
        store.record(label, row)
    v = native.blueprint_lp("leduc")[0]
    intervals = store.intervals(0.05)
    uniform = weights_from_intervals(intervals)
    importance = confidence_sensitivity(intervals, v_ref=v, eps_safe=0.0, game="leduc")
    weighted = weights_from_intervals(intervals, importance)
    assert uniform != weighted

    assert any(weighted[label] == 0.0 < uniform[label] for label in uniform)


def test_zero_budget_grants_no_slack():
    """Verify that zero budget grants no slack."""
    b = ProbeBudget(total=0.0)
    assert b.allowance() == 0.0
    b.charge(0.0)
    assert b.allowance() == 0.0


def test_budget_allowance_tracks_spend_and_per_round_cap():
    """Verify that budget allowance tracks spend and per round cap."""
    b = ProbeBudget(total=1.0, per_round=0.3)
    assert b.allowance() == pytest.approx(0.3)
    b.charge(0.3)
    b.charge(0.3)
    assert b.remaining() == pytest.approx(0.4)
    assert b.allowance() == pytest.approx(0.3)
    b.charge(0.3)
    assert b.remaining() == pytest.approx(0.1)
    assert b.allowance() == pytest.approx(0.1)


def test_budget_rejects_negative_charge():
    """Verify that budget rejects negative charge."""
    with pytest.raises(ValueError):
        ProbeBudget(total=1.0).charge(-0.1)


def test_safety_grant_proportional_to_signal():
    """Verify that safety grant proportional to signal."""
    b = SafetyBudgetLedger(rho_cap=0.5, s_scale=2.0, gamma=1.0, debt_max=10.0)
    assert b.allowance(0.0) == 0.0
    assert b.allowance(1.0) == pytest.approx(0.25)
    assert b.allowance(2.0) == pytest.approx(0.5)
    assert b.allowance(10.0) == pytest.approx(0.5)


def test_safety_budget_gamma_shapes_the_grant():
    """Verify that safety budget gamma shapes the grant."""
    lo = SafetyBudgetLedger(rho_cap=0.5, s_scale=2.0, gamma=0.5, debt_max=10.0)
    hi = SafetyBudgetLedger(rho_cap=0.5, s_scale=2.0, gamma=2.0, debt_max=10.0)

    assert lo.allowance(1.0) > 0.25 > hi.allowance(1.0)


def test_safety_debt_rises_when_spend_not_recouped():
    """Verify that safety debt rises when spend not recouped."""
    b = SafetyBudgetLedger(rho_cap=0.5, s_scale=1.0, gamma=1.0, debt_max=1.0)
    total = 0.0
    for _ in range(10):
        rho = b.allowance(5.0)
        b.charge(rho)
        b.settle(rho, realized_gain=0.0)
        total += rho

    assert b.debt == pytest.approx(1.0, abs=1e-9)
    assert total == pytest.approx(1.0, abs=1e-9)
    assert b.allowance(5.0) == 0.0


def test_safety_debt_shrinks_with_realized_gain():
    """Verify that safety debt shrinks with realized gain."""
    b = SafetyBudgetLedger(rho_cap=0.5, s_scale=1.0, gamma=1.0, debt_max=1.0)
    for _ in range(10):
        rho = b.allowance(5.0)
        b.charge(rho)
        b.settle(rho, realized_gain=2.0)
    assert b.debt == pytest.approx(0.0)
    assert b.allowance(5.0) == pytest.approx(0.5)


def test_safety_hard_budget_caps_total_spend():
    """Verify that safety hard budget caps total spend."""
    b = SafetyBudgetLedger(rho_cap=0.5, hard_total=1.0, s_scale=1.0, debt_max=100.0)
    spent = 0.0
    for _ in range(10):
        rho = b.allowance(5.0)
        b.charge(rho)
        b.settle(rho, realized_gain=5.0)
        spent += rho
    assert spent == pytest.approx(1.0)
    assert b.allowance(5.0) == 0.0


def test_safety_eta_decay_lets_debt_fade():
    """Verify that safety eta decay lets debt fade."""
    b = SafetyBudgetLedger(rho_cap=0.5, eta_debt=0.5, s_scale=1.0, debt_max=10.0)
    b.settle(1.0, realized_gain=0.0)
    assert b.debt == pytest.approx(1.0)
    b.settle(0.0, realized_gain=0.0)
    assert b.debt == pytest.approx(0.5)


def test_safety_rejects_negative_spend():
    """Verify that safety rejects negative spend."""
    with pytest.raises(ValueError):
        SafetyBudgetLedger().charge(-0.1)
    with pytest.raises(ValueError):
        SafetyBudgetLedger().settle(-0.1, 0.0)


def test_safety_micro_trial_adds_grant_only_above_threshold():
    """Verify that safety micro trial adds grant only above threshold."""
    b = SafetyBudgetLedger(
        rho_cap=0.5, s_scale=1.0, gamma=1.0, micro_rho=0.05, tau_point=0.5
    )

    assert b.allowance(0.0, point_signal=1.0) == pytest.approx(0.05)

    assert b.allowance(0.0, point_signal=0.3) == 0.0

    assert b.allowance(2.0, point_signal=1.0) == pytest.approx(0.5)

    off = SafetyBudgetLedger(rho_cap=0.5, s_scale=1.0, micro_rho=0.0)
    assert off.allowance(0.0, point_signal=10.0) == 0.0


def test_planner_bundles_weights_and_budget():
    """Verify that planner bundles weights and budget."""
    planner = ProbePlanner(beta=2.0, budget=ProbeBudget(total=0.5))
    w = planner.weights({"a": [(0.1, 0.4)]})
    assert w["a"] == pytest.approx(0.3)
    assert planner.rho() == pytest.approx(0.5)
    planner.record(0.2)
    assert planner.rho() == pytest.approx(0.3)


def test_probe_lp_delegates_to_passive_when_beta_and_rho_zero():
    """Verify that probe linear program delegates to passive when beta and rho zero."""
    intervals = {label: [(1.0, 1.0), (0.0, 0.0)] for label in _kuhn_p2_labels()}
    passive = robust_safe_response(
        intervals, v_ref=KUHN_VALUE, eps_safe=0.0, game="kuhn"
    )
    probe = robust_safe_response_probe(
        intervals,
        {},
        {},
        v_ref=KUHN_VALUE,
        eps_safe=0.0,
        beta=0.0,
        rho=0.0,
        game="kuhn",
    )
    assert probe.robust_value == pytest.approx(passive.robust_value)
    assert probe.realization == pytest.approx(passive.realization)


def test_probe_lp_hard_budget_stays_safe():
    """Verify that probe linear program hard budget stays safe."""
    labels = _kuhn_p2_labels()
    intervals = {label: [(0.0, 1.0), (0.0, 1.0)] for label in labels}
    weights = {label: 1.0 for label in labels}
    opp_behavior = {label: [0.5, 0.5] for label in labels}
    probe = robust_safe_response_probe(
        intervals,
        opp_behavior,
        weights,
        v_ref=KUHN_VALUE,
        eps_safe=0.0,
        beta=5.0,
        rho=0.0,
        game="kuhn",
    )
    assert probe.rho_spent() == pytest.approx(0.0, abs=1e-8)
    assert probe.safety_value() >= KUHN_VALUE - 1e-8
    assert probe.is_within_budget()


def _leduc_leak_evidence(hands: int = 2000, seed: int = 7) -> OpponentEvidenceStore:
    """Compute Leduc leak evidence for the test probe workflow."""
    store = OpponentEvidenceStore.for_game("leduc")
    opp = leduc_low_reach_leak_opponent(0.9)
    beh0 = compile_leduc(0).behavior_from_realization(
        native.blueprint_realization("leduc", 0)
    )
    _, counts = native.simulate("leduc", beh0, opp.behavior, hands, seed)
    for label, row in counts.items():
        store.record(label, row)
    return store


def test_probe_budget_increases_information_gain_and_exploitation():
    """Verify that probe budget increases information gain and exploitation."""
    store = _leduc_leak_evidence()
    intervals = store.intervals(0.05)
    opp_behavior = {label: list(store.p_hat(label)) for label in store.labels}
    weights = weights_from_intervals(intervals)
    coeffs = probe_coefficients(opp_behavior, weights, game="leduc")

    passive = robust_safe_response(
        intervals, v_ref=LEDUC_VALUE, eps_safe=0.0, game="leduc"
    )
    hard = robust_safe_response_probe(
        intervals,
        opp_behavior,
        weights,
        v_ref=LEDUC_VALUE,
        eps_safe=0.0,
        beta=1.0,
        rho=0.0,
        game="leduc",
    )
    budgeted = robust_safe_response_probe(
        intervals,
        opp_behavior,
        weights,
        v_ref=LEDUC_VALUE,
        eps_safe=0.0,
        beta=1.0,
        rho=0.5,
        game="leduc",
    )

    ig_hard = information_gain(coeffs, hard.realization)
    ig_budgeted = information_gain(coeffs, budgeted.realization)

    assert ig_budgeted > ig_hard + 1e-6

    assert budgeted.robust_value > passive.robust_value + 1e-3


def test_probe_budget_respects_granted_slack():
    """Verify that probe budget respects granted slack."""
    store = _leduc_leak_evidence()
    intervals = store.intervals(0.05)
    opp_behavior = {label: list(store.p_hat(label)) for label in store.labels}
    weights = weights_from_intervals(intervals)
    rho = 0.5
    budgeted = robust_safe_response_probe(
        intervals,
        opp_behavior,
        weights,
        v_ref=LEDUC_VALUE,
        eps_safe=0.0,
        beta=5.0,
        rho=rho,
        game="leduc",
    )

    assert budgeted.is_within_budget()
    assert budgeted.rho_spent() <= rho + 1e-8


def test_probe_hard_safe_matches_theorem1_on_leak():
    """Verify that probe hard safe matches theorem1 on leak."""
    store = _leduc_leak_evidence()
    intervals = store.intervals(0.05)
    opp_behavior = {label: list(store.p_hat(label)) for label in store.labels}
    weights = weights_from_intervals(intervals)
    hard = robust_safe_response_probe(
        intervals,
        opp_behavior,
        weights,
        v_ref=LEDUC_VALUE,
        eps_safe=0.0,
        beta=1.0,
        rho=0.0,
        game="leduc",
    )

    assert hard.safety_value() >= LEDUC_VALUE - 1e-8
    assert hard.rho_spent() == pytest.approx(0.0, abs=1e-8)


def test_probe_weights_are_higher_on_underexplored_cap_sites():
    """Verify that probe weights are higher on underexplored cap sites."""
    store = _leduc_leak_evidence()
    intervals = store.intervals(0.05)
    cap = set(_leduc_cap_facing_labels())

    def mean_width(label: str) -> float:
        """Compute mean width for the test probe workflow."""
        bounds = intervals[label]
        return sum(u - lo for lo, u in bounds) / len(bounds)

    cap_mean = sum(mean_width(label) for label in cap) / len(cap)
    others = [label for label in intervals if label not in cap]
    non_mean = sum(mean_width(label) for label in others) / len(others)
    assert cap_mean > non_mean
