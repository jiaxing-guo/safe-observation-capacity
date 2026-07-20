""

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
    intervals = {"a": [(0.1, 0.4), (0.0, 0.5)], "b": [(0.2, 0.2)]}
    w = weights_from_intervals(intervals)
    assert w["a"] == pytest.approx(0.3 + 0.5)
    assert w["b"] == pytest.approx(0.0)


def test_weights_apply_importance():
    intervals = {"a": [(0.0, 0.4), (0.0, 0.2)]}
    importance = {"a": [2.0, 0.5]}
    w = weights_from_intervals(intervals, importance)
    assert w["a"] == pytest.approx(0.4 * 2.0 + 0.2 * 0.5)


def test_probe_coefficients_length_matches_player1_sequences():
    n_seq = len(compile_kuhn(0).sequences)
    coeffs = probe_coefficients(
        {}, {label: 1.0 for label in _kuhn_p2_labels()}, game="kuhn"
    )
    assert len(coeffs) == n_seq


def test_probe_coefficients_zero_weights_are_zero():
    coeffs = probe_coefficients({}, {}, game="kuhn")
    assert all(c == 0.0 for c in coeffs)


def test_information_gain_is_the_dot_product():
    coeffs = (1.0, 2.0, 3.0)
    x = (0.5, 0.0, 2.0)
    assert information_gain(coeffs, x) == pytest.approx(0.5 + 0.0 + 6.0)


def _kuhn_p2_labels():
    return [info.label for info in compile_kuhn(1).info_sets]


def test_sensitivity_is_nonnegative_per_action():

    labels = _kuhn_p2_labels()
    intervals = {label: [(0.3, 0.7), (0.3, 0.7)] for label in labels}
    imp = confidence_sensitivity(intervals, v_ref=KUHN_VALUE, eps_safe=0.0, game="kuhn")
    assert set(imp) == set(labels)
    for row in imp.values():
        assert all(v >= -1e-9 for v in row)


def test_sensitivity_zero_when_unconstrained():

    labels = _kuhn_p2_labels()
    intervals = {label: [(0.0, 1.0), (0.0, 1.0)] for label in labels}
    imp = confidence_sensitivity(intervals, v_ref=KUHN_VALUE, eps_safe=0.0, game="kuhn")
    assert all(v == 0.0 for row in imp.values() for v in row)


def test_sensitivity_concentrates_on_binding_sites_leduc():

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
    b = ProbeBudget(total=0.0)
    assert b.allowance() == 0.0
    b.charge(0.0)
    assert b.allowance() == 0.0


def test_budget_allowance_tracks_spend_and_per_round_cap():
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
    with pytest.raises(ValueError):
        ProbeBudget(total=1.0).charge(-0.1)


def test_safety_grant_proportional_to_signal():
    b = SafetyBudgetLedger(rho_cap=0.5, s_scale=2.0, gamma=1.0, debt_max=10.0)
    assert b.allowance(0.0) == 0.0
    assert b.allowance(1.0) == pytest.approx(0.25)
    assert b.allowance(2.0) == pytest.approx(0.5)
    assert b.allowance(10.0) == pytest.approx(0.5)


def test_safety_budget_gamma_shapes_the_grant():
    lo = SafetyBudgetLedger(rho_cap=0.5, s_scale=2.0, gamma=0.5, debt_max=10.0)
    hi = SafetyBudgetLedger(rho_cap=0.5, s_scale=2.0, gamma=2.0, debt_max=10.0)

    assert lo.allowance(1.0) > 0.25 > hi.allowance(1.0)


def test_safety_debt_rises_when_spend_not_recouped():

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

    b = SafetyBudgetLedger(rho_cap=0.5, s_scale=1.0, gamma=1.0, debt_max=1.0)
    for _ in range(10):
        rho = b.allowance(5.0)
        b.charge(rho)
        b.settle(rho, realized_gain=2.0)
    assert b.debt == pytest.approx(0.0)
    assert b.allowance(5.0) == pytest.approx(0.5)


def test_safety_hard_budget_caps_total_spend():
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

    b = SafetyBudgetLedger(rho_cap=0.5, eta_debt=0.5, s_scale=1.0, debt_max=10.0)
    b.settle(1.0, realized_gain=0.0)
    assert b.debt == pytest.approx(1.0)
    b.settle(0.0, realized_gain=0.0)
    assert b.debt == pytest.approx(0.5)


def test_safety_rejects_negative_spend():
    with pytest.raises(ValueError):
        SafetyBudgetLedger().charge(-0.1)
    with pytest.raises(ValueError):
        SafetyBudgetLedger().settle(-0.1, 0.0)


def test_safety_micro_trial_adds_grant_only_above_threshold():

    b = SafetyBudgetLedger(
        rho_cap=0.5, s_scale=1.0, gamma=1.0, micro_rho=0.05, tau_point=0.5
    )

    assert b.allowance(0.0, point_signal=1.0) == pytest.approx(0.05)

    assert b.allowance(0.0, point_signal=0.3) == 0.0

    assert b.allowance(2.0, point_signal=1.0) == pytest.approx(0.5)

    off = SafetyBudgetLedger(rho_cap=0.5, s_scale=1.0, micro_rho=0.0)
    assert off.allowance(0.0, point_signal=10.0) == 0.0


def test_planner_bundles_weights_and_budget():
    planner = ProbePlanner(beta=2.0, budget=ProbeBudget(total=0.5))
    w = planner.weights({"a": [(0.1, 0.4)]})
    assert w["a"] == pytest.approx(0.3)
    assert planner.rho() == pytest.approx(0.5)
    planner.record(0.2)
    assert planner.rho() == pytest.approx(0.3)


def test_probe_lp_delegates_to_passive_when_beta_and_rho_zero():
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

    store = _leduc_leak_evidence()
    intervals = store.intervals(0.05)
    cap = set(_leduc_cap_facing_labels())

    def mean_width(label: str) -> float:
        bounds = intervals[label]
        return sum(u - lo for lo, u in bounds) / len(bounds)

    cap_mean = sum(mean_width(label) for label in cap) / len(cap)
    others = [label for label in intervals if label not in cap]
    non_mean = sum(mean_width(label) for label in others) / len(others)
    assert cap_mean > non_mean
