"""Regression tests for test baselines. See the corresponding implementation module and supplementary Reproducibility."""

import pytest

from safe_observation import native
from safe_observation.experiments.online import (
    run_adversarial_stress,
    run_baseline_comparison,
    run_nonstationary_stress,
)
from safe_observation.opponents import (
    leduc_equilibrium_opponent,
    leduc_low_reach_leak_opponent,
    leduc_near_equilibrium_opponent,
    leduc_static_biased_opponent,
)
from safe_observation.payoff import build as build_payoff
from safe_observation.solvers import (
    best_response,
    floor_shadow_price,
    restricted_nash_response,
    safety_filtered_restricted_nash_response,
    safety_verifier,
)

LEDUC_VALUE = native.blueprint_lp("leduc")[0]


def test_rnr_p_zero_is_blueprint_value():
    """Verify that restricted Nash response p zero is blueprint value."""
    opp = leduc_static_biased_opponent()
    r = restricted_nash_response(list(opp.realization()), 0.0, game="leduc")
    assert r.value == pytest.approx(LEDUC_VALUE, abs=1e-6)
    assert r.safety_value() >= LEDUC_VALUE - 1e-8


def test_rnr_p_one_matches_best_response():
    """Verify that restricted Nash response p one matches best response."""
    opp = leduc_static_biased_opponent()
    y = list(opp.realization())
    r = restricted_nash_response(y, 1.0, game="leduc")
    br = best_response(y, game="leduc")
    assert r.value == pytest.approx(br.value, abs=1e-6)


def test_rnr_validates_p():
    """Verify that restricted Nash response validates p."""
    y = list(leduc_static_biased_opponent().realization())
    with pytest.raises(ValueError):
        restricted_nash_response(y, 1.5, game="leduc")


def test_rnr_exploitation_monotone_safety_degrades_in_p():
    """Verify that restricted Nash response exploitation monotone safety degrades in p."""
    opp = leduc_static_biased_opponent()
    y = list(opp.realization())
    payoff = build_payoff("leduc")
    last_expl = float("-inf")
    last_safety = float("inf")
    for p in (0.0, 0.25, 0.5, 0.75, 1.0):
        r = restricted_nash_response(y, p, game="leduc")
        expl = payoff.bilinear(r.realization, y)
        safety = safety_verifier(r.realization, game="leduc").value
        assert expl >= last_expl - 1e-9
        assert safety <= last_safety + 1e-9
        last_expl, last_safety = expl, safety

    assert last_safety < LEDUC_VALUE - 1e-6


def test_rnr_against_equilibrium_stays_near_game_value():
    """Verify that restricted Nash response against equilibrium stays near game value."""
    opp = leduc_equilibrium_opponent()
    r = restricted_nash_response(list(opp.realization()), 1.0, game="leduc")
    assert r.value == pytest.approx(LEDUC_VALUE, abs=1e-3)


def test_safety_filtered_rnr_clears_the_floor():
    """Verify that safety filtered restricted Nash response clears the floor."""
    y = list(leduc_static_biased_opponent().realization())
    floor = LEDUC_VALUE - 0.5
    r = safety_filtered_restricted_nash_response(y, floor=floor, game="leduc")
    assert r.safety_value >= floor - 1e-6
    assert r.safety_value == pytest.approx(
        safety_verifier(r.realization, game="leduc").value, abs=1e-9
    )
    assert 0.0 <= r.p <= 1.0


def test_safety_filtered_rnr_hard_floor_is_blueprint_safe():
    """Verify that safety filtered restricted Nash response hard floor is blueprint safe."""
    y = list(leduc_static_biased_opponent().realization())
    r = safety_filtered_restricted_nash_response(y, floor=LEDUC_VALUE, game="leduc")
    assert r.safety_value >= LEDUC_VALUE - 1e-6


def test_safety_filtered_rnr_spends_more_p_with_more_budget():
    """Verify that safety filtered restricted Nash response spends more p with more budget."""
    y = list(leduc_static_biased_opponent().realization())
    payoff = build_payoff("leduc")
    tight = safety_filtered_restricted_nash_response(
        y, floor=LEDUC_VALUE - 0.1, game="leduc"
    )
    loose = safety_filtered_restricted_nash_response(
        y, floor=LEDUC_VALUE - 1.0, game="leduc"
    )
    assert loose.p >= tight.p - 1e-9
    assert (
        payoff.bilinear(loose.realization, y)
        >= payoff.bilinear(tight.realization, y) - 1e-6
    )


def test_safety_filtered_rnr_equilibrium_stays_safe():
    """Verify that safety filtered restricted Nash response equilibrium stays safe."""
    y = list(leduc_equilibrium_opponent().realization())
    r = safety_filtered_restricted_nash_response(
        y, floor=LEDUC_VALUE - 0.5, game="leduc"
    )
    assert r.safety_value >= LEDUC_VALUE - 0.5 - 1e-6


def _point_intervals(opponent):
    """Compute the point intervals for the test baselines workflow."""
    from safe_observation.confidence import OpponentEvidenceStore

    store = OpponentEvidenceStore.for_game(opponent.game)
    return {label: [(p, p) for p in opponent.behavior[label]] for label in store.labels}


def test_floor_shadow_price_large_for_exploitable_zero_for_equilibrium():
    """Verify that floor shadow price large for exploitable zero for equilibrium."""
    biased = floor_shadow_price(
        _point_intervals(leduc_static_biased_opponent()),
        v_ref=LEDUC_VALUE,
        game="leduc",
    )
    leak = floor_shadow_price(
        _point_intervals(leduc_low_reach_leak_opponent(0.9)),
        v_ref=LEDUC_VALUE,
        game="leduc",
    )
    eq = floor_shadow_price(
        _point_intervals(leduc_equilibrium_opponent()),
        v_ref=LEDUC_VALUE,
        game="leduc",
    )
    assert biased > 0.5
    assert leak > 0.5
    assert abs(eq) <= 1e-6

    assert leak > biased


def test_floor_shadow_price_nonnegative():
    """Verify that floor shadow price nonnegative."""
    for opp in (
        leduc_static_biased_opponent(),
        leduc_near_equilibrium_opponent(),
        leduc_equilibrium_opponent(),
    ):
        s = floor_shadow_price(_point_intervals(opp), v_ref=LEDUC_VALUE, game="leduc")
        assert s >= -1e-6


def test_baseline_comparison_methods_and_guarantees():
    """Verify that baseline comparison methods and guarantees."""
    suite = {
        "static_biased": leduc_static_biased_opponent(),
        "low_reach_leak": leduc_low_reach_leak_opponent(0.9),
    }
    res = run_baseline_comparison(
        suite,
        rounds=20,
        episodes_per_round=150,
        rnr_ps=(0.5,),
        seeds=(42, 43),
        out_dir=None,
    )
    assert res["method_names"] == [
        "blueprint",
        "empirical_br",
        "rnr_p0.5",
        "gift_based",
        "passive",
        "probe_budgeted",
        "safety_filtered_rnr",
        "budgeted_empirical_br",
        "value_aware_br",
        "point_response",
    ]
    for row in res["opponents"].values():
        methods = row["methods"]

        for name in ("blueprint", "passive"):
            assert methods[name]["safety_violation_max"] <= 1e-8
            assert methods[name]["guarantee_held"]

        assert methods["probe_budgeted"]["guarantee_held"]

        assert methods["safety_filtered_rnr"]["guarantee"] == "certified_budget"
        assert methods["safety_filtered_rnr"]["guarantee_held"]

        assert methods["budgeted_empirical_br"]["guarantee_held"]
        assert methods["value_aware_br"]["guarantee_held"]

        assert methods["gift_based"]["guarantee"] == "gift_funded"
        assert methods["gift_based"]["guarantee_held"]

        assert methods["empirical_br"]["guarantee"] == "none"
        assert methods["rnr_p0.5"]["guarantee"] == "none"


def test_baseline_comparison_unsafe_methods_violate_floor():
    """Verify that baseline comparison unsafe methods violate floor."""
    suite = {"static_biased": leduc_static_biased_opponent()}
    res = run_baseline_comparison(
        suite,
        rounds=20,
        episodes_per_round=150,
        rnr_ps=(0.75,),
        seeds=(42, 43),
        out_dir=None,
    )
    methods = res["opponents"]["static_biased"]["methods"]

    assert methods["empirical_br"]["safety_violation_max"] > 0.1
    assert (
        methods["empirical_br"]["exploitation_gain_mean"]
        > methods["passive"]["exploitation_gain_mean"]
    )

    assert methods["rnr_p0.75"]["safety_violation_max"] > 1e-6

    assert methods["passive"]["safety_violation_max"] <= 1e-8
    assert methods["passive"]["exploitation_gain_mean"] > 0.1


def test_value_aware_br_spends_nothing_on_equilibrium():
    """Verify that value aware br spends nothing on equilibrium."""
    suite = {"equilibrium": leduc_equilibrium_opponent()}
    res = run_baseline_comparison(
        suite,
        rounds=20,
        episodes_per_round=150,
        rnr_ps=(0.5,),
        seeds=(42, 43),
        out_dir=None,
    )
    m = res["opponents"]["equilibrium"]["methods"]
    va = m["value_aware_br"]

    assert va["gate_rate_mean"] <= 0.1
    assert va["budget_spent_mean"] <= 0.05
    assert va["safety_violation_max"] <= 1e-8

    assert m["budgeted_empirical_br"]["safety_violation_max"] > 0.1


def test_value_aware_br_matches_point_objective_on_biased():
    """Verify that value aware br matches point objective on biased."""
    suite = {"static_biased": leduc_static_biased_opponent()}
    res = run_baseline_comparison(
        suite,
        rounds=20,
        episodes_per_round=150,
        rnr_ps=(0.5,),
        seeds=(42, 43),
        out_dir=None,
    )
    m = res["opponents"]["static_biased"]["methods"]
    va = m["value_aware_br"]
    assert va["gate_rate_mean"] >= 0.5

    assert va["exploitation_gain_mean"] >= (
        m["budgeted_empirical_br"]["exploitation_gain_mean"] - 0.15
    )

    assert (
        va["exploitation_gain_mean"]
        >= m["probe_budgeted"]["exploitation_gain_mean"] - 1e-6
    )
    assert va["guarantee_held"]


def test_point_response_matches_point_on_biased():
    """Verify that point response matches point on biased."""
    suite = {"static_biased": leduc_static_biased_opponent()}
    res = run_baseline_comparison(
        suite,
        rounds=20,
        episodes_per_round=150,
        rnr_ps=(0.5,),
        seeds=(42, 43),
        out_dir=None,
    )
    m = res["opponents"]["static_biased"]["methods"]
    sc = m["point_response"]
    assert sc["guarantee_held"]

    assert sc["exploitation_gain_mean"] >= (
        m["budgeted_empirical_br"]["exploitation_gain_mean"] - 0.15
    )
    assert sc["safety_spend_rate_mean"] >= 0.5

    assert sc["safety_debt_final_mean"] <= 0.5


def test_point_response_withdraws_against_adversary():
    """Verify that point response withdraws against adversary."""
    res = run_adversarial_stress(
        rounds=30, episodes_per_round=150, rnr_ps=(0.5,), seeds=(42, 43), out_dir=None
    )
    floor = res["safety_floor"]
    sc = res["methods"]["point_response"]
    va = res["methods"]["value_aware_br"]

    assert sc["realized_value_mean"] >= floor - 0.1
    assert sc["realized_value_mean"] > va["realized_value_mean"] + 0.2

    assert sc["realized_value_worst"] >= floor - 0.5 - 1e-8


def test_adversarial_stress_safe_methods_hold_floor():
    """Verify that adversarial stress safe methods hold floor."""
    res = run_adversarial_stress(
        rounds=20, episodes_per_round=150, rnr_ps=(0.5,), seeds=(42, 43), out_dir=None
    )
    floor = res["safety_floor"]
    for name in ("blueprint", "passive"):
        d = res["methods"][name]
        assert d["realized_value_worst"] >= floor - 1e-8
        assert d["realized_loss_below_floor"] <= 1e-8
        assert d["guarantee_held"]


def test_adversarial_stress_certified_budget_bounded():
    """Verify that adversarial stress certified budget bounded."""
    res = run_adversarial_stress(
        rounds=20,
        episodes_per_round=150,
        rnr_ps=(0.5,),
        probe_per_round=0.5,
        seeds=(42, 43),
        out_dir=None,
    )
    d = res["methods"]["probe_budgeted"]
    assert d["realized_value_worst"] >= res["safety_floor"] - 0.5 - 1e-8
    assert d["guarantee_held"]


def test_adversarial_stress_unsafe_methods_realize_losses():
    """Verify that adversarial stress unsafe methods realize losses."""
    res = run_adversarial_stress(
        rounds=20, episodes_per_round=150, rnr_ps=(0.5,), seeds=(42, 43), out_dir=None
    )
    floor = res["safety_floor"]

    assert res["methods"]["empirical_br"]["realized_value_worst"] < floor - 0.5
    assert res["methods"]["empirical_br"]["realized_loss_below_floor"] > 0.5
    assert res["methods"]["rnr_p0.5"]["realized_value_worst"] < floor - 1e-6

    assert (
        res["methods"]["passive"]["realized_value_worst"]
        > res["methods"]["empirical_br"]["realized_value_worst"]
    )


def test_gift_based_holds_aggregate_floor_and_exploits_reactively():
    """Verify that gift based holds aggregate floor and exploits reactively."""
    suite = {"static_biased": leduc_static_biased_opponent()}
    res = run_baseline_comparison(
        suite,
        rounds=20,
        episodes_per_round=150,
        rnr_ps=(0.5,),
        seeds=(42, 43),
        out_dir=None,
    )
    g = res["opponents"]["static_biased"]["methods"]["gift_based"]
    assert g["guarantee"] == "gift_funded"

    assert g["guarantee_held"]

    assert g["exploitation_gain_mean"] > 0.0


def test_gift_based_per_round_dips_are_funded_by_winnings():
    """Verify that gift based per round dips are funded by winnings."""
    suite = {"static_biased": leduc_static_biased_opponent()}
    res = run_baseline_comparison(
        suite,
        rounds=30,
        episodes_per_round=150,
        rnr_ps=(0.5,),
        seeds=(42, 43),
        out_dir=None,
    )
    methods = res["opponents"]["static_biased"]["methods"]
    g = methods["gift_based"]

    assert g["min_safety_value"] < methods["passive"]["min_safety_value"] + 1e-9

    assert g["guarantee_held"]

    assert methods["passive"]["safety_violation_max"] <= 1e-8


def test_gift_based_is_aggregate_safe_against_adversary():
    """Verify that gift based is aggregate safe against adversary."""
    res = run_adversarial_stress(
        rounds=20, episodes_per_round=150, rnr_ps=(0.5,), seeds=(42, 43), out_dir=None
    )
    floor = res["safety_floor"]
    g = res["methods"]["gift_based"]
    assert g["guarantee"] == "gift_funded"
    assert g["realized_value_worst"] >= floor - 1e-6
    assert g["realized_loss_below_floor"] <= 1e-6
    assert g["guarantee_held"]

    assert (
        g["realized_value_worst"]
        > res["methods"]["empirical_br"]["realized_value_worst"]
    )


def test_lure_then_strike_separates_keep_from_give_back():
    """Verify that lure then strike separates keep from give back."""
    res = run_nonstationary_stress(
        kind="lure_then_strike",
        rounds=24,
        episodes_per_round=150,
        strike_round=12,
        rnr_ps=(0.5,),
        seeds=(42, 43),
        out_dir=None,
    )
    m = res["methods"]

    floor = res["safety_floor"]
    for name in ("blueprint", "passive"):
        assert m[name]["cumulative_above_floor"] > 0.5
        assert m[name]["realized_value_worst"] >= floor - 1e-8
        assert m[name]["guarantee_held"]

    g = m["gift_based"]
    assert g["peak_above_floor"] > 0.5
    assert g["cumulative_above_floor"] < 0.5 * g["peak_above_floor"]
    assert g["realized_value_worst"] < floor - 0.5
    assert g["guarantee_held"]

    assert m["passive"]["cumulative_above_floor"] > g["cumulative_above_floor"]


def test_lure_then_strike_unsafe_baseline_realizes_net_loss():
    """Verify that lure then strike unsafe baseline realizes net loss."""
    res = run_nonstationary_stress(
        kind="lure_then_strike",
        rounds=24,
        episodes_per_round=150,
        strike_round=12,
        rnr_ps=(0.5,),
        seeds=(42, 43),
        out_dir=None,
    )
    m = res["methods"]
    assert m["empirical_br"]["cumulative_above_floor"] < 0.0
    assert (
        m["passive"]["cumulative_above_floor"]
        > m["empirical_br"]["cumulative_above_floor"]
    )

    assert (
        m["empirical_br"]["realized_value_worst"]
        < m["probe_budgeted"]["realized_value_worst"]
    )


def test_drift_keeps_per_round_safe_methods_safe():
    """Verify that drift keeps per round safe methods safe."""
    res = run_nonstationary_stress(
        kind="drift",
        rounds=24,
        episodes_per_round=150,
        rnr_ps=(0.5,),
        seeds=(42, 43),
        out_dir=None,
    )
    m = res["methods"]
    floor = res["safety_floor"]
    for name in ("blueprint", "passive"):
        assert m[name]["realized_value_worst"] >= floor - 1e-8
        assert m[name]["guarantee_held"]

    assert m["probe_budgeted"]["realized_value_worst"] >= floor - 0.5 - 1e-8

    assert m["gift_based"]["aggregate_min"] >= -1e-6
    assert m["gift_based"]["guarantee_held"]


def test_nonstationary_rejects_unknown_kind():
    """Verify that nonstationary rejects unknown kind."""
    with pytest.raises(ValueError, match="unknown schedule kind"):
        run_nonstationary_stress(
            kind="nope", rounds=4, episodes_per_round=20, seeds=(42,), out_dir=None
        )
