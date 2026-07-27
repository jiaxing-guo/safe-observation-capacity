"""Regression tests for test showdown. See the corresponding implementation module and supplementary Reproducibility."""

from safe_observation import native
from safe_observation.confidence import OpponentEvidenceStore
from safe_observation.experiments.online import run_showdown_comparison
from safe_observation.opponents import (
    holdem_censored_fold_opponent,
    holdem_equilibrium_opponent,
    leduc_equilibrium_opponent,
    leduc_low_reach_leak_opponent,
    leduc_static_biased_opponent,
)
from safe_observation.sequence_form import compile as compile_game
from safe_observation.solvers import (
    confidence_guarded_point_probe,
    robust_safe_response_public,
)

LEDUC_VALUE = native.blueprint_lp("leduc")[0]
HOLDEM_VALUE = native.blueprint_lp("holdem")[0]


def _blueprint_behaviors():
    """Compute blueprint behaviors for the test showdown workflow."""
    sf0 = compile_game("leduc", 0)
    sf1 = compile_game("leduc", 1)
    beh0 = sf0.behavior_from_realization(native.blueprint_realization("leduc", 0))
    beh1 = sf1.behavior_from_realization(native.blueprint_realization("leduc", 1))
    return beh0, beh1


def test_showdown_split_sums_to_full_counts():
    """Verify that showdown split sums to full counts."""
    beh0, beh1 = _blueprint_behaviors()
    full_pay, full = native.simulate("leduc", beh0, beh1, 6000, 2026)
    pay, show, fold = native.simulate_showdown("leduc", beh0, beh1, 6000, 2026)

    assert pay == full_pay

    assert show and fold
    for label, counts in full.items():
        s = show.get(label, [0] * len(counts))
        f = fold.get(label, [0] * len(counts))
        assert [a + b for a, b in zip(s, f, strict=True)] == counts


def test_showdown_estimate_is_more_biased_than_full():
    """Verify that showdown estimate is more biased than full."""
    opp = leduc_static_biased_opponent()
    beh0, _ = _blueprint_behaviors()
    ev_show = OpponentEvidenceStore.for_game("leduc")
    ev_full = OpponentEvidenceStore.for_game("leduc")
    for t in range(20):
        _pay, show, fold = native.simulate_showdown(
            "leduc", beh0, opp.behavior, 200, 100 + t
        )
        for label, c in show.items():
            ev_show.record(label, c)
            ev_full.record(label, c)
        for label, c in fold.items():
            ev_full.record(label, c)

    def mean_bias(store):
        """Compute mean bias for the test showdown workflow."""
        tot, n = 0.0, 0
        for label in store.labels:
            if label not in opp.behavior or store.visits(label) < 30:
                continue
            phat = store.p_hat(label)
            truth = opp.behavior[label]
            tot += 0.5 * sum(abs(a - b) for a, b in zip(phat, truth, strict=True))
            n += 1
        return tot / max(1, n)

    assert mean_bias(ev_show) > 2.0 * mean_bias(ev_full)


def _public_set(opp, rounds=20, seed=7):
    """Compute public set for the test showdown workflow."""
    beh0, _ = _blueprint_behaviors()
    ev = OpponentEvidenceStore.for_game("leduc")
    for t in range(rounds):
        _pay, show, fold = native.simulate_showdown(
            "leduc", beh0, opp.behavior, 200, seed + t
        )
        for label, c in show.items():
            ev.record(label, c)
        for label, c in fold.items():
            ev.record(label, c)
    return ev.public_groups(), ev.public_intervals(0.1)


def test_guard_plan_respects_the_budget_relaxed_floor():
    """Verify that guard plan respects the budget relaxed floor."""
    opp = leduc_static_biased_opponent()
    groups, intervals = _public_set(opp)
    y_hat = list(native.blueprint_realization("leduc", 1))
    rho = 0.5
    resp = confidence_guarded_point_probe(
        groups,
        intervals,
        y_hat,
        v_ref=LEDUC_VALUE,
        eps_safe=0.0,
        rho=rho,
        kappa=0.2,
        game="leduc",
    )
    assert resp.safety_value() >= LEDUC_VALUE - rho - 1e-6


def test_tight_guard_approaches_robust_public_value():
    """Verify that tight guard approaches robust public value."""
    opp = leduc_low_reach_leak_opponent()
    groups, intervals = _public_set(opp)
    y_hat = list(native.blueprint_realization("leduc", 1))
    rho = 0.5
    j_rho = robust_safe_response_public(
        groups, intervals, v_ref=LEDUC_VALUE, eps_safe=rho, game="leduc"
    ).robust_value
    tight = confidence_guarded_point_probe(
        groups,
        intervals,
        y_hat,
        v_ref=LEDUC_VALUE,
        eps_safe=0.0,
        rho=rho,
        kappa=0.0,
        game="leduc",
    )

    assert tight.safety_value() >= LEDUC_VALUE - rho - 1e-6
    assert abs(tight.j_rho - j_rho) < 1e-9


def test_guard_recovers_censored_leak_and_all_methods_stay_safe():
    """Verify that guard recovers censored leak and all methods stay safe."""
    suite = {
        "equilibrium": leduc_equilibrium_opponent(),
        "low_reach_leak": leduc_low_reach_leak_opponent(),
    }
    res = run_showdown_comparison(
        suite,
        rounds=30,
        episodes_per_round=200,
        kappa=0.2,
        seeds=(42, 43),
        out_dir=None,
        workers=None,
    )
    assert res["monitoring"] == "showdown"
    leak = res["opponents"]["low_reach_leak"]["methods"]

    assert set(res["method_names"]) == {
        "blueprint",
        "passive_public",
        "rnr",
        "gift_based",
        "public_point",
        "censored_em",
        "public_robust",
        "point_response",
        "confidence_guarded",
        "safe_active_decensoring",
    }

    assert (
        leak["confidence_guarded"]["exploitation_gain_mean"]
        > leak["point_response"]["exploitation_gain_mean"] + 0.5
    )

    for opp_row in res["opponents"].values():
        for m in opp_row["methods"].values():
            assert m["guarantee_held"]


def test_uncertified_baselines_breach_the_floor_that_safe_response_respects():
    """Verify that uncertified baselines breach the floor that safe response respects."""
    suite = {"static_biased": leduc_static_biased_opponent()}
    rho_cap = 0.5
    res = run_showdown_comparison(
        suite,
        rounds=30,
        episodes_per_round=200,
        kappa=0.2,
        rho_cap=rho_cap,
        seeds=(42, 43),
        out_dir=None,
        workers=None,
    )
    m = res["opponents"]["static_biased"]["methods"]
    certified_floor = LEDUC_VALUE - rho_cap

    assert m["rnr"]["min_safety_value"] < certified_floor - 1e-3
    assert m["gift_based"]["min_safety_value"] < certified_floor - 1e-3

    assert m["point_response"]["min_safety_value"] >= certified_floor - 1e-6
    assert m["confidence_guarded"]["min_safety_value"] >= certified_floor - 1e-6

    assert m["rnr"]["guarantee"] == "none"
    assert m["gift_based"]["guarantee"] == "gift_funded"
    assert m["gift_based"]["guarantee_held"]


def test_point_response_shows_a_censoring_phantom():
    """Verify that point response shows a censoring phantom."""
    suite = {"equilibrium": leduc_equilibrium_opponent()}
    res = run_showdown_comparison(
        suite,
        rounds=30,
        episodes_per_round=200,
        kappa=0.2,
        seeds=(42, 43),
        out_dir=None,
        workers=None,
    )
    eq = res["opponents"]["equilibrium"]["methods"]

    assert eq["point_response"]["phantom_gap_mean"] > 0.05

    assert eq["point_response"]["exploitation_gain_mean"] < 0.1


def test_holdem_guard_recovers_censored_fold_at_hunl_scale():
    """Verify that holdem guard recovers censored fold at hunl scale."""
    suite = {
        "equilibrium": holdem_equilibrium_opponent(),
        "censored_fold": holdem_censored_fold_opponent(),
    }
    rho_cap = 0.5
    res = run_showdown_comparison(
        suite,
        rounds=12,
        episodes_per_round=120,
        kappa=0.2,
        rho_cap=rho_cap,
        seeds=(42,),
        out_dir=None,
        workers=None,
    )
    assert res["game"] == "holdem"
    leak = res["opponents"]["censored_fold"]["methods"]

    assert (
        leak["confidence_guarded"]["exploitation_gain_mean"]
        > leak["point_response"]["exploitation_gain_mean"] + 0.2
    )

    certified_floor = HOLDEM_VALUE - rho_cap
    for opp_row in res["opponents"].values():
        m = opp_row["methods"]
        assert m["confidence_guarded"]["min_safety_value"] >= certified_floor - 1e-6
        assert m["point_response"]["min_safety_value"] >= certified_floor - 1e-6

    eq = res["opponents"]["equilibrium"]["methods"]
    assert eq["confidence_guarded"]["exploitation_gain_mean"] < 0.1

    assert set(res["wall_time_by_method"]) == set(res["method_names"])
    assert all(t >= 0.0 for t in res["wall_time_by_method"].values())
