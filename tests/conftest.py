"""Regression tests for conftest. See the corresponding implementation module and supplementary Reproducibility."""

import pytest

SLOW_TESTS: frozenset[str] = frozenset(
    {
        "test_baseline_comparison_methods_and_guarantees",
        "test_baseline_comparison_unsafe_methods_violate_floor",
        "test_value_aware_br_matches_point_objective_on_biased",
        "test_value_aware_br_spends_nothing_on_equilibrium",
        "test_point_response_matches_point_on_biased",
        "test_point_response_withdraws_against_adversary",
        "test_gift_based_holds_aggregate_floor_and_exploits_reactively",
        "test_gift_based_per_round_dips_are_funded_by_winnings",
        "test_gift_based_is_aggregate_safe_against_adversary",
        "test_adversarial_stress_safe_methods_hold_floor",
        "test_adversarial_stress_certified_budget_bounded",
        "test_adversarial_stress_unsafe_methods_realize_losses",
        "test_lure_then_strike_unsafe_baseline_realizes_net_loss",
        "test_lure_then_strike_separates_keep_from_give_back",
        "test_drift_keeps_per_round_safe_methods_safe",
        "test_guard_recovers_censored_leak_and_all_methods_stay_safe",
        "test_uncertified_baselines_breach_the_floor_that_safe_response_respects",
        "test_point_response_shows_a_censoring_phantom",
        "test_holdem_guard_recovers_censored_fold_at_hunl_scale",
        "test_information_bonus_adds_identification_not_exploitation",
        "test_sensitivity_importance_outperforms_uniform_at_small_budget",
        "test_probing_comparison_shrinks_target_intervals",
        "test_probing_comparison_arms_and_metrics",
        "test_budgeted_probing_cracks_the_leak",
        "test_low_reach_leak_probe_improves_identification",
        "test_finite_sample_radius_shrinks_over_rounds",
        "test_finite_sample_gap_decays_and_is_bounded",
    }
)


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Compute pytest collection modifyitems for the conftest workflow."""
    for item in items:
        if getattr(item, "originalname", item.name) in SLOW_TESTS:
            item.add_marker(pytest.mark.slow)
