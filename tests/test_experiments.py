""

import pytest

from safe_observation.experiments.online import (
    DEFAULT_SEEDS,
    run_ablation,
    run_online_adaptation,
    run_online_adaptation_replicated,
)
from safe_observation.opponents import static_biased_opponent

KNOWN_KUHN_VALUE = -1.0 / 18.0


def test_default_seeds_are_42_to_46():
    assert DEFAULT_SEEDS == (42, 43, 44, 45, 46)


def test_rounds_log_has_new_series_keys():
    results = run_online_adaptation(
        static_biased_opponent(), rounds=3, episodes_per_round=20, out_dir=None
    )
    row = results["rounds_log"][0]
    assert {"empirical_br_value", "mean_ci_width", "ci_width_by_infoset"} <= set(row)
    assert len(row["ci_width_by_infoset"]) == 6


def test_ci_width_shrinks_with_evidence():

    results = run_online_adaptation(
        static_biased_opponent(), rounds=40, episodes_per_round=200, out_dir=None
    )
    widths = [r["mean_ci_width"] for r in results["rounds_log"]]
    assert widths[-1] < widths[0]


def test_replicated_run_aggregates_and_is_safe():
    results = run_online_adaptation_replicated(
        static_biased_opponent(bet_prob=0.05),
        rounds=15,
        episodes_per_round=100,
        seeds=(42, 43, 44),
        out_dir=None,
    )
    assert results["seeds"] == [42, 43, 44]
    assert results["safety_preserved"]
    assert results["min_safety_value"] >= KNOWN_KUHN_VALUE - 1e-8
    assert results["exploitation_gain_mean"] > 0.0
    assert len(results["per_seed"]) == 3

    row = results["aggregated_rounds"][0]
    assert {"actual_value_mean", "actual_value_std", "safety_value_mean"} <= set(row)
    assert len(row["ci_width_by_infoset_mean"]) == 6


def test_ablation_sweeps_three_axes():
    ablation = run_ablation(
        static_biased_opponent(),
        deltas=(0.05, 0.2),
        eps_safes=(0.0, 0.1),
        methods=("hoeffding",),
        rounds=8,
        episodes_per_round=40,
        seeds=(42, 43),
        out_dir=None,
    )
    assert set(ablation) == {"delta", "eps_safe", "method"}
    assert len(ablation["delta"]) == 2
    assert len(ablation["eps_safe"]) == 2

    for cells in ablation.values():
        for cell in cells:
            assert cell["safety_preserved"]


def test_render_online_figures_smoke(tmp_path):
    pytest.importorskip("matplotlib")
    from safe_observation.evaluation.plots import render_online_figures

    results = run_online_adaptation_replicated(
        static_biased_opponent(),
        rounds=10,
        episodes_per_round=50,
        seeds=(42, 43),
        out_dir=None,
    )
    saved = render_online_figures(results, out_dir=tmp_path)
    assert set(saved) == {"value", "safety", "ci_shrinkage", "robust_vs_br"}
    for path in saved.values():
        assert path.exists() and path.stat().st_size > 0


def test_render_ablation_figure_smoke(tmp_path):
    pytest.importorskip("matplotlib")
    from safe_observation.evaluation.plots import render_ablation_figure

    ablation = run_ablation(
        static_biased_opponent(),
        deltas=(0.05, 0.2),
        eps_safes=(0.0,),
        methods=("hoeffding",),
        rounds=6,
        episodes_per_round=30,
        seeds=(42,),
        out_dir=None,
    )
    path = render_ablation_figure(ablation, "static_biased", out_dir=tmp_path)
    assert path.exists() and path.stat().st_size > 0
