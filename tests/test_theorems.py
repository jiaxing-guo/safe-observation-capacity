"""Regression tests for test theorems. See the corresponding implementation module and supplementary Reproducibility."""

import pytest

from safe_observation.confidence import OpponentEvidenceStore, time_uniform_delta
from safe_observation.experiments.online import (
    run_coverage_experiment,
    run_finite_sample_gap,
)
from safe_observation.opponents import leduc_static_biased_opponent


def test_time_uniform_delta_sums_to_delta():
    """Verify that time uniform delta sums to delta."""
    delta = 0.1
    total = sum(time_uniform_delta(delta, t) for t in range(1, 100_000))
    assert total == pytest.approx(delta, rel=1e-3)


def test_time_uniform_delta_validates_round():
    """Verify that time uniform delta validates round."""
    with pytest.raises(ValueError):
        time_uniform_delta(0.1, 0)


def test_time_uniform_intervals_are_wider_than_spatial():
    """Verify that time uniform intervals are wider than spatial."""
    store = OpponentEvidenceStore.for_game("leduc")
    for label in store.labels:
        store.record(label, [10] * len(store.p_hat(label)))
    spatial = store.intervals(0.1, union_bound=True)
    anytime = store.intervals(0.1, union_bound=True, round_index=5)
    label = store.labels[0]
    sp_w = spatial[label][0][1] - spatial[label][0][0]
    at_w = anytime[label][0][1] - anytime[label][0][0]
    assert at_w >= sp_w


def test_coverage_union_bound_is_necessary_and_sufficient():
    """Verify that coverage union bound is necessary and sufficient."""
    res = run_coverage_experiment(
        leduc_static_biased_opponent(),
        deltas=(0.1, 0.2),
        rounds=20,
        episodes_per_round=50,
        seeds=(42, 43, 44),
        out_dir=None,
    )
    for cell in res["cells"]:
        arms = cell["arms"]

        assert arms["spatial"]["meets_guarantee"]
        assert arms["time_uniform"]["meets_guarantee"]

        assert (
            arms["no_union"]["anytime_coverage"] < arms["spatial"]["anytime_coverage"]
        )


def test_coverage_no_union_degrades_with_delta():
    """Verify that coverage no union degrades with delta."""
    res = run_coverage_experiment(
        leduc_static_biased_opponent(),
        deltas=(0.05, 0.3),
        rounds=20,
        episodes_per_round=50,
        seeds=(42, 43, 44),
        out_dir=None,
    )
    low, high = res["cells"][0], res["cells"][1]
    assert (
        high["arms"]["no_union"]["anytime_coverage"]
        <= low["arms"]["no_union"]["anytime_coverage"]
    )


def test_finite_sample_gap_decays_and_is_bounded():
    """Verify that finite sample gap decays and is bounded."""
    res = run_finite_sample_gap(
        leduc_static_biased_opponent(),
        rounds=40,
        episodes_per_round=100,
        seeds=(42, 43, 44),
        out_dir=None,
    )

    assert res["scbr_gain"] > 0.1

    assert res["gap_decreased"]
    assert all(r["scbr_gap"] >= -1e-6 for r in res["series"])

    assert 0.0 < res["bound_constant"] < 2.0
    for r in res["series"]:
        assert r["scbr_gap"] <= res["bound_constant"] * r["radius"] + 1e-9


def test_finite_sample_radius_shrinks_over_rounds():
    """Verify that finite sample radius shrinks over rounds."""
    res = run_finite_sample_gap(
        leduc_static_biased_opponent(),
        rounds=40,
        episodes_per_round=100,
        seeds=(42, 43, 44),
        out_dir=None,
    )
    radii = [r["radius"] for r in res["series"]]
    assert radii[-1] < radii[0]
