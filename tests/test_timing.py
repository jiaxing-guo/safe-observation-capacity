"""Regression tests for test timing. See the corresponding implementation module and supplementary Reproducibility."""

from safe_observation.experiments.online import (
    run_online_adaptation,
    run_online_adaptation_replicated,
)
from safe_observation.opponents import static_biased_opponent
from safe_observation.timing import StageTimer


def test_stage_timer_accumulates_seconds_and_calls():
    """Verify that stage timer accumulates seconds and calls."""
    timer = StageTimer()
    with timer.stage("work"):
        sum(range(1000))
    with timer.stage("work"):
        sum(range(1000))
    d = timer.as_dict()
    assert d["work"]["calls"] == 2
    assert d["work"]["seconds"] >= 0.0
    assert timer.total_seconds() == d["work"]["seconds"]


def test_stage_timer_add_and_merge():
    """Verify that stage timer add and merge."""
    a = StageTimer()
    a.add("solve", 1.5)
    b = StageTimer()
    b.add("solve", 0.5)
    b.add("build", 2.0)
    a.merge(b)
    d = a.as_dict()
    assert d["solve"]["seconds"] == 2.0
    assert d["solve"]["calls"] == 2
    assert d["build"]["seconds"] == 2.0
    assert d["build"]["calls"] == 1


def test_online_results_include_stage_timings():
    """Verify that online results include stage timings."""
    results = run_online_adaptation(
        static_biased_opponent(),
        rounds=5,
        episodes_per_round=20,
        seed=2026,
        out_dir=None,
    )
    timings = results["timings"]

    for stage in ("confidence_build", "robust_solve", "safety_verify", "simulate"):
        assert stage in timings
        assert timings[stage]["calls"] >= 1
        assert timings[stage]["seconds"] >= 0.0


def test_replicated_results_aggregate_timings_across_seeds():
    """Verify that replicated results aggregate timings across seeds."""
    results = run_online_adaptation_replicated(
        static_biased_opponent(),
        rounds=4,
        episodes_per_round=20,
        seeds=(2026, 2027),
        out_dir=None,
    )
    timings = results["timings"]
    assert "robust_solve" in timings

    assert timings["robust_solve"]["calls"] >= 8
