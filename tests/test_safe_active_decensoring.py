"""Contracts for the matched-budget Safe Active De-censoring controller."""

from importlib import util
import inspect
from pathlib import Path
import sys

import pytest


def _load_controller():
    path = (
        Path(__file__).parents[1]
        / "scripts"
        / "poker"
        / "run_safe_active_decensoring.py"
    )
    spec = util.spec_from_file_location("safe_active_decensoring_under_test", path)
    assert spec is not None and spec.loader is not None
    module = util.module_from_spec(spec)
    previous_argv = sys.argv
    sys.argv = [str(path), "holdem_tr_b2", "0.5", "1000", "1", "1", "600"]
    try:
        spec.loader.exec_module(module)
    finally:
        sys.argv = previous_argv
    return module


def test_sampling_plan_charges_one_total_budget():
    """Verify the configured 20/80 split and disjoint deterministic seeds."""
    controller = _load_controller()
    plan = controller.plan_sampling_batches(10_000, 2026)
    assert (plan.public_budget, plan.reveal_budget) == (2_000, 8_000)
    assert plan.public_budget + plan.reveal_budget == plan.total_budget
    assert (plan.public_seed, plan.reveal_seed) == (4052, 4053)


def test_checkpoint_filter_retries_failures_and_rejects_stale_protocols():
    """Verify resumes cannot silently mix incompatible experiment definitions."""
    controller = _load_controller()
    requested = [("sad", "opponent", 100, 1)]
    base = {
        "mode": "sad",
        "opponent": "opponent",
        "episodes": 100,
        "seed": 1,
        "pipeline_version": controller.PIPELINE_VERSION,
        "config_fingerprint": controller.CONFIG_FINGERPRINT,
        "public_pilot_fraction_config": controller.PUBLIC_PILOT_FRACTION,
    }
    records = [
        {**base, "certified": 0.1},
        {**base, "certified": 0.2},
        {**base, "lp_timeout": True},
        {**base, "pipeline_version": "legacy"},
    ]
    rows, ignored, retried = controller.select_checkpoint_rows(records, requested)
    assert len(rows) == 1
    assert rows[0]["certified"] == 0.2
    assert ignored == 1
    assert retried == 1


def test_solver_guard_only_handles_expected_empty_sets():
    """Verify unexpected native failures cannot masquerade as safe fallbacks."""
    controller = _load_controller()
    assert (
        controller._try_solve(
            lambda: (_ for _ in ()).throw(
                ValueError("linear-observation LP infeasible: confidence set is empty")
            )
        )
        is None
    )

    class UnexpectedFailure(BaseException):
        pass

    with pytest.raises(UnexpectedFailure):
        controller._try_solve(
            lambda: (_ for _ in ()).throw(UnexpectedFailure("solver crashed"))
        )


def test_simultaneous_screen_abstains_on_noise_and_routes_large_anomalies():
    """Verify the reference control is screened while a large public shift survives."""
    controller = _load_controller()
    quiet = controller.public_anomaly_weights(
        {"turn": [510, 490]},
        {"turn": [0.5, 0.5]},
        ["type|turn/river"],
        screen_delta=0.1,
    )
    shifted = controller.public_anomaly_weights(
        {"turn": [900, 100]},
        {"turn": [0.5, 0.5]},
        ["type|turn/river"],
        screen_delta=0.1,
    )
    assert quiet == {}
    assert shifted["type|turn/river"] > 0.3


def test_capture_uses_the_matched_public_cell():
    """Verify comparisons match both acquisition budget and random seed."""
    controller = _load_controller()
    public = {
        ("opponent", 100, 1): {"realized": 0.0},
        ("opponent", 100, 2): {"realized": 0.5},
    }
    oracle = {"opponent": {"realized": 1.0}}
    row = {
        "mode": "sad",
        "opponent": "opponent",
        "episodes": 100,
        "seed": 2,
        "realized": 0.75,
    }
    assert controller.matched_capture(row, public, oracle) == 0.5


def test_empirical_controller_has_no_population_fallback():
    """Verify truth appears only at the post-response evaluation boundary."""
    controller = _load_controller()
    source = inspect.getsource(controller.deploy_cell)
    assert "_population_public_intervals" not in source
    assert source.index("response_source =") < source.index("y_evaluation =")
    assert '"bp_realization"' in source
