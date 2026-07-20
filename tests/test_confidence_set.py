""

import pytest

from safe_observation.confidence import (
    OpponentEvidenceStore,
    build_kuhn_confidence_set,
)
from safe_observation.sequence_form import compile_kuhn


def _opponent_plan(behavior_bet: float):
    ""
    sf = compile_kuhn(1)
    behavior = {info.label: [1.0 - behavior_bet, behavior_bet] for info in sf.info_sets}
    return sf.realization_from_behavior(behavior)


def test_empty_intervals_have_no_rows_and_contain_everything():
    cs = build_kuhn_confidence_set({})
    assert cs.nrows == 0
    assert cs.ncols == 13
    assert cs.contains(_opponent_plan(0.5))
    assert cs.contains(_opponent_plan(0.0))


def test_rhs_is_zero_and_shapes_consistent():
    sf = compile_kuhn(1)
    intervals = {info.label: [(0.3, 0.7), (0.3, 0.7)] for info in sf.info_sets}
    cs = build_kuhn_confidence_set(intervals)
    assert cs.nrows > 0
    assert len(cs.h) == cs.nrows
    assert all(v == 0.0 for v in cs.h)

    assert len(cs.g_entries) == 2 * cs.nrows


def test_confidence_set_contains_consistent_opponent():

    sf = compile_kuhn(1)
    intervals = {info.label: [(0.4, 0.6), (0.4, 0.6)] for info in sf.info_sets}
    cs = build_kuhn_confidence_set(intervals)
    assert cs.contains(_opponent_plan(0.5))
    assert cs.max_violation(_opponent_plan(0.5)) <= 1e-12


def test_confidence_set_excludes_inconsistent_opponent():

    sf = compile_kuhn(1)
    intervals = {info.label: [(0.4, 0.6), (0.4, 0.6)] for info in sf.info_sets}
    cs = build_kuhn_confidence_set(intervals)
    assert not cs.contains(_opponent_plan(0.0))
    assert cs.max_violation(_opponent_plan(0.0)) == pytest.approx(0.4)


def test_pipeline_from_evidence_brackets_true_opponent():

    sf = compile_kuhn(1)
    store = OpponentEvidenceStore.for_kuhn()

    for info in sf.info_sets:
        store.record(info.label, [700, 300])
    intervals = store.intervals(0.1, method="hoeffding")
    cs = build_kuhn_confidence_set(intervals)
    assert cs.contains(_opponent_plan(0.3))


def test_length_validation():
    cs = build_kuhn_confidence_set(
        {info.label: [(0.4, 0.6), (0.4, 0.6)] for info in compile_kuhn(1).info_sets}
    )
    with pytest.raises(ValueError):
        cs.max_violation([0.0] * 5)
