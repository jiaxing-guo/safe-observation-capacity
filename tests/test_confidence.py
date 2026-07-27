"""Regression tests for test confidence. See the corresponding implementation module and supplementary Reproducibility."""

import math
import random

import pytest

from safe_observation.confidence import (
    OpponentEvidenceStore,
    allocate_simultaneous_delta,
    empirical_bernstein_halfwidth,
    empirical_bernstein_interval,
    hoeffding_halfwidth,
    hoeffding_interval,
)


def test_simultaneous_delta_splits_only_across_active_families():
    """Verify the global guarantee is charged once across active families."""
    both = allocate_simultaneous_delta(0.1, public_rows=20, reveal_rows=30)
    assert both.public_delta == pytest.approx(0.05)
    assert both.reveal_delta == pytest.approx(0.05)
    assert both.public_row_delta == pytest.approx(0.05 / 20)
    assert both.reveal_row_delta == pytest.approx(0.05 / 30)
    assert both.allocated_delta == pytest.approx(0.1)

    public_only = allocate_simultaneous_delta(0.1, public_rows=20, reveal_rows=0)
    assert public_only.public_delta == pytest.approx(0.1)
    assert public_only.reveal_delta == 0.0
    assert public_only.reveal_row_delta is None


def test_halfwidth_matches_formula():
    """Verify that halfwidth matches formula."""
    n, delta = 100, 0.05
    expected = math.sqrt(math.log(2.0 / delta) / (2.0 * n))
    assert math.isclose(hoeffding_halfwidth(n, delta), expected)


def test_halfwidth_no_samples_is_maximal():
    """Verify that halfwidth no samples is maximal."""
    assert hoeffding_halfwidth(0, 0.05) == 1.0


def test_interval_contains_and_clips():
    """Verify that interval contains and clips."""
    lo, hi = hoeffding_interval(0.5, 100, 0.05)
    assert 0.0 <= lo < 0.5 < hi <= 1.0
    assert hoeffding_interval(0.0, 10, 0.05)[0] == 0.0
    assert hoeffding_interval(1.0, 10, 0.05)[1] == 1.0


def test_invalid_delta_raises():
    """Verify that invalid delta raises."""
    with pytest.raises(ValueError):
        hoeffding_halfwidth(10, 0.0)
    with pytest.raises(ValueError):
        hoeffding_halfwidth(10, 1.0)


def test_bernstein_matches_formula():
    """Verify that bernstein matches formula."""
    n, var, delta = 200, 0.21, 0.05
    ln = math.log(3.0 / delta)
    expected = math.sqrt(2.0 * var * ln / n) + 3.0 * ln / n
    assert math.isclose(empirical_bernstein_halfwidth(n, var, delta), expected)


def test_bernstein_tighter_than_hoeffding_for_low_variance():
    """Verify that bernstein tighter than hoeffding for low variance."""
    n, delta = 500, 0.05
    eb = empirical_bernstein_interval(0.02, n, delta)
    ho = hoeffding_interval(0.02, n, delta)
    assert (eb[1] - eb[0]) < (ho[1] - ho[0])


def test_bernstein_no_samples_is_maximal():
    """Verify that bernstein no samples is maximal."""
    assert empirical_bernstein_halfwidth(0, 0.25, 0.05) == 1.0


def test_evidence_store_counts_and_phat():
    """Verify that evidence store counts and phat."""
    store = OpponentEvidenceStore.for_kuhn()
    assert len(store.labels) == 6

    label = store.labels[0]
    assert store.visits(label) == 0
    assert store.p_hat(label) == (0.5, 0.5)

    store.update(label, 1, times=3)
    store.update(label, 0, times=1)
    assert store.counts(label) == (1, 3)
    assert store.visits(label) == 4
    assert store.p_hat(label) == (0.25, 0.75)


def test_evidence_store_record_validates_length():
    """Verify that evidence store record validates length."""
    store = OpponentEvidenceStore.for_kuhn()
    with pytest.raises(ValueError):
        store.record(store.labels[0], [1, 2, 3])


def test_public_count_view_hides_private_labels_and_counts_rows():
    """Verify public aggregation exposes only public histories."""
    store = OpponentEvidenceStore.for_kuhn()
    for label in store.labels:
        store.record(label, [1, 2])
    public = store.public_counts()
    assert set(public) == {label.split(":", 1)[1] for label in store.labels}
    assert all(isinstance(row, tuple) for row in public.values())
    assert store.num_public_pairs == sum(len(row) for row in public.values())


def test_union_bound_widens_intervals():
    """Verify that union bound widens intervals."""
    store = OpponentEvidenceStore.for_kuhn()
    for label in store.labels:
        store.record(label, [50, 50])
    wide = store.intervals(0.1, union_bound=True)
    narrow = store.intervals(0.1, union_bound=False)
    label = store.labels[0]
    assert (wide[label][0][1] - wide[label][0][0]) > (
        narrow[label][0][1] - narrow[label][0][0]
    )


def test_confidence_intervals_contain_true_opponent():
    """Verify that confidence intervals contain true opponent."""
    rng = random.Random(2026)
    true_p, n, delta, trials = 0.3, 200, 0.1, 2000
    covered = 0
    for _ in range(trials):
        bets = sum(1 for _ in range(n) if rng.random() < true_p)
        lo, hi = hoeffding_interval(bets / n, n, delta)
        if lo <= true_p <= hi:
            covered += 1
    assert covered / trials >= 1.0 - delta
