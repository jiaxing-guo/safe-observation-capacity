""

import math
import random

import pytest

from safe_observation.confidence import (
    OpponentEvidenceStore,
    empirical_bernstein_halfwidth,
    empirical_bernstein_interval,
    hoeffding_halfwidth,
    hoeffding_interval,
)


def test_halfwidth_matches_formula():
    n, delta = 100, 0.05
    expected = math.sqrt(math.log(2.0 / delta) / (2.0 * n))
    assert math.isclose(hoeffding_halfwidth(n, delta), expected)


def test_halfwidth_no_samples_is_maximal():
    assert hoeffding_halfwidth(0, 0.05) == 1.0


def test_interval_contains_and_clips():
    lo, hi = hoeffding_interval(0.5, 100, 0.05)
    assert 0.0 <= lo < 0.5 < hi <= 1.0
    assert hoeffding_interval(0.0, 10, 0.05)[0] == 0.0
    assert hoeffding_interval(1.0, 10, 0.05)[1] == 1.0


def test_invalid_delta_raises():
    with pytest.raises(ValueError):
        hoeffding_halfwidth(10, 0.0)
    with pytest.raises(ValueError):
        hoeffding_halfwidth(10, 1.0)


def test_bernstein_matches_formula():
    n, var, delta = 200, 0.21, 0.05
    ln = math.log(3.0 / delta)
    expected = math.sqrt(2.0 * var * ln / n) + 3.0 * ln / n
    assert math.isclose(empirical_bernstein_halfwidth(n, var, delta), expected)


def test_bernstein_tighter_than_hoeffding_for_low_variance():

    n, delta = 500, 0.05
    eb = empirical_bernstein_interval(0.02, n, delta)
    ho = hoeffding_interval(0.02, n, delta)
    assert (eb[1] - eb[0]) < (ho[1] - ho[0])


def test_bernstein_no_samples_is_maximal():
    assert empirical_bernstein_halfwidth(0, 0.25, 0.05) == 1.0


def test_evidence_store_counts_and_phat():
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
    store = OpponentEvidenceStore.for_kuhn()
    with pytest.raises(ValueError):
        store.record(store.labels[0], [1, 2, 3])


def test_union_bound_widens_intervals():
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

    rng = random.Random(2026)
    true_p, n, delta, trials = 0.3, 200, 0.1, 2000
    covered = 0
    for _ in range(trials):
        bets = sum(1 for _ in range(n) if rng.random() < true_p)
        lo, hi = hoeffding_interval(bets / n, n, delta)
        if lo <= true_p <= hi:
            covered += 1
    assert covered / trials >= 1.0 - delta
