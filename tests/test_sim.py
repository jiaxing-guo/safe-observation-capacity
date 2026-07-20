""

import pytest

from safe_observation import native
from safe_observation.sequence_form import compile_kuhn


def _uniform(player: int) -> dict[str, list[float]]:
    sf = compile_kuhn(player)
    return {info.label: [0.5, 0.5] for info in sf.info_sets}


def test_deterministic_for_fixed_seed():
    x1, y2 = _uniform(0), _uniform(1)
    a = native.simulate("kuhn", x1, y2, 1000, 2026)
    b = native.simulate("kuhn", x1, y2, 1000, 2026)
    assert a == b


def test_different_seeds_differ():
    x1, y2 = _uniform(0), _uniform(1)
    a = native.simulate("kuhn", x1, y2, 1000, 2026)
    b = native.simulate("kuhn", x1, y2, 1000, 2027)
    assert a[0] != b[0]


def test_counts_are_length_two_and_bounded():
    x1, y2 = _uniform(0), _uniform(1)
    total_payoff, counts = native.simulate("kuhn", x1, y2, 5000, 2026)
    visits = 0
    for label, c in counts.items():
        assert label in {f"{card}:{h}" for card in range(3) for h in ("p", "b")}
        assert len(c) == 2
        visits += c[0] + c[1]
    assert 0 < visits <= 5000
    assert -2.0 <= total_payoff / 5000 <= 2.0


def test_always_pass_opponent_never_bets():
    x1 = _uniform(0)
    sf1 = compile_kuhn(1)
    y2 = {info.label: [1.0, 0.0] for info in sf1.info_sets}
    _payoff, counts = native.simulate("kuhn", x1, y2, 3000, 2026)
    assert all(c[1] == 0 for c in counts.values())


def test_empirical_frequency_approaches_true_bias():

    x1 = _uniform(0)
    sf1 = compile_kuhn(1)
    y2 = {info.label: [0.7, 0.3] for info in sf1.info_sets}
    _payoff, counts = native.simulate("kuhn", x1, y2, 200_000, 2026)
    passes = sum(c[0] for c in counts.values())
    bets = sum(c[1] for c in counts.values())
    bet_freq = bets / (passes + bets)
    assert bet_freq == pytest.approx(0.3, abs=0.02)
