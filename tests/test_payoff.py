""

import pytest

from safe_observation.payoff import build_kuhn
from safe_observation.sequence_form import compile_kuhn


def test_shapes_and_nnz():
    a = build_kuhn()
    assert a.nrows == 13
    assert a.ncols == 13

    assert a.nnz == 30


def test_known_terminal_entries():
    a = build_kuhn()
    sf0 = compile_kuhn(0)
    sf1 = compile_kuhn(1)
    dense = a.dense()

    r = sf0.seq_index["2:>b"]
    c = sf1.seq_index["0:b>b"]
    assert dense[r][c] == pytest.approx(2.0 / 6.0)

    r = sf0.seq_index["0:>b"]
    c = sf1.seq_index["2:b>b"]
    assert dense[r][c] == pytest.approx(-2.0 / 6.0)

    r = sf0.seq_index["2:>p"]
    c = sf1.seq_index["0:p>p"]
    assert dense[r][c] == pytest.approx(1.0 / 6.0)


def test_oracle_consistency():
    a = build_kuhn()
    sf0 = compile_kuhn(0)
    sf1 = compile_kuhn(1)
    x = sf0.realization_from_behavior(
        {info.label: [0.3, 0.7] for info in sf0.info_sets}
    )
    y = sf1.realization_from_behavior(
        {info.label: [0.6, 0.4] for info in sf1.info_sets}
    )

    value = a.bilinear(x, y)
    ay = a.matvec_a_y(y)
    assert value == pytest.approx(sum(xi * ai for xi, ai in zip(x, ay, strict=True)))
    atx = a.matvec_at_x(x)
    assert value == pytest.approx(sum(ai * yi for ai, yi in zip(atx, y, strict=True)))


def test_length_validation():
    a = build_kuhn()
    with pytest.raises(ValueError):
        a.bilinear([0.0] * 5, [0.0] * 13)
    with pytest.raises(ValueError):
        a.matvec_a_y([0.0] * 12)
    with pytest.raises(ValueError):
        a.matvec_at_x([0.0] * 12)
