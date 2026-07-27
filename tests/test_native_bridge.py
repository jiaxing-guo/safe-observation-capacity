"""Regression tests for test native bridge. See the corresponding implementation module and supplementary Reproducibility."""

import math

from safe_observation import native


def test_version_is_nonempty_str():
    """Verify that version is nonempty str."""
    assert isinstance(native.version(), str)
    assert native.version()


def test_kuhn_value_matches_known_value():
    """Verify that Kuhn value matches known value."""
    value, strategy = native.solve_kuhn(50_000)

    assert math.isclose(value, -1.0 / 18.0, abs_tol=5e-3)

    assert len(strategy) == 12
    for probs in strategy.values():
        assert len(probs) == 2
        assert math.isclose(sum(probs), 1.0, abs_tol=1e-9)


def test_kuhn_sequence_form_sizes():
    """Verify that Kuhn sequence form sizes."""
    n_seq1, n_info1, n_seq2, n_info2 = native.sequence_form_sizes("kuhn")
    assert (n_seq1, n_info1) == (13, 6)
    assert (n_seq2, n_info2) == (13, 6)


def test_leduc_blueprint_value_matches_literature():
    """Verify that Leduc blueprint value matches literature."""
    value, realization = native.blueprint_lp("leduc")

    assert math.isclose(value, -0.0856064240780, abs_tol=1e-6)
    assert len(realization) == 337


def test_leduc_sequence_form_sizes():
    """Verify that Leduc sequence form sizes."""
    n_seq1, n_info1, n_seq2, n_info2 = native.sequence_form_sizes("leduc")

    assert (n_seq1, n_info1) == (337, 144)
    assert (n_seq2, n_info2) == (337, 144)
