""

import pytest

from safe_observation.sequence_form import compile_kuhn, kuhn_sizes


@pytest.mark.parametrize("player", [0, 1])
def test_shapes(player):
    sf = compile_kuhn(player)
    assert sf.num_sequences == 13
    assert sf.num_infosets == 6
    assert sf.num_constraints == 7
    assert len(sf.e) == 7

    dense = sf.dense_e()
    assert len(dense) == 7
    assert all(len(row) == 13 for row in dense)


def test_empty_sequence_index_zero():
    sf = compile_kuhn(0)
    assert sf.sequences[0] == ""
    assert sf.seq_index[""] == 0


def test_rhs_is_root_indicator():
    sf = compile_kuhn(1)
    assert sf.e[0] == 1.0
    assert all(v == 0.0 for v in sf.e[1:])


@pytest.mark.parametrize("player", [0, 1])
def test_uniform_plan_satisfies_constraints(player):

    sf = compile_kuhn(player)
    x = sf.realization_from_behavior()
    assert abs(x[0] - 1.0) < 1e-12
    assert sf.constraint_residual(x) < 1e-12


def test_biased_plan_satisfies_constraints():
    sf = compile_kuhn(1)
    behavior = {info.label: [0.25, 0.75] for info in sf.info_sets}
    x = sf.realization_from_behavior(behavior)
    assert sf.constraint_residual(x) < 1e-12


def test_infeasible_vector_has_positive_residual():
    sf = compile_kuhn(0)
    x = list(sf.realization_from_behavior())
    x[1] += 0.5
    assert sf.constraint_residual(x) > 1e-6


def test_parent_links():
    sf = compile_kuhn(0)

    parent = sf.seq_index["0:>p"]
    info = next(i for i in sf.info_sets if i.label == "0:pb")
    assert info.parent_seq == parent


def test_invalid_player_raises():
    with pytest.raises(ValueError):
        compile_kuhn(2)


def test_kuhn_sizes_unchanged():
    sizes = kuhn_sizes()
    assert (sizes.num_sequences_p1, sizes.num_infosets_p1) == (13, 6)
    assert (sizes.num_sequences_p2, sizes.num_infosets_p2) == (13, 6)
