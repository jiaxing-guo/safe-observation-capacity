""

import pytest

from safe_observation import native
from safe_observation.opponents import (
    Opponent,
    _holdem_decision_profile,
    _holdem_deep_fold_lines,
    _holdem_hole_class,
    _holdem_p2_actions,
    _holdem_shallow_fold_lines,
    best_response_value,
    holdem_ambiguous_fold_marginal_opponent,
    holdem_ambiguous_fold_strong_opponent,
    holdem_board_marginal_overfold_opponent,
    holdem_calling_station_opponent,
    holdem_censored_fold_opponent,
    holdem_equilibrium_opponent,
    holdem_fold_and_call_opponent,
    holdem_fold_mild_call_opponent,
    holdem_low_reach_leak_opponent,
    holdem_maniac_opponent,
    holdem_near_equilibrium_opponent,
    holdem_overfold_opponent,
    holdem_private_card_fold_opponent,
    holdem_public_fold_opponent,
    holdem_showdown_opponent_suite,
    holdem_structured_opponent_suite,
    leduc_always_fold_opponent,
    leduc_equilibrium_opponent,
    leduc_near_equilibrium_opponent,
    leduc_opponent_suite,
    leduc_private_state_leak_opponent,
    leduc_static_biased_opponent,
    leduc_trap_opponent,
    opponent_from_spec,
    opponent_suite,
)
from safe_observation.sequence_form import compile_holdem, compile_leduc

LEDUC_VALUE = -0.0856064240780
HOLDEM_VALUE = -0.042830296192800967


def _is_valid_behavior(opp: Opponent, player: int = 1) -> None:
    sf = compile_leduc(player) if opp.game == "leduc" else None

    if sf is not None:
        for info in sf.info_sets:
            dist = opp.behavior[info.label]
            assert len(dist) == len(info.children)
            assert all(p >= -1e-12 for p in dist)
            assert sum(dist) == pytest.approx(1.0)

    assert sf.constraint_residual(opp.realization()) < 1e-9


@pytest.mark.parametrize("name", sorted(leduc_opponent_suite()))
def test_leduc_opponents_are_well_formed(name):
    _is_valid_behavior(leduc_opponent_suite()[name])


def test_kuhn_suite_still_well_formed():

    for opp in opponent_suite().values():
        assert opp.game == "kuhn"
        from safe_observation.sequence_form import compile_kuhn

        assert compile_kuhn(1).constraint_residual(opp.realization()) < 1e-9


def test_equilibrium_opponent_is_essentially_unexploitable():

    expl = best_response_value(leduc_equilibrium_opponent())
    assert expl == pytest.approx(LEDUC_VALUE, abs=1e-6)


def test_static_biased_is_clearly_exploitable():
    expl = best_response_value(leduc_static_biased_opponent())
    assert expl > LEDUC_VALUE + 0.1


def test_exploitability_orders_eq_below_near_below_static():
    eq = best_response_value(leduc_equilibrium_opponent())
    near = best_response_value(leduc_near_equilibrium_opponent(eps=0.1))
    static = best_response_value(leduc_static_biased_opponent())
    assert eq <= near + 1e-9
    assert near < static


def test_near_equilibrium_exploitability_shrinks_with_eps():

    small = best_response_value(leduc_near_equilibrium_opponent(eps=0.02))
    large = best_response_value(leduc_near_equilibrium_opponent(eps=0.3))
    assert small < large


def test_near_equilibrium_is_within_eps_of_nash():
    eps = 0.1
    eq = leduc_equilibrium_opponent().behavior
    near = leduc_near_equilibrium_opponent(eps=eps).behavior
    for label, dist in near.items():
        for p, q in zip(dist, eq[label], strict=True):
            assert abs(p - q) <= eps + 1e-12


def test_private_state_leak_deviates_only_on_leak_rank():
    eq = leduc_equilibrium_opponent().behavior
    leak = leduc_private_state_leak_opponent(leak_rank="J", leak=0.6).behavior
    deviating_ranks = set()
    for label, dist in leak.items():
        if any(abs(p - q) > 1e-9 for p, q in zip(dist, eq[label], strict=True)):
            deviating_ranks.add(label[0])
    assert deviating_ranks == {"J"}


def test_trap_is_aggressive_with_king():
    trap = leduc_trap_opponent()
    actions_by_label = {
        info.label: [a for a, _ in info.children] for info in compile_leduc(1).info_sets
    }
    for label, dist in trap.behavior.items():
        if label[0] == "K":
            actions = actions_by_label[label]
            assert "f" not in actions or dist[actions.index("f")] == 0.0
            target = "r" if "r" in actions else "c"
            assert dist[actions.index(target)] == pytest.approx(1.0)


def test_always_fold_folds_when_possible():
    opp = leduc_always_fold_opponent()
    actions_by_label = {
        info.label: [a for a, _ in info.children] for info in compile_leduc(1).info_sets
    }
    for label, dist in opp.behavior.items():
        actions = actions_by_label[label]
        if "f" in actions:
            assert dist[actions.index("f")] == pytest.approx(1.0)


def test_opponent_from_spec_leduc():
    opp = opponent_from_spec({"game": "leduc", "type": "near_equilibrium", "eps": 0.05})
    assert opp.game == "leduc"
    assert opp.name == "near_equilibrium"


def test_opponent_from_spec_unknown_game():
    with pytest.raises(ValueError):
        opponent_from_spec({"game": "chess", "type": "static_biased"})


def test_opponent_from_spec_unknown_type_for_game():
    with pytest.raises(ValueError):
        opponent_from_spec({"game": "kuhn", "type": "near_equilibrium"})


def test_leduc_game_value_anchor():

    value, _ = native.blueprint_lp("leduc")
    assert value == pytest.approx(LEDUC_VALUE, abs=1e-6)


def _is_valid_holdem_behavior(opp: Opponent) -> None:
    sf = compile_holdem(1)
    for info in sf.info_sets:
        dist = opp.behavior[info.label]
        assert len(dist) == len(info.children)
        assert all(p >= -1e-12 for p in dist)
        assert sum(dist) == pytest.approx(1.0)
    assert sf.constraint_residual(opp.realization()) < 1e-9


@pytest.mark.parametrize("name", sorted(holdem_showdown_opponent_suite()))
def test_holdem_opponents_are_well_formed(name):
    opp = holdem_showdown_opponent_suite()[name]
    assert opp.game == "holdem"
    _is_valid_holdem_behavior(opp)


@pytest.mark.parametrize("name", sorted(holdem_structured_opponent_suite()))
def test_holdem_structured_opponents_are_well_formed(name):
    opp = holdem_structured_opponent_suite()[name]
    assert opp.game == "holdem"
    _is_valid_holdem_behavior(opp)


def test_holdem_game_value_anchor():
    value, _ = native.blueprint_lp("holdem")
    assert value == pytest.approx(HOLDEM_VALUE, abs=1e-6)


def test_holdem_equilibrium_is_essentially_unexploitable():
    expl = best_response_value(holdem_equilibrium_opponent())
    assert expl == pytest.approx(HOLDEM_VALUE, abs=1e-6)


def test_holdem_population_leaks_are_exploitable():

    eq = best_response_value(holdem_equilibrium_opponent())
    for factory in (
        holdem_overfold_opponent,
        holdem_calling_station_opponent,
        holdem_maniac_opponent,
    ):
        assert best_response_value(factory()) > eq + 0.1


def test_holdem_exploitability_orders_eq_below_near():
    eq = best_response_value(holdem_equilibrium_opponent())
    near = best_response_value(holdem_near_equilibrium_opponent(eps=0.1))
    assert eq <= near + 1e-9


def test_holdem_censored_fold_deviates_only_on_shallow_lines():

    eq = holdem_equilibrium_opponent().behavior
    leak = holdem_censored_fold_opponent().behavior
    shallow = _holdem_shallow_fold_lines()
    for label, dist in leak.items():
        hist = label.split("|", 1)[1]
        deviates = any(abs(p - q) > 1e-9 for p, q in zip(dist, eq[label], strict=True))
        if deviates:
            assert hist in shallow


def test_holdem_low_reach_leak_deviates_only_on_deep_lines():
    eq = holdem_equilibrium_opponent().behavior
    leak = holdem_low_reach_leak_opponent().behavior
    deep = _holdem_deep_fold_lines()
    for label, dist in leak.items():
        hist = label.split("|", 1)[1]
        deviates = any(abs(p - q) > 1e-9 for p, q in zip(dist, eq[label], strict=True))
        if deviates:
            assert hist in deep


def test_holdem_censored_fold_is_highly_exploitable():

    assert best_response_value(holdem_censored_fold_opponent()) > HOLDEM_VALUE + 0.5


def test_holdem_shallow_and_deep_lines_are_disjoint_and_nonempty():
    shallow = _holdem_shallow_fold_lines()
    deep = _holdem_deep_fold_lines()
    assert shallow and deep
    assert shallow.isdisjoint(deep)
    assert all(len(h) == 1 for h in shallow)
    assert all(len(h) >= 3 for h in deep)


def test_holdem_mixed_fold_call_overfolds_on_shallow_lines():

    eq = holdem_equilibrium_opponent().behavior
    actions = _holdem_p2_actions()
    shallow = _holdem_shallow_fold_lines()
    for factory in (holdem_fold_and_call_opponent, holdem_fold_mild_call_opponent):
        behavior = factory().behavior
        strictly_more = False
        for label, dist in behavior.items():
            hist = label.split("|", 1)[1]
            if hist in shallow and "f" in actions[label]:
                fi = actions[label].index("f")
                assert dist[fi] >= eq[label][fi] - 1e-9
                if dist[fi] > eq[label][fi] + 0.05:
                    strictly_more = True
        assert strictly_more


def test_holdem_fold_and_call_overcalls_off_the_shallow_lines():

    eq = holdem_equilibrium_opponent().behavior
    actions = _holdem_p2_actions()
    shallow = _holdem_shallow_fold_lines()
    behavior = holdem_fold_and_call_opponent().behavior
    overcalls = False
    for label, dist in behavior.items():
        hist = label.split("|", 1)[1]
        if hist not in shallow and "c" in actions[label]:
            ci = actions[label].index("c")
            if dist[ci] > eq[label][ci] + 1e-6:
                overcalls = True
                break
    assert overcalls


def test_holdem_private_card_fold_deviates_only_on_no_ace_combos():
    eq = holdem_equilibrium_opponent().behavior
    behavior = holdem_private_card_fold_opponent().behavior
    for label, dist in behavior.items():
        hole = label.split("|", 1)[0]
        deviates = any(abs(p - q) > 1e-9 for p, q in zip(dist, eq[label], strict=True))
        if deviates:
            assert "A" not in hole


@pytest.mark.parametrize(
    "game", ["holdem", "holdem_paired", "holdem_dry", "holdem_wet", "holdem_low"]
)
def test_holdem_structured_marginal_overfold_targets_board_aware_headroom(game):
    eq = holdem_equilibrium_opponent(game).behavior
    behavior = holdem_board_marginal_overfold_opponent(game=game).behavior
    actions = _holdem_p2_actions(game)
    changed = []
    for label, dist in behavior.items():
        if "f" not in actions[label]:
            continue
        fi = actions[label].index("f")
        if dist[fi] > eq[label][fi] + 1e-9:
            profile = _holdem_decision_profile(label, game)
            changed.append(profile)
            assert profile.hand_class == "medium"
            assert profile.fold_headroom > 0.05
    assert changed, f"no board-aware marginal fold targets found for {game}"


def test_holdem_structured_ambiguous_twins_move_different_private_classes():
    eq = holdem_equilibrium_opponent().behavior
    marginal = holdem_ambiguous_fold_marginal_opponent().behavior
    strong = holdem_ambiguous_fold_strong_opponent().behavior
    actions = _holdem_p2_actions()
    marginal_classes = set()
    strong_classes = set()
    for label, eq_dist in eq.items():
        if "f" not in actions[label]:
            continue
        if any(
            abs(p - q) > 1e-9 for p, q in zip(marginal[label], eq_dist, strict=True)
        ):
            marginal_classes.add(_holdem_hole_class(label))
        if any(abs(p - q) > 1e-9 for p, q in zip(strong[label], eq_dist, strict=True)):
            strong_classes.add(_holdem_hole_class(label))
    assert marginal_classes == {"medium"}
    assert strong_classes == {"strong"}


def test_holdem_structured_public_fold_is_public_homogeneous_control():
    eq = holdem_equilibrium_opponent().behavior
    behavior = holdem_public_fold_opponent().behavior
    actions = _holdem_p2_actions()
    shallow = _holdem_shallow_fold_lines()
    changed_classes = set()
    for label, dist in behavior.items():
        hist = label.split("|", 1)[1]
        if hist in shallow and "f" in actions[label]:
            fi = actions[label].index("f")
            if dist[fi] > eq[label][fi] + 1e-9:
                changed_classes.add(_holdem_hole_class(label))

    assert changed_classes == {"medium", "strong"}


def test_opponent_from_spec_holdem_structured_private_structured():
    opp = opponent_from_spec({"game": "holdem", "type": "bluffcatcher_station"})
    assert opp.name == "bluffcatcher_station"
    _is_valid_holdem_behavior(opp)


def test_holdem_mixed_and_limitation_opponents_are_exploitable():
    eq = best_response_value(holdem_equilibrium_opponent())
    for factory in (
        holdem_fold_and_call_opponent,
        holdem_fold_mild_call_opponent,
        holdem_private_card_fold_opponent,
    ):
        assert best_response_value(factory()) > eq + 0.1


def test_opponent_from_spec_holdem():
    opp = opponent_from_spec({"game": "holdem", "type": "censored_fold", "leak": 0.8})
    assert opp.game == "holdem"
    assert opp.name == "censored_fold"
