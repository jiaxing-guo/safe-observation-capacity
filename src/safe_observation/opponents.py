"""Opponents primitives for safe-observation experiments. See Safe Active De-censoring, Experiments, and supplementary Game Instances and Experimental Setup."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import cache
from typing import Any

from . import native
from .sequence_form import (
    compile as compile_game,
    compile_goofspiel,
    compile_kuhn,
    compile_leduc,
)

_KING = 2


@dataclass(frozen=True)
class Opponent:
    """Implement the opponent for the opponents workflow."""

    name: str
    behavior: dict[str, list[float]]
    game: str = "kuhn"

    def realization(self) -> tuple[float, ...]:
        """Return the opponent policy as a sequence-form realization plan."""
        return compile_game(self.game, 1).realization_from_behavior(self.behavior)


def _kuhn_p2_labels() -> list[str]:
    """Compute Kuhn player-two labels for the opponents workflow."""
    return [info.label for info in compile_kuhn(1).info_sets]


def static_biased_opponent(
    bet_prob: float = 0.1, name: str = "static_biased"
) -> Opponent:
    """Construct the static biased opponent policy."""
    if not 0.0 <= bet_prob <= 1.0:
        raise ValueError("bet_prob must be in [0, 1]")
    behavior = {label: [1.0 - bet_prob, bet_prob] for label in _kuhn_p2_labels()}
    return Opponent(name=name, behavior=behavior, game="kuhn")


def always_fold_opponent() -> Opponent:
    """Construct the always fold opponent policy."""
    return static_biased_opponent(bet_prob=0.0, name="always_fold")


@cache
def equilibrium_opponent(iterations: int = 200_000) -> Opponent:
    """Construct the equilibrium opponent policy."""
    _value, strategy = native.solve_kuhn(iterations)
    behavior: dict[str, list[float]] = {}
    for label in _kuhn_p2_labels():
        card, history = label.split(":")
        behavior[label] = list(strategy[card + history])
    return Opponent(name="equilibrium", behavior=behavior, game="kuhn")


def trap_opponent(weak_bet_prob: float = 0.15) -> Opponent:
    """Construct the trap opponent policy."""
    behavior: dict[str, list[float]] = {}
    for label in _kuhn_p2_labels():
        card = int(label.split(":")[0])
        if card == _KING:
            behavior[label] = [0.0, 1.0]
        else:
            behavior[label] = [1.0 - weak_bet_prob, weak_bet_prob]
    return Opponent(name="trap", behavior=behavior, game="kuhn")


def _normalize(weights: list[float]) -> list[float]:
    """Normalize weights into a probability distribution."""
    total = sum(weights)
    if total <= 0.0:
        n = len(weights)
        return [1.0 / n] * n
    return [w / total for w in weights]


@cache
def _leduc_p2_actions() -> dict[str, tuple[str, ...]]:
    """Compute Leduc player-two actions for the opponents workflow."""
    return {
        info.label: tuple(a for a, _ in info.children)
        for info in compile_leduc(1).info_sets
    }


def _leduc_behavior(
    rule: Callable[[str, tuple[str, ...]], list[float]],
) -> dict[str, list[float]]:
    """Compute Leduc behavior for the opponents workflow."""
    return {
        label: rule(label, actions) for label, actions in _leduc_p2_actions().items()
    }


def _passive_dist(
    actions: tuple[str, ...], raise_w: float, fold_w: float, call_w: float = 1.0
) -> list[float]:
    """Compute passive dist for the opponents workflow."""
    weight = {"c": call_w, "r": raise_w, "f": fold_w}
    return _normalize([weight[a] for a in actions])


def _fold_else_check(actions: tuple[str, ...]) -> list[float]:
    """Compute fold else check for the opponents workflow."""
    target = "f" if "f" in actions else "c"
    return [1.0 if a == target else 0.0 for a in actions]


def leduc_static_biased_opponent(
    raise_weight: float = 0.1, fold_weight: float = 0.5, name: str = "static_biased"
) -> Opponent:
    """Construct the Leduc static biased opponent policy."""
    behavior = _leduc_behavior(
        lambda _label, actions: _passive_dist(actions, raise_weight, fold_weight)
    )
    return Opponent(name=name, behavior=behavior, game="leduc")


def leduc_always_fold_opponent() -> Opponent:
    """Construct the Leduc always fold opponent policy."""
    return Opponent(
        name="always_fold",
        behavior=_leduc_behavior(lambda _label, actions: _fold_else_check(actions)),
        game="leduc",
    )


@cache
def leduc_equilibrium_opponent() -> Opponent:
    """Construct the Leduc equilibrium opponent policy."""
    realization = native.blueprint_realization("leduc", 1)
    behavior = compile_leduc(1).behavior_from_realization(realization)
    return Opponent(name="equilibrium", behavior=behavior, game="leduc")


def leduc_trap_opponent(
    raise_weight: float = 0.1, fold_weight: float = 0.5
) -> Opponent:
    """Construct the Leduc trap opponent policy."""

    def rule(label: str, actions: tuple[str, ...]) -> list[float]:
        """Construct the action distribution for one information set."""
        if label[0] == "K":
            target = "r" if "r" in actions else "c"
            return [1.0 if a == target else 0.0 for a in actions]
        return _passive_dist(actions, raise_weight, fold_weight)

    return Opponent(name="trap", behavior=_leduc_behavior(rule), game="leduc")


def leduc_near_equilibrium_opponent(eps: float = 0.1) -> Opponent:
    """Construct the Leduc near equilibrium opponent policy."""
    if not 0.0 <= eps <= 1.0:
        raise ValueError("eps must be in [0, 1]")
    eq = leduc_equilibrium_opponent().behavior
    behavior: dict[str, list[float]] = {}
    for label, dist in eq.items():
        n = len(dist)
        behavior[label] = [(1.0 - eps) * p + eps / n for p in dist]
    return Opponent(name="near_equilibrium", behavior=behavior, game="leduc")


def leduc_private_state_leak_opponent(
    leak_rank: str = "J", leak: float = 0.6
) -> Opponent:
    """Construct the Leduc private state leak opponent policy."""
    if leak_rank not in ("J", "Q", "K"):
        raise ValueError("leak_rank must be one of 'J', 'Q', 'K'")
    eq = leduc_equilibrium_opponent().behavior
    actions_by_label = _leduc_p2_actions()
    behavior: dict[str, list[float]] = {}
    for label, dist in eq.items():
        if label[0] == leak_rank:
            biased = _fold_else_check(actions_by_label[label])
            behavior[label] = [
                (1.0 - leak) * p + leak * q for p, q in zip(dist, biased, strict=True)
            ]
        else:
            behavior[label] = list(dist)
    return Opponent(name="private_state_leak", behavior=behavior, game="leduc")


def _leduc_cap_facing_labels() -> list[str]:
    """Compute Leduc cap facing labels."""
    return [
        label for label, actions in _leduc_p2_actions().items() if actions == ("f", "c")
    ]


def leduc_low_reach_leak_opponent(leak: float = 0.9) -> Opponent:
    """Construct the Leduc low reach leak opponent policy."""
    if not 0.0 <= leak <= 1.0:
        raise ValueError("leak must be in [0, 1]")
    eq = leduc_equilibrium_opponent().behavior
    cap_facing = set(_leduc_cap_facing_labels())
    behavior: dict[str, list[float]] = {}
    for label, dist in eq.items():
        if label in cap_facing:
            behavior[label] = [(1.0 - leak) * dist[0] + leak, (1.0 - leak) * dist[1]]
        else:
            behavior[label] = list(dist)
    return Opponent(name="low_reach_leak", behavior=behavior, game="leduc")


@cache
def _goofspiel_p2_action_counts() -> dict[str, int]:
    """Compute goofspiel player-two action counts."""
    return {info.label: len(info.children) for info in compile_goofspiel(1).info_sets}


def goofspiel_lowball_opponent() -> Opponent:
    """Construct the goofspiel lowball opponent policy."""

    behavior = {
        label: [1.0] + [0.0] * (k - 1)
        for label, k in _goofspiel_p2_action_counts().items()
    }
    return Opponent(name="lowball", behavior=behavior, game="goofspiel")


def goofspiel_highball_opponent() -> Opponent:
    """Construct the goofspiel highball opponent policy."""
    behavior = {
        label: [0.0] * (k - 1) + [1.0]
        for label, k in _goofspiel_p2_action_counts().items()
    }
    return Opponent(name="highball", behavior=behavior, game="goofspiel")


def goofspiel_uniform_opponent() -> Opponent:
    """Construct the goofspiel uniform opponent policy."""
    behavior = {
        label: [1.0 / k] * k for label, k in _goofspiel_p2_action_counts().items()
    }
    return Opponent(name="uniform", behavior=behavior, game="goofspiel")


@cache
def goofspiel_equilibrium_opponent() -> Opponent:
    """Construct the goofspiel equilibrium opponent policy."""
    realization = native.blueprint_realization("goofspiel", 1)
    behavior = compile_goofspiel(1).behavior_from_realization(realization)
    return Opponent(name="equilibrium", behavior=behavior, game="goofspiel")


@cache
def _holdem_p2_actions(game: str = "holdem") -> dict[str, tuple[str, ...]]:
    """Compute holdem player-two actions for the opponents workflow."""
    return {
        info.label: tuple(a for a, _ in info.children)
        for info in compile_game(game, 1).info_sets
    }


@cache
def holdem_equilibrium_opponent(game: str = "holdem") -> Opponent:
    """Construct the holdem equilibrium opponent policy."""
    realization = native.blueprint_realization(game, 1)
    behavior = compile_game(game, 1).behavior_from_realization(realization)
    return Opponent(name="equilibrium", behavior=behavior, game=game)


def holdem_near_equilibrium_opponent(
    eps: float = 0.1, game: str = "holdem"
) -> Opponent:
    """Construct the holdem near equilibrium opponent policy."""
    if not 0.0 <= eps <= 1.0:
        raise ValueError("eps must be in [0, 1]")
    eq = holdem_equilibrium_opponent(game).behavior
    behavior: dict[str, list[float]] = {}
    for label, dist in eq.items():
        n = len(dist)
        behavior[label] = [(1.0 - eps) * p + eps / n for p in dist]
    return Opponent(name="near_equilibrium", behavior=behavior, game=game)


def _holdem_perturb_toward(
    target: str, weight: float, game: str = "holdem"
) -> dict[str, list[float]]:
    """Compute holdem perturb toward for the opponents workflow."""
    if not 0.0 <= weight <= 1.0:
        raise ValueError("weight must be in [0, 1]")
    eq = holdem_equilibrium_opponent(game).behavior
    actions = _holdem_p2_actions(game)
    behavior: dict[str, list[float]] = {}
    for label, dist in eq.items():
        acts = actions[label]
        if target in acts:
            behavior[label] = [
                (1.0 - weight) * p + (weight if a == target else 0.0)
                for a, p in zip(acts, dist, strict=True)
            ]
        else:
            behavior[label] = list(dist)
    return behavior


def holdem_overfold_opponent(leak: float = 0.5, game: str = "holdem") -> Opponent:
    """Construct the holdem overfold opponent policy."""
    return Opponent(
        name="overfold", behavior=_holdem_perturb_toward("f", leak, game), game=game
    )


def holdem_calling_station_opponent(
    leak: float = 0.6, game: str = "holdem"
) -> Opponent:
    """Construct the holdem calling station opponent policy."""
    return Opponent(
        name="calling_station",
        behavior=_holdem_perturb_toward("c", leak, game),
        game=game,
    )


def holdem_maniac_opponent(leak: float = 0.4, game: str = "holdem") -> Opponent:
    """Construct the holdem maniac opponent policy."""
    return Opponent(
        name="maniac", behavior=_holdem_perturb_toward("p", leak, game), game=game
    )


def _holdem_deep_fold_lines(game: str = "holdem") -> set[str]:
    """Compute holdem deep fold lines."""
    lines: set[str] = set()
    for info in compile_game(game, 1).info_sets:
        hist = info.label.split("|", 1)[1]
        acts = tuple(a for a, _ in info.children)
        if "f" in acts and len(hist) >= 3:
            lines.add(hist)
    return lines


def holdem_low_reach_leak_opponent(leak: float = 0.9, game: str = "holdem") -> Opponent:
    """Construct the holdem low reach leak opponent policy."""
    if not 0.0 <= leak <= 1.0:
        raise ValueError("leak must be in [0, 1]")
    eq = holdem_equilibrium_opponent(game).behavior
    actions = _holdem_p2_actions(game)
    deep = _holdem_deep_fold_lines(game)
    behavior: dict[str, list[float]] = {}
    for label, dist in eq.items():
        hist = label.split("|", 1)[1]
        acts = actions[label]
        if hist in deep and "f" in acts:
            behavior[label] = [
                (1.0 - leak) * p + (leak if a == "f" else 0.0)
                for a, p in zip(acts, dist, strict=True)
            ]
        else:
            behavior[label] = list(dist)
    return Opponent(name="low_reach_leak", behavior=behavior, game=game)


def _holdem_shallow_fold_lines(game: str = "holdem") -> set[str]:
    """Compute holdem shallow fold lines."""
    lines: set[str] = set()
    for info in compile_game(game, 1).info_sets:
        hist = info.label.split("|", 1)[1]
        acts = tuple(a for a, _ in info.children)
        if "f" in acts and len(hist) == 1:
            lines.add(hist)
    return lines


def holdem_censored_fold_opponent(leak: float = 0.85, game: str = "holdem") -> Opponent:
    """Construct the holdem censored fold opponent policy."""
    if not 0.0 <= leak <= 1.0:
        raise ValueError("leak must be in [0, 1]")
    eq = holdem_equilibrium_opponent(game).behavior
    actions = _holdem_p2_actions(game)
    shallow = _holdem_shallow_fold_lines(game)
    behavior: dict[str, list[float]] = {}
    for label, dist in eq.items():
        hist = label.split("|", 1)[1]
        acts = actions[label]
        if hist in shallow and "f" in acts:
            behavior[label] = [
                (1.0 - leak) * p + (leak if a == "f" else 0.0)
                for a, p in zip(acts, dist, strict=True)
            ]
        else:
            behavior[label] = list(dist)
    return Opponent(name="censored_fold", behavior=behavior, game=game)


def _holdem_perturb_lines(
    behavior: dict[str, list[float]],
    target: str,
    weight: float,
    hist_pred: Callable[[str], bool],
    game: str = "holdem",
) -> dict[str, list[float]]:
    """Compute holdem perturb lines for the opponents workflow."""
    if not 0.0 <= weight <= 1.0:
        raise ValueError("weight must be in [0, 1]")
    actions = _holdem_p2_actions(game)
    out: dict[str, list[float]] = {}
    for label, dist in behavior.items():
        hist = label.split("|", 1)[1]
        acts = actions[label]
        if hist_pred(hist) and target in acts:
            out[label] = [
                (1.0 - weight) * p + (weight if a == target else 0.0)
                for a, p in zip(acts, dist, strict=True)
            ]
        else:
            out[label] = list(dist)
    return out


def _holdem_fold_and_call_behavior(
    fold_leak: float, call_leak: float, game: str = "holdem"
) -> dict[str, list[float]]:
    """Compute holdem fold and call behavior."""
    eq = holdem_equilibrium_opponent(game).behavior
    shallow = _holdem_shallow_fold_lines(game)

    base = _holdem_perturb_lines(eq, "c", call_leak, lambda h: h not in shallow, game)

    return _holdem_perturb_lines(base, "f", fold_leak, lambda h: h in shallow, game)


def holdem_fold_and_call_opponent(
    fold_leak: float = 0.6, call_leak: float = 0.6, game: str = "holdem"
) -> Opponent:
    """Construct the holdem fold and call opponent policy."""
    return Opponent(
        name="fold_and_call",
        behavior=_holdem_fold_and_call_behavior(fold_leak, call_leak, game),
        game=game,
    )


def holdem_fold_mild_call_opponent(
    fold_leak: float = 0.7, call_leak: float = 0.2, game: str = "holdem"
) -> Opponent:
    """Construct the holdem fold mild call opponent policy."""
    return Opponent(
        name="fold_mild_call",
        behavior=_holdem_fold_and_call_behavior(fold_leak, call_leak, game),
        game=game,
    )


def holdem_private_card_fold_opponent(
    leak: float = 0.7, game: str = "holdem"
) -> Opponent:
    """Construct the holdem private card fold opponent policy."""
    if not 0.0 <= leak <= 1.0:
        raise ValueError("leak must be in [0, 1]")
    eq = holdem_equilibrium_opponent(game).behavior
    actions = _holdem_p2_actions(game)
    behavior: dict[str, list[float]] = {}
    for label, dist in eq.items():
        hole = label.split("|", 1)[0]
        acts = actions[label]
        if "A" not in hole and "f" in acts:
            behavior[label] = [
                (1.0 - leak) * p + (leak if a == "f" else 0.0)
                for a, p in zip(acts, dist, strict=True)
            ]
        else:
            behavior[label] = list(dist)
    return Opponent(name="private_card_fold", behavior=behavior, game=game)


_RANK_TO_INT = {rank: idx for idx, rank in enumerate("23456789TJQKA")}


def _card_from_str(card: str) -> tuple[int, str]:
    """Compute card from str for the opponents workflow."""
    return _RANK_TO_INT[card[0]], card[1]


def _holdem_board_strings(game: str = "holdem") -> tuple[str, ...]:
    """Compute holdem board strings for the opponents workflow."""
    variant = game.removeprefix("holdem_") if game.startswith("holdem_") else ""
    boards = {
        "": ("As", "Ks", "Qd", "Jc", "9h"),
        "paired": ("Ks", "Kh", "Qd", "7c", "2s"),
        "monotone": ("As", "Ts", "7s", "4s", "2s"),
        "dry": ("Kd", "8c", "5h", "3s", "2c"),
        "wet": ("Jh", "Th", "9c", "8d", "4s"),
        "low": ("8d", "6c", "4h", "3s", "2c"),
    }
    return boards.get(variant, boards[""])


def _straight_high(ranks: set[int]) -> int | None:
    """Compute straight high for the opponents workflow."""
    work = set(ranks)
    if 12 in work:
        work.add(-1)
    for high in range(12, 2, -1):
        if all((high - offset) in work for offset in range(5)):
            return high
    return None


def _holdem_7card_rank(cards: tuple[tuple[int, str], ...]) -> tuple[int, ...]:
    """Compute holdem 7card rank for the opponents workflow."""
    ranks = sorted((rank for rank, _suit in cards), reverse=True)
    counts = {rank: ranks.count(rank) for rank in set(ranks)}
    by_suit: dict[str, list[int]] = {}
    for rank, suit in cards:
        by_suit.setdefault(suit, []).append(rank)

    for suited_ranks in by_suit.values():
        if len(suited_ranks) >= 5:
            high = _straight_high(set(suited_ranks))
            if high is not None:
                return (8, high)

    quads = sorted((rank for rank, count in counts.items() if count == 4), reverse=True)
    if quads:
        kicker = max(rank for rank in ranks if rank != quads[0])
        return (7, quads[0], kicker)

    trips = sorted((rank for rank, count in counts.items() if count == 3), reverse=True)
    pairs = sorted((rank for rank, count in counts.items() if count == 2), reverse=True)
    if trips and (pairs or len(trips) >= 2):
        full_pair = pairs[0] if pairs else trips[1]
        return (6, trips[0], full_pair)

    flushes = [
        sorted(suited, reverse=True)[:5]
        for suited in by_suit.values()
        if len(suited) >= 5
    ]
    if flushes:
        return (5, *max(flushes))

    straight = _straight_high(set(ranks))
    if straight is not None:
        return (4, straight)

    if trips:
        kickers = [rank for rank in ranks if rank != trips[0]][:2]
        return (3, trips[0], *kickers)
    if len(pairs) >= 2:
        kicker = max(rank for rank in ranks if rank not in pairs[:2])
        return (2, pairs[0], pairs[1], kicker)
    if pairs:
        kickers = [rank for rank in ranks if rank != pairs[0]][:3]
        return (1, pairs[0], *kickers)
    return (0, *ranks[:5])


def _holdem_hole(label: str) -> str:
    """Compute holdem hole for the opponents workflow."""
    return label.split("|", 1)[0]


@cache
def _holdem_hole_classes(game: str = "holdem") -> dict[str, str]:
    """Compute holdem hole classes for the opponents workflow."""
    board = tuple(_card_from_str(card) for card in _holdem_board_strings(game))
    holes = sorted({_holdem_hole(label) for label in _holdem_p2_actions(game)})
    ranked = []
    for hole in holes:
        cards = (_card_from_str(hole[:2]), _card_from_str(hole[2:4]), *board)
        ranked.append((hole, _holdem_7card_rank(cards)))
    ranked.sort(key=lambda item: item[1])
    n_holes = len(ranked)
    classes: dict[str, str] = {}
    for idx, (hole, _rank) in enumerate(ranked):
        if idx < n_holes / 3:
            classes[hole] = "weak"
        elif idx < 2 * n_holes / 3:
            classes[hole] = "medium"
        else:
            classes[hole] = "strong"
    return classes


def _holdem_hole_class(label: str, game: str = "holdem") -> str:
    """Compute holdem hole class for the opponents workflow."""
    return _holdem_hole_classes(game)[_holdem_hole(label)]


@cache
def _holdem_hole_percentiles(game: str = "holdem") -> dict[str, float]:
    """Compute holdem hole percentiles for the opponents workflow."""
    board = tuple(_card_from_str(card) for card in _holdem_board_strings(game))
    holes = sorted({_holdem_hole(label) for label in _holdem_p2_actions(game)})
    ranked = []
    for hole in holes:
        cards = (_card_from_str(hole[:2]), _card_from_str(hole[2:4]), *board)
        ranked.append((hole, _holdem_7card_rank(cards)))
    ranked.sort(key=lambda item: item[1])
    denom = max(1, len(ranked) - 1)
    return {hole: idx / denom for idx, (hole, _rank) in enumerate(ranked)}


@dataclass(frozen=True)
class _HoldemDecisionProfile:
    """Represent holdem decision profile for the opponents workflow."""

    label: str
    hist: str
    hole: str
    hand_class: str
    hand_percentile: float
    mixedness: float
    fold_prob: float | None
    fold_headroom: float
    call_prob: float | None
    call_headroom: float
    raise_prob: float | None
    raise_headroom: float


@cache
def _holdem_decision_profiles(
    game: str = "holdem",
) -> dict[str, _HoldemDecisionProfile]:
    """Compute holdem decision profiles for the opponents workflow."""
    eq = holdem_equilibrium_opponent(game).behavior
    actions = _holdem_p2_actions(game)
    classes = _holdem_hole_classes(game)
    percentiles = _holdem_hole_percentiles(game)
    profiles: dict[str, _HoldemDecisionProfile] = {}
    for label, acts in actions.items():
        dist = eq[label]
        hist = label.split("|", 1)[1]
        hole = _holdem_hole(label)
        fold_prob = dist[acts.index("f")] if "f" in acts else None
        call_prob = dist[acts.index("c")] if "c" in acts else None
        raise_prob = dist[acts.index("p")] if "p" in acts else None
        profiles[label] = _HoldemDecisionProfile(
            label=label,
            hist=hist,
            hole=hole,
            hand_class=classes[hole],
            hand_percentile=percentiles[hole],
            mixedness=1.0 - max(dist),
            fold_prob=fold_prob,
            fold_headroom=1.0 - fold_prob if fold_prob is not None else 0.0,
            call_prob=call_prob,
            call_headroom=1.0 - call_prob if call_prob is not None else 0.0,
            raise_prob=raise_prob,
            raise_headroom=1.0 - raise_prob if raise_prob is not None else 0.0,
        )
    return profiles


def _holdem_decision_profile(
    label: str, game: str = "holdem"
) -> _HoldemDecisionProfile:
    """Compute holdem decision profile for the opponents workflow."""
    return _holdem_decision_profiles(game)[label]


def _holdem_perturb_by_profile(
    behavior: dict[str, list[float]],
    target: str,
    weight_fn: Callable[[_HoldemDecisionProfile], float],
    hist_pred: Callable[[str], bool],
    game: str = "holdem",
    action_pred: Callable[[tuple[str, ...]], bool] | None = None,
) -> dict[str, list[float]]:
    """Compute holdem perturb by profile."""
    actions = _holdem_p2_actions(game)
    out: dict[str, list[float]] = {}
    for label, dist in behavior.items():
        acts = actions[label]
        profile = _holdem_decision_profile(label, game)
        weight = weight_fn(profile)
        if (
            weight > 0.0
            and hist_pred(profile.hist)
            and target in acts
            and (action_pred is None or action_pred(acts))
        ):
            out[label] = [
                (1.0 - weight) * prob + (weight if action == target else 0.0)
                for action, prob in zip(acts, dist, strict=True)
            ]
        else:
            out[label] = list(dist)
    return out


def _holdem_perturb_by_hand_class(
    behavior: dict[str, list[float]],
    target: str,
    weights_by_class: Mapping[str, float],
    hist_pred: Callable[[str], bool],
    game: str = "holdem",
    action_pred: Callable[[tuple[str, ...]], bool] | None = None,
) -> dict[str, list[float]]:
    """Compute holdem perturb by hand class."""
    return _holdem_perturb_by_profile(
        behavior,
        target,
        lambda profile: weights_by_class.get(profile.hand_class, 0.0),
        hist_pred,
        game,
        action_pred,
    )


def holdem_board_public_fold_opponent(
    leak: float = 0.65, game: str = "holdem"
) -> Opponent:
    """Construct the holdem board public fold opponent policy."""
    eq = holdem_equilibrium_opponent(game).behavior
    shallow = _holdem_shallow_fold_lines(game)
    behavior = _holdem_perturb_by_profile(
        eq,
        "f",
        lambda profile: leak if profile.fold_headroom > 1e-9 else 0.0,
        lambda hist: hist in shallow,
        game,
        action_pred=lambda acts: "f" in acts,
    )
    return Opponent(name="board_public_fold", behavior=behavior, game=game)


def holdem_public_fold_opponent(leak: float = 0.65, game: str = "holdem") -> Opponent:
    """Construct the holdem public fold opponent policy."""
    opp = holdem_board_public_fold_opponent(leak=leak, game=game)
    return Opponent(name="public_fold", behavior=opp.behavior, game=game)


def holdem_board_public_call_opponent(
    leak: float = 0.55, game: str = "holdem"
) -> Opponent:
    """Construct the holdem board public call opponent policy."""
    behavior = _holdem_perturb_by_profile(
        holdem_equilibrium_opponent(game).behavior,
        "c",
        lambda profile: leak if profile.call_headroom > 1e-9 else 0.0,
        lambda _hist: True,
        game,
        action_pred=lambda acts: "c" in acts,
    )
    return Opponent(name="board_public_call", behavior=behavior, game=game)


def holdem_public_call_opponent(leak: float = 0.55, game: str = "holdem") -> Opponent:
    """Construct the holdem public call opponent policy."""
    opp = holdem_board_public_call_opponent(leak=leak, game=game)
    return Opponent(name="public_call", behavior=opp.behavior, game=game)


def holdem_board_public_aggression_opponent(
    leak: float = 0.4, game: str = "holdem"
) -> Opponent:
    """Construct the holdem board public aggression opponent policy."""
    behavior = _holdem_perturb_by_profile(
        holdem_equilibrium_opponent(game).behavior,
        "p",
        lambda profile: leak if profile.raise_headroom > 1e-9 else 0.0,
        lambda _hist: True,
        game,
        action_pred=lambda acts: "p" in acts,
    )
    return Opponent(name="board_public_aggression", behavior=behavior, game=game)


def holdem_public_aggression_opponent(
    leak: float = 0.4, game: str = "holdem"
) -> Opponent:
    """Construct the holdem public aggression opponent policy."""
    opp = holdem_board_public_aggression_opponent(leak=leak, game=game)
    return Opponent(name="public_aggression", behavior=opp.behavior, game=game)


def holdem_board_marginal_overfold_opponent(
    leak: float = 0.75,
    min_headroom: float = 0.05,
    game: str = "holdem",
) -> Opponent:
    """Construct the holdem board marginal overfold opponent policy."""
    behavior = _holdem_perturb_by_profile(
        holdem_equilibrium_opponent(game).behavior,
        "f",
        lambda profile: (
            leak
            if profile.hand_class == "medium" and profile.fold_headroom > min_headroom
            else 0.0
        ),
        lambda _hist: True,
        game,
        action_pred=lambda acts: "f" in acts,
    )
    return Opponent(name="board_marginal_overfold", behavior=behavior, game=game)


def holdem_weak_hand_overfold_opponent(
    weak_leak: float = 0.85,
    medium_leak: float = 0.25,
    strong_leak: float = 0.0,
    game: str = "holdem",
) -> Opponent:
    """Construct the holdem weak hand overfold opponent policy."""
    _ = (weak_leak, strong_leak)
    return holdem_board_marginal_overfold_opponent(leak=medium_leak, game=game)


def holdem_board_bluffcatcher_station_opponent(
    leak: float = 0.65,
    min_headroom: float = 0.05,
    game: str = "holdem",
) -> Opponent:
    """Construct the holdem board bluffcatcher station opponent policy."""
    behavior = _holdem_perturb_by_profile(
        holdem_equilibrium_opponent(game).behavior,
        "c",
        lambda profile: (
            leak
            if profile.hand_class == "medium" and profile.call_headroom > min_headroom
            else 0.0
        ),
        lambda _hist: True,
        game,
        action_pred=lambda acts: "f" in acts and "c" in acts,
    )
    return Opponent(name="board_bluffcatcher_station", behavior=behavior, game=game)


def holdem_bluffcatcher_station_opponent(
    weak_call_leak: float = 0.0,
    medium_call_leak: float = 0.65,
    strong_call_leak: float = 0.15,
    game: str = "holdem",
) -> Opponent:
    """Construct the holdem bluffcatcher station opponent policy."""
    _ = (weak_call_leak, strong_call_leak)
    opp = holdem_board_bluffcatcher_station_opponent(leak=medium_call_leak, game=game)
    return Opponent(name="bluffcatcher_station", behavior=opp.behavior, game=game)


def holdem_board_polarized_maniac_opponent(
    leak: float = 0.5,
    min_headroom: float = 0.05,
    game: str = "holdem",
) -> Opponent:
    """Construct the holdem board polarized maniac opponent policy."""
    behavior = _holdem_perturb_by_profile(
        holdem_equilibrium_opponent(game).behavior,
        "p",
        lambda profile: (
            leak
            if profile.hand_class in {"weak", "strong"}
            and profile.raise_headroom > min_headroom
            else 0.0
        ),
        lambda _hist: True,
        game,
        action_pred=lambda acts: "p" in acts,
    )
    return Opponent(name="board_polarized_maniac", behavior=behavior, game=game)


def holdem_polarized_maniac_opponent(
    weak_raise_leak: float = 0.5,
    medium_raise_leak: float = 0.0,
    strong_raise_leak: float = 0.5,
    game: str = "holdem",
) -> Opponent:
    """Construct the holdem polarized maniac opponent policy."""
    _ = (weak_raise_leak, medium_raise_leak)
    opp = holdem_board_polarized_maniac_opponent(leak=strong_raise_leak, game=game)
    return Opponent(name="polarized_maniac", behavior=opp.behavior, game=game)


def holdem_ambiguous_fold_marginal_opponent(
    leak: float = 0.75, min_headroom: float = 0.05, game: str = "holdem"
) -> Opponent:
    """Construct the holdem ambiguous fold marginal opponent policy."""
    behavior = _holdem_perturb_by_profile(
        holdem_equilibrium_opponent(game).behavior,
        "f",
        lambda profile: (
            leak
            if profile.hand_class == "medium" and profile.fold_headroom > min_headroom
            else 0.0
        ),
        lambda hist: hist in _holdem_shallow_fold_lines(game),
        game,
        action_pred=lambda acts: "f" in acts,
    )
    return Opponent(name="ambiguous_fold_marginal", behavior=behavior, game=game)


def holdem_ambiguous_fold_weak_opponent(
    leak: float = 0.75, game: str = "holdem"
) -> Opponent:
    """Construct the holdem ambiguous fold weak opponent policy."""
    return holdem_ambiguous_fold_marginal_opponent(leak=leak, game=game)


def holdem_ambiguous_fold_strong_opponent(
    leak: float = 0.75, min_headroom: float = 0.05, game: str = "holdem"
) -> Opponent:
    """Construct the holdem ambiguous fold strong opponent policy."""
    behavior = _holdem_perturb_by_profile(
        holdem_equilibrium_opponent(game).behavior,
        "f",
        lambda profile: (
            leak
            if profile.hand_class == "strong" and profile.fold_headroom > min_headroom
            else 0.0
        ),
        lambda hist: hist in _holdem_shallow_fold_lines(game),
        game,
        action_pred=lambda acts: "f" in acts,
    )
    return Opponent(name="ambiguous_fold_strong", behavior=behavior, game=game)


def holdem_showdown_selection_trap_opponent(game: str = "holdem") -> Opponent:
    """Construct the holdem showdown selection trap opponent policy."""
    shallow = _holdem_shallow_fold_lines(game)
    base = _holdem_perturb_by_profile(
        holdem_equilibrium_opponent(game).behavior,
        "f",
        lambda profile: (
            0.9
            if profile.hand_class == "medium" and profile.fold_headroom > 0.05
            else 0.0
        ),
        lambda hist: hist in shallow,
        game,
        action_pred=lambda acts: "f" in acts,
    )
    behavior = _holdem_perturb_by_profile(
        base,
        "c",
        lambda profile: (
            0.45
            if profile.hand_class == "strong" and profile.call_headroom > 0.05
            else 0.0
        ),
        lambda hist: hist in shallow,
        game,
        action_pred=lambda acts: "f" in acts and "c" in acts,
    )
    return Opponent(name="showdown_selection_trap", behavior=behavior, game=game)


def holdem_mixed_public_private_opponent(game: str = "holdem") -> Opponent:
    """Construct the holdem mixed public private opponent policy."""
    shallow = _holdem_shallow_fold_lines(game)
    base = holdem_board_public_fold_opponent(leak=0.55, game=game).behavior
    behavior = _holdem_perturb_by_profile(
        base,
        "c",
        lambda profile: (
            0.55
            if profile.hand_class == "medium" and profile.call_headroom > 0.05
            else 0.0
        ),
        lambda hist: hist not in shallow,
        game,
        action_pred=lambda acts: "f" in acts and "c" in acts,
    )
    return Opponent(name="mixed_public_private", behavior=behavior, game=game)


def holdem_population_opponent(
    archetype: str,
    leak: float,
    call_leak: float = 0.0,
    game: str = "holdem",
    name: str | None = None,
) -> Opponent:
    """Construct the holdem population opponent policy."""
    if not 0.0 <= leak <= 1.0:
        raise ValueError("leak must be in [0, 1]")
    nm = name or f"pop_{archetype}_{int(round(leak * 100)):02d}"
    if archetype in ("overfold", "censored_fold"):
        behavior = _holdem_perturb_toward("f", leak, game)
    elif archetype == "calling_station":
        behavior = _holdem_perturb_toward("c", leak, game)
    elif archetype == "maniac":
        behavior = _holdem_perturb_toward("p", leak, game)
    elif archetype == "mixed":
        behavior = _holdem_fold_and_call_behavior(leak, call_leak, game)
    else:
        raise ValueError(
            f"unknown archetype {archetype!r}; expected one of overfold, "
            "censored_fold, calling_station, maniac, mixed"
        )
    return Opponent(name=nm, behavior=behavior, game=game)


def holdem_population_sample(
    n: int = 50,
    seed: int = 2026,
    game: str = "holdem",
    leak_range: tuple[float, float] = (0.3, 0.9),
) -> dict[str, Opponent]:
    """Compute holdem population sample for the opponents workflow."""
    import random

    rng = random.Random(seed)
    archetypes = ("overfold", "censored_fold", "calling_station", "maniac", "mixed")
    lo, hi = leak_range
    suite: dict[str, Opponent] = {}
    for i in range(n):
        arch = rng.choice(archetypes)
        leak = rng.uniform(lo, hi)
        call_leak = rng.uniform(lo, hi) if arch == "mixed" else 0.0
        opp = holdem_population_opponent(
            arch, leak, call_leak=call_leak, game=game, name=f"draw{i:03d}_{arch}"
        )
        suite[opp.name] = opp
    return suite


def opponent_suite() -> dict[str, Opponent]:
    """Compute opponent suite for the opponents workflow."""
    return {
        "equilibrium": equilibrium_opponent(),
        "static_biased": static_biased_opponent(),
        "always_fold": always_fold_opponent(),
        "trap": trap_opponent(),
    }


def leduc_opponent_suite() -> dict[str, Opponent]:
    """Compute Leduc opponent suite for the opponents workflow."""
    return {
        "equilibrium": leduc_equilibrium_opponent(),
        "near_equilibrium": leduc_near_equilibrium_opponent(),
        "static_biased": leduc_static_biased_opponent(),
        "always_fold": leduc_always_fold_opponent(),
        "trap": leduc_trap_opponent(),
        "private_state_leak": leduc_private_state_leak_opponent(),
        "low_reach_leak": leduc_low_reach_leak_opponent(),
    }


def goofspiel_opponent_suite() -> dict[str, Opponent]:
    """Compute goofspiel opponent suite for the opponents workflow."""
    return {
        "equilibrium": goofspiel_equilibrium_opponent(),
        "lowball": goofspiel_lowball_opponent(),
        "highball": goofspiel_highball_opponent(),
        "uniform": goofspiel_uniform_opponent(),
    }


def holdem_showdown_opponent_suite(game: str = "holdem") -> dict[str, Opponent]:
    """Compute holdem showdown opponent suite."""
    return {
        "equilibrium": holdem_equilibrium_opponent(game),
        "near_equilibrium": holdem_near_equilibrium_opponent(game=game),
        "overfold": holdem_overfold_opponent(game=game),
        "calling_station": holdem_calling_station_opponent(game=game),
        "maniac": holdem_maniac_opponent(game=game),
        "censored_fold": holdem_censored_fold_opponent(game=game),
        "low_reach_leak": holdem_low_reach_leak_opponent(game=game),
        "fold_and_call": holdem_fold_and_call_opponent(game=game),
        "fold_mild_call": holdem_fold_mild_call_opponent(game=game),
        "private_card_fold": holdem_private_card_fold_opponent(game=game),
    }


def holdem_structured_opponent_suite(game: str = "holdem") -> dict[str, Opponent]:
    """Compute holdem structured opponent suite."""
    return {
        "equilibrium": holdem_equilibrium_opponent(game),
        "near_equilibrium": holdem_near_equilibrium_opponent(game=game),
        "board_public_fold": holdem_board_public_fold_opponent(game=game),
        "board_public_call": holdem_board_public_call_opponent(game=game),
        "board_marginal_overfold": holdem_board_marginal_overfold_opponent(game=game),
        "board_bluffcatcher_station": holdem_board_bluffcatcher_station_opponent(
            game=game
        ),
        "board_polarized_maniac": holdem_board_polarized_maniac_opponent(game=game),
        "board_ambiguous_fold_marginal": holdem_ambiguous_fold_marginal_opponent(
            game=game
        ),
        "ambiguous_fold_strong": holdem_ambiguous_fold_strong_opponent(game=game),
        "showdown_selection_trap": holdem_showdown_selection_trap_opponent(game=game),
        "mixed_public_private": holdem_mixed_public_private_opponent(game=game),
        "low_reach_private": holdem_low_reach_leak_opponent(game=game),
    }


_FACTORIES: dict[str, dict[str, Callable[..., Opponent]]] = {
    "kuhn": {
        "static_biased": static_biased_opponent,
        "always_fold": always_fold_opponent,
        "equilibrium": equilibrium_opponent,
        "trap": trap_opponent,
    },
    "leduc": {
        "static_biased": leduc_static_biased_opponent,
        "always_fold": leduc_always_fold_opponent,
        "equilibrium": leduc_equilibrium_opponent,
        "trap": leduc_trap_opponent,
        "near_equilibrium": leduc_near_equilibrium_opponent,
        "private_state_leak": leduc_private_state_leak_opponent,
        "low_reach_leak": leduc_low_reach_leak_opponent,
    },
    "goofspiel": {
        "lowball": goofspiel_lowball_opponent,
        "highball": goofspiel_highball_opponent,
        "uniform": goofspiel_uniform_opponent,
        "equilibrium": goofspiel_equilibrium_opponent,
    },
    "holdem": {
        "equilibrium": holdem_equilibrium_opponent,
        "near_equilibrium": holdem_near_equilibrium_opponent,
        "overfold": holdem_overfold_opponent,
        "calling_station": holdem_calling_station_opponent,
        "maniac": holdem_maniac_opponent,
        "censored_fold": holdem_censored_fold_opponent,
        "low_reach_leak": holdem_low_reach_leak_opponent,
        "fold_and_call": holdem_fold_and_call_opponent,
        "fold_mild_call": holdem_fold_mild_call_opponent,
        "private_card_fold": holdem_private_card_fold_opponent,
        "board_public_fold": holdem_board_public_fold_opponent,
        "board_public_call": holdem_board_public_call_opponent,
        "board_public_aggression": holdem_board_public_aggression_opponent,
        "board_marginal_overfold": holdem_board_marginal_overfold_opponent,
        "board_bluffcatcher_station": holdem_board_bluffcatcher_station_opponent,
        "board_polarized_maniac": holdem_board_polarized_maniac_opponent,
        "board_ambiguous_fold_marginal": holdem_ambiguous_fold_marginal_opponent,
        "public_fold": holdem_public_fold_opponent,
        "public_call": holdem_public_call_opponent,
        "public_aggression": holdem_public_aggression_opponent,
        "weak_hand_overfold": holdem_weak_hand_overfold_opponent,
        "bluffcatcher_station": holdem_bluffcatcher_station_opponent,
        "polarized_maniac": holdem_polarized_maniac_opponent,
        "ambiguous_fold_marginal": holdem_ambiguous_fold_marginal_opponent,
        "ambiguous_fold_weak": holdem_ambiguous_fold_weak_opponent,
        "ambiguous_fold_strong": holdem_ambiguous_fold_strong_opponent,
        "showdown_selection_trap": holdem_showdown_selection_trap_opponent,
        "mixed_public_private": holdem_mixed_public_private_opponent,
    },
}


def opponent_from_spec(spec: Mapping[str, Any]) -> Opponent:
    """Compute opponent from spec for the opponents workflow."""
    params = dict(spec)
    game = params.pop("game", "kuhn")
    try:
        kind = params.pop("type")
    except KeyError:
        raise ValueError("opponent spec must include a 'type' key") from None
    if game not in _FACTORIES:
        raise ValueError(f"unknown game {game!r}; expected one of {sorted(_FACTORIES)}")
    registry = _FACTORIES[game]
    if kind not in registry:
        raise ValueError(
            f"unknown opponent type {kind!r} for {game!r}; expected one of "
            f"{sorted(registry)}"
        )
    return registry[kind](**params)


def best_response_value(opponent: Opponent) -> float:
    """Compute the best response value."""
    from .solvers import best_response

    return best_response(opponent.realization(), game=opponent.game).value


__all__ = [
    "Opponent",
    "static_biased_opponent",
    "always_fold_opponent",
    "equilibrium_opponent",
    "trap_opponent",
    "opponent_suite",
    "leduc_static_biased_opponent",
    "leduc_always_fold_opponent",
    "leduc_equilibrium_opponent",
    "leduc_trap_opponent",
    "leduc_near_equilibrium_opponent",
    "leduc_private_state_leak_opponent",
    "leduc_low_reach_leak_opponent",
    "leduc_opponent_suite",
    "goofspiel_lowball_opponent",
    "goofspiel_highball_opponent",
    "goofspiel_uniform_opponent",
    "goofspiel_equilibrium_opponent",
    "goofspiel_opponent_suite",
    "holdem_equilibrium_opponent",
    "holdem_near_equilibrium_opponent",
    "holdem_overfold_opponent",
    "holdem_calling_station_opponent",
    "holdem_maniac_opponent",
    "holdem_censored_fold_opponent",
    "holdem_low_reach_leak_opponent",
    "holdem_public_fold_opponent",
    "holdem_public_call_opponent",
    "holdem_public_aggression_opponent",
    "holdem_weak_hand_overfold_opponent",
    "holdem_bluffcatcher_station_opponent",
    "holdem_polarized_maniac_opponent",
    "holdem_ambiguous_fold_marginal_opponent",
    "holdem_ambiguous_fold_weak_opponent",
    "holdem_ambiguous_fold_strong_opponent",
    "holdem_showdown_selection_trap_opponent",
    "holdem_mixed_public_private_opponent",
    "holdem_showdown_opponent_suite",
    "holdem_structured_opponent_suite",
    "opponent_from_spec",
    "best_response_value",
]
