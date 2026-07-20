""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from .. import native


@dataclass(frozen=True)
class SequenceFormSizes:
    ""

    num_sequences_p1: int
    num_infosets_p1: int
    num_sequences_p2: int
    num_infosets_p2: int


@dataclass(frozen=True)
class InfoSet:
    ""

    label: str
    parent_seq: int
    children: tuple[tuple[str, int], ...]


@dataclass(eq=False)
class SequenceForm:
    ""

    game: str
    player: int
    sequences: tuple[str, ...]
    info_sets: tuple[InfoSet, ...]
    e_entries: tuple[tuple[int, int, float], ...]
    e: tuple[float, ...]
    seq_index: dict[str, int] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.seq_index = {label: i for i, label in enumerate(self.sequences)}

    @property
    def num_sequences(self) -> int:
        return len(self.sequences)

    @property
    def num_infosets(self) -> int:
        return len(self.info_sets)

    @property
    def num_constraints(self) -> int:
        return len(self.e)

    def realization_from_behavior(
        self, behavior: Mapping[str, Sequence[float]] | None = None
    ) -> tuple[float, ...]:
        ""
        b = {} if behavior is None else behavior
        x = [0.0] * self.num_sequences
        x[0] = 1.0
        for info in self.info_sets:
            parent = x[info.parent_seq]
            dist = b.get(info.label)
            n = len(info.children)
            for i, (_action, child) in enumerate(info.children):
                p = dist[i] if dist is not None else 1.0 / n
                x[child] = parent * p
        return tuple(x)

    def behavior_from_realization(
        self, x: Sequence[float], tol: float = 1e-12
    ) -> dict[str, list[float]]:
        ""
        behavior: dict[str, list[float]] = {}
        for info in self.info_sets:
            parent = x[info.parent_seq]
            if parent > tol:
                behavior[info.label] = [
                    x[child] / parent for _action, child in info.children
                ]
            else:
                n = len(info.children)
                behavior[info.label] = [1.0 / n] * n
        return behavior

    def constraint_residual(self, x: Sequence[float]) -> float:
        ""
        ex = [0.0] * self.num_constraints
        for row, col, value in self.e_entries:
            ex[row] += value * x[col]
        return max(abs(ex[i] - self.e[i]) for i in range(self.num_constraints))

    def dense_e(self) -> tuple[tuple[float, ...], ...]:
        ""
        dense = [[0.0] * self.num_sequences for _ in range(self.num_constraints)]
        for row, col, value in self.e_entries:
            dense[row][col] = value
        return tuple(tuple(row) for row in dense)


_KNOWN_GAMES = ("kuhn", "leduc", "goofspiel", "holdem")


def compile(game: str, player: int) -> SequenceForm:
    ""
    if player not in (0, 1):
        raise ValueError("player must be 0 (player 1) or 1 (player 2)")
    if (
        game not in _KNOWN_GAMES
        and not game.startswith("holdem_")
        and not game.startswith("cchain")
    ):
        raise ValueError(f"unknown game {game!r}; expected one of {list(_KNOWN_GAMES)}")

    sequences, raw_info_sets, entries, rhs = native.sequence_form(game, player)
    info_sets = tuple(
        InfoSet(
            label=label,
            parent_seq=parent_seq,
            children=tuple((action, child) for action, child in children),
        )
        for label, parent_seq, children in raw_info_sets
    )
    return SequenceForm(
        game=game,
        player=player,
        sequences=tuple(sequences),
        info_sets=info_sets,
        e_entries=tuple((int(r), int(c), float(v)) for r, c, v in entries),
        e=tuple(float(v) for v in rhs),
    )


def compile_kuhn(player: int) -> SequenceForm:
    ""
    return compile("kuhn", player)


def compile_leduc(player: int) -> SequenceForm:
    ""
    return compile("leduc", player)


def compile_goofspiel(player: int) -> SequenceForm:
    ""
    return compile("goofspiel", player)


def compile_holdem(player: int) -> SequenceForm:
    ""
    return compile("holdem", player)


def kuhn_sizes() -> SequenceFormSizes:
    ""
    return SequenceFormSizes(*native.sequence_form_sizes("kuhn"))


def leduc_sizes() -> SequenceFormSizes:
    ""
    return SequenceFormSizes(*native.sequence_form_sizes("leduc"))


__all__ = [
    "InfoSet",
    "SequenceForm",
    "SequenceFormSizes",
    "compile",
    "compile_kuhn",
    "compile_leduc",
    "compile_goofspiel",
    "compile_holdem",
    "kuhn_sizes",
    "leduc_sizes",
]
