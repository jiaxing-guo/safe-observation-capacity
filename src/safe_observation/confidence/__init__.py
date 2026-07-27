"""Public interfaces for confidence. See Public and Active Confidence Sets and supplementary Confidence Sets under Censoring."""

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
import math

from .. import native
from ..sequence_form import compile as compile_game


@dataclass(frozen=True)
class SimultaneousDeltaBudget:
    """One failure budget allocated across public and reveal constraints."""

    global_delta: float
    public_rows: int
    reveal_rows: int
    public_delta: float
    reveal_delta: float
    public_row_delta: float | None
    reveal_row_delta: float | None

    @property
    def allocated_delta(self) -> float:
        """Return the family-level failure probability charged by the allocation."""
        return self.public_delta + self.reveal_delta


def allocate_simultaneous_delta(
    delta: float,
    *,
    public_rows: int,
    reveal_rows: int,
) -> SimultaneousDeltaBudget:
    """Split a global failure budget over nonempty families and their rows."""
    if not 0.0 < delta < 1.0:
        raise ValueError("delta must be in the open interval (0, 1)")
    if public_rows < 0 or reveal_rows < 0:
        raise ValueError("confidence-family row counts must be non-negative")
    active_families = int(public_rows > 0) + int(reveal_rows > 0)
    family_delta = delta / active_families if active_families else 0.0
    public_delta = family_delta if public_rows else 0.0
    reveal_delta = family_delta if reveal_rows else 0.0
    return SimultaneousDeltaBudget(
        global_delta=delta,
        public_rows=public_rows,
        reveal_rows=reveal_rows,
        public_delta=public_delta,
        reveal_delta=reveal_delta,
        public_row_delta=public_delta / public_rows if public_rows else None,
        reveal_row_delta=reveal_delta / reveal_rows if reveal_rows else None,
    )


def hoeffding_halfwidth(n: int, delta: float) -> float:
    """Compute hoeffding halfwidth for the init workflow."""
    if not 0.0 < delta < 1.0:
        raise ValueError("delta must be in the open interval (0, 1)")
    if n <= 0:
        return 1.0
    return math.sqrt(math.log(2.0 / delta) / (2.0 * n))


def hoeffding_interval(p_hat: float, n: int, delta: float) -> tuple[float, float]:
    """Compute the hoeffding interval for the init workflow."""
    half = hoeffding_halfwidth(n, delta)
    return (max(0.0, p_hat - half), min(1.0, p_hat + half))


def empirical_bernstein_halfwidth(n: int, variance: float, delta: float) -> float:
    """Compute empirical bernstein halfwidth for the init workflow."""
    if not 0.0 < delta < 1.0:
        raise ValueError("delta must be in the open interval (0, 1)")
    if n <= 0:
        return 1.0
    ln = math.log(3.0 / delta)
    return math.sqrt(2.0 * variance * ln / n) + 3.0 * ln / n


def empirical_bernstein_interval(
    p_hat: float, n: int, delta: float
) -> tuple[float, float]:
    """Compute the empirical bernstein interval."""
    half = empirical_bernstein_halfwidth(n, p_hat * (1.0 - p_hat), delta)
    return (max(0.0, p_hat - half), min(1.0, p_hat + half))


def _interval(p_hat: float, n: int, delta: float, method: str) -> tuple[float, float]:
    """Compute interval for the init workflow."""
    if method == "hoeffding":
        return hoeffding_interval(p_hat, n, delta)
    if method == "empirical_bernstein":
        return empirical_bernstein_interval(p_hat, n, delta)
    raise ValueError(
        f"unknown method {method!r}; expected 'hoeffding' or 'empirical_bernstein'"
    )


def time_uniform_delta(delta: float, round_index: int) -> float:
    """Compute time uniform delta for the init workflow."""
    if round_index < 1:
        raise ValueError("round_index must be >= 1")
    # The Basel-series allocation sums to delta over an unbounded run.
    return delta * 6.0 / (math.pi**2 * round_index * round_index)


def public_key(game: str, label: str) -> str:
    """Compute public key for the init workflow."""
    if game == "kuhn":
        return label.split(":", 1)[1]
    if game == "leduc":
        return label.split("|", 1)[1]
    if game == "goofspiel":
        return label.split("|", 1)[1]
    if game == "holdem" or game.startswith("holdem_"):
        return label.split("|", 1)[1]
    if game.startswith("cchain"):
        return label.split("|", 1)[1]
    raise ValueError(
        f"unknown game {game!r}; expected 'kuhn', 'leduc', 'goofspiel', or 'holdem'"
    )


class OpponentEvidenceStore:
    """Represent opponent evidence store for the init workflow."""

    def __init__(
        self, info_sets: Iterable[tuple[str, int]], game: str = "kuhn"
    ) -> None:
        """Initialize the opponent evidence store."""
        self.game = game
        self._counts: dict[str, list[int]] = {label: [0] * k for label, k in info_sets}
        if not self._counts:
            raise ValueError("at least one information set is required")

    @classmethod
    def for_kuhn(cls) -> "OpponentEvidenceStore":
        """Compute for Kuhn for the init workflow."""
        return cls.for_game("kuhn")

    @classmethod
    def for_game(cls, game: str) -> "OpponentEvidenceStore":
        """Compute for game for the init workflow."""
        sf = compile_game(game, 1)
        return cls(
            ((info.label, len(info.children)) for info in sf.info_sets), game=game
        )

    @property
    def labels(self) -> tuple[str, ...]:
        """Return the labels for the init workflow."""
        return tuple(self._counts)

    def update(self, label: str, action_index: int, times: int = 1) -> None:
        """Update the stored state with the supplied value."""
        self._counts[label][action_index] += times

    def record(self, label: str, counts: Sequence[int]) -> None:
        """Record the supplied observation or budget expenditure."""
        row = self._counts[label]
        if len(counts) != len(row):
            raise ValueError(f"expected {len(row)} action counts, got {len(counts)}")
        for i, c in enumerate(counts):
            row[i] += c

    def visits(self, label: str) -> int:
        """Compute visits for the init workflow."""
        return sum(self._counts[label])

    def counts(self, label: str) -> tuple[int, ...]:
        """Compute counts for the init workflow."""
        return tuple(self._counts[label])

    def p_hat(self, label: str) -> tuple[float, ...]:
        """Compute p hat for the init workflow."""
        row = self._counts[label]
        n = sum(row)
        if n == 0:
            k = len(row)
            return tuple(1.0 / k for _ in range(k))
        return tuple(c / n for c in row)

    def _num_pairs(self) -> int:
        """Compute num pairs for the init workflow."""
        return sum(len(row) for row in self._counts.values())

    def interval(
        self, label: str, delta: float, method: str = "hoeffding"
    ) -> list[tuple[float, float]]:
        """Compute interval for the init workflow."""
        n = self.visits(label)
        return [_interval(p, n, delta, method) for p in self.p_hat(label)]

    def intervals(
        self,
        delta: float,
        method: str = "hoeffding",
        union_bound: bool = True,
        round_index: int | None = None,
    ) -> dict[str, list[tuple[float, float]]]:
        """Compute intervals for the init workflow."""
        if round_index is not None:
            delta = time_uniform_delta(delta, round_index)
        # Allocate failure probability over every action coordinate at once.
        per = delta / self._num_pairs() if union_bound else delta
        return {label: self.interval(label, per, method) for label in self._counts}

    def public_groups(self) -> dict[str, list[str]]:
        """Compute public groups for the init workflow."""
        groups: dict[str, list[str]] = {}
        for label in self._counts:
            # Private information sets sharing one public history form a fiber.
            groups.setdefault(public_key(self.game, label), []).append(label)
        return groups

    def _public_counts(self) -> dict[str, list[int]]:
        """Compute public counts for the init workflow."""
        agg: dict[str, list[int]] = {}
        for label, row in self._counts.items():
            key = public_key(self.game, label)
            cur = agg.get(key)
            if cur is None:
                agg[key] = list(row)
            else:
                # Public monitoring pools counts without exposing private labels.
                if len(cur) != len(row):
                    raise ValueError(
                        f"public state {key!r} aggregates info sets with "
                        "differing action counts"
                    )
                for i, c in enumerate(row):
                    cur[i] += c
        return agg

    @property
    def num_public_pairs(self) -> int:
        """Return the number of public action coordinates in a simultaneous set."""
        return sum(len(row) for row in self._public_counts().values())

    def public_counts(self) -> dict[str, tuple[int, ...]]:
        """Return aggregated public counts without exposing private labels."""
        return {key: tuple(row) for key, row in self._public_counts().items()}

    def public_intervals(
        self, delta: float, method: str = "hoeffding", union_bound: bool = True
    ) -> dict[str, list[tuple[float, float]]]:
        """Compute the public intervals for the init workflow."""
        agg = self._public_counts()
        num_pairs = sum(len(row) for row in agg.values())
        per = delta / num_pairs if union_bound else delta
        out: dict[str, list[tuple[float, float]]] = {}
        for key, counts in agg.items():
            n = sum(counts)
            phat = [c / n for c in counts] if n else [0.0] * len(counts)
            out[key] = [_interval(p, n, per, method) for p in phat]
        return out


class ConfidenceSet:
    """Represent confidence set for the init workflow."""

    def __init__(self, native_set) -> None:
        """Initialize the confidence set for the init workflow."""
        self._native = native_set

    @property
    def nrows(self) -> int:
        """Return the nrows for the init workflow."""
        return self._native.nrows

    @property
    def ncols(self) -> int:
        """Return the ncols for the init workflow."""
        return self._native.ncols

    @property
    def g_entries(self) -> tuple[tuple[int, int, float], ...]:
        """Return the g entries for the init workflow."""
        return tuple((int(r), int(c), float(v)) for r, c, v in self._native.g_entries)

    @property
    def h(self) -> tuple[float, ...]:
        """Return the h for the init workflow."""
        return tuple(float(v) for v in self._native.h)

    def max_violation(self, y: Sequence[float]) -> float:
        """Compute max violation for the init workflow."""
        return self._native.max_violation(list(y))

    def contains(self, y: Sequence[float], tol: float = 1e-9) -> bool:
        """Return whether the confidence set contains the supplied realization."""
        if self.nrows == 0:
            return True
        return self.max_violation(y) <= tol


def build_confidence_set(
    game: str,
    intervals: Mapping[str, Sequence[tuple[float, float]]],
) -> ConfidenceSet:
    """Build confidence set for the init workflow."""
    payload = {label: [tuple(b) for b in bounds] for label, bounds in intervals.items()}
    return ConfidenceSet(native.ConfidenceSet(game, payload))


def build_kuhn_confidence_set(
    intervals: Mapping[str, Sequence[tuple[float, float]]],
) -> ConfidenceSet:
    """Build Kuhn confidence set for the init workflow."""
    return build_confidence_set("kuhn", intervals)


def build_public_confidence_set(
    game: str,
    groups: Mapping[str, Sequence[str]],
    intervals: Mapping[str, Sequence[tuple[float, float]]],
) -> ConfidenceSet:
    """Build public confidence set for the init workflow."""
    grp = {key: list(labels) for key, labels in groups.items()}
    payload = {key: [tuple(b) for b in bounds] for key, bounds in intervals.items()}
    return ConfidenceSet(native.ConfidenceSet.from_public(game, grp, payload))


__all__ = [
    "hoeffding_halfwidth",
    "hoeffding_interval",
    "empirical_bernstein_halfwidth",
    "empirical_bernstein_interval",
    "time_uniform_delta",
    "public_key",
    "OpponentEvidenceStore",
    "ConfidenceSet",
    "build_confidence_set",
    "build_kuhn_confidence_set",
    "build_public_confidence_set",
]
