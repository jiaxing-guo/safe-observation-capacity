"""Public interfaces for payoff. See Preliminaries and Problem Setup."""

from collections.abc import Sequence

from .. import native


class PayoffMatrix:
    """Represent payoff matrix for the init workflow."""

    def __init__(self, game: str = "kuhn") -> None:
        """Initialize the payoff matrix for the init workflow."""
        self.game = game
        self._native = native.PayoffMatrix(game)

    @property
    def nrows(self) -> int:
        """Return the nrows for the init workflow."""
        return self._native.nrows

    @property
    def ncols(self) -> int:
        """Return the ncols for the init workflow."""
        return self._native.ncols

    @property
    def entries(self) -> tuple[tuple[int, int, float], ...]:
        """Return the entries for the init workflow."""
        return tuple((int(r), int(c), float(v)) for r, c, v in self._native.entries)

    @property
    def nnz(self) -> int:
        """Return the nnz for the init workflow."""
        return len(self._native.entries)

    def bilinear(self, x: Sequence[float], y: Sequence[float]) -> float:
        """Compute bilinear for the init workflow."""
        return self._native.bilinear(list(x), list(y))

    def matvec_a_y(self, y: Sequence[float]) -> tuple[float, ...]:
        """Compute matvec a y for the init workflow."""
        return tuple(self._native.matvec_a_y(list(y)))

    def matvec_at_x(self, x: Sequence[float]) -> tuple[float, ...]:
        """Compute matvec at x for the init workflow."""
        return tuple(self._native.matvec_at_x(list(x)))

    def dense(self) -> tuple[tuple[float, ...], ...]:
        """Compute dense for the init workflow."""
        a = [[0.0] * self.ncols for _ in range(self.nrows)]
        for r, c, v in self.entries:
            a[r][c] += v
        return tuple(tuple(row) for row in a)


def build(game: str = "kuhn") -> PayoffMatrix:
    """Build the configured game or confidence object."""
    return PayoffMatrix(game)


def build_kuhn() -> PayoffMatrix:
    """Build Kuhn for the init workflow."""
    return PayoffMatrix("kuhn")


def build_leduc() -> PayoffMatrix:
    """Build Leduc for the init workflow."""
    return PayoffMatrix("leduc")


__all__ = ["PayoffMatrix", "build", "build_kuhn", "build_leduc"]
