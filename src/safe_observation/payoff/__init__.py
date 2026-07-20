""

from collections.abc import Sequence

from .. import native


class PayoffMatrix:
    ""

    def __init__(self, game: str = "kuhn") -> None:
        self.game = game
        self._native = native.PayoffMatrix(game)

    @property
    def nrows(self) -> int:
        ""
        return self._native.nrows

    @property
    def ncols(self) -> int:
        ""
        return self._native.ncols

    @property
    def entries(self) -> tuple[tuple[int, int, float], ...]:
        ""
        return tuple((int(r), int(c), float(v)) for r, c, v in self._native.entries)

    @property
    def nnz(self) -> int:
        ""
        return len(self._native.entries)

    def bilinear(self, x: Sequence[float], y: Sequence[float]) -> float:
        ""
        return self._native.bilinear(list(x), list(y))

    def matvec_a_y(self, y: Sequence[float]) -> tuple[float, ...]:
        ""
        return tuple(self._native.matvec_a_y(list(y)))

    def matvec_at_x(self, x: Sequence[float]) -> tuple[float, ...]:
        ""
        return tuple(self._native.matvec_at_x(list(x)))

    def dense(self) -> tuple[tuple[float, ...], ...]:
        ""
        a = [[0.0] * self.ncols for _ in range(self.nrows)]
        for r, c, v in self.entries:
            a[r][c] += v
        return tuple(tuple(row) for row in a)


def build(game: str = "kuhn") -> PayoffMatrix:
    ""
    return PayoffMatrix(game)


def build_kuhn() -> PayoffMatrix:
    ""
    return PayoffMatrix("kuhn")


def build_leduc() -> PayoffMatrix:
    ""
    return PayoffMatrix("leduc")


__all__ = ["PayoffMatrix", "build", "build_kuhn", "build_leduc"]
