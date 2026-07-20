""

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from time import perf_counter


@dataclass
class _Stage:
    seconds: float = 0.0
    calls: int = 0


@dataclass
class StageTimer:
    ""

    _stages: dict[str, _Stage] = field(default_factory=dict)

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        ""
        start = perf_counter()
        try:
            yield
        finally:
            self.add(name, perf_counter() - start)

    def add(self, name: str, seconds: float) -> None:
        ""
        st = self._stages.get(name)
        if st is None:
            st = self._stages[name] = _Stage()
        st.seconds += seconds
        st.calls += 1

    def add_totals(self, name: str, seconds: float, calls: int) -> None:
        ""
        st = self._stages.get(name)
        if st is None:
            st = self._stages[name] = _Stage()
        st.seconds += seconds
        st.calls += calls

    def merge(self, other: "StageTimer") -> None:
        ""
        for name, st in other._stages.items():
            cur = self._stages.get(name)
            if cur is None:
                cur = self._stages[name] = _Stage()
            cur.seconds += st.seconds
            cur.calls += st.calls

    def total_seconds(self) -> float:
        ""
        return sum(st.seconds for st in self._stages.values())

    def as_dict(self) -> dict[str, dict[str, float]]:
        ""
        return {
            name: {"seconds": st.seconds, "calls": st.calls}
            for name, st in self._stages.items()
        }


__all__ = ["StageTimer"]
