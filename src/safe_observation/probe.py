""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field


def weights_from_intervals(
    intervals: Mapping[str, Sequence[tuple[float, float]]],
    importance: Mapping[str, Sequence[float]] | None = None,
) -> dict[str, float]:
    ""
    weights: dict[str, float] = {}
    for label, bounds in intervals.items():
        if importance is None:
            weights[label] = sum(u - lo for lo, u in bounds)
        else:
            imp = importance.get(label)
            if imp is None:
                weights[label] = sum(u - lo for lo, u in bounds)
            else:
                weights[label] = sum(
                    (u - lo) * w for (lo, u), w in zip(bounds, imp, strict=True)
                )
    return weights


@dataclass
class ProbeBudget:
    ""

    total: float = 0.0
    per_round: float = float("inf")
    spent: float = 0.0

    def remaining(self) -> float:
        ""
        return max(0.0, self.total - self.spent)

    def allowance(self) -> float:
        ""
        return min(self.per_round, self.remaining())

    def charge(self, rho_spent: float) -> None:
        ""
        if rho_spent < 0.0:
            raise ValueError("rho_spent must be non-negative")
        self.spent += rho_spent


@dataclass
class SafetyBudgetLedger:
    ""

    rho_cap: float = 0.5
    hard_total: float = float("inf")
    debt_max: float = 3.0
    eta_debt: float = 1.0
    s_scale: float = 1.0
    gamma: float = 1.0
    micro_rho: float = 0.0
    tau_point: float = 1.0
    spent: float = 0.0
    debt: float = 0.0

    def raw_grant(self, signal: float) -> float:
        ""
        scaled = min(max(signal / self.s_scale, 0.0), 1.0)
        return self.rho_cap * scaled**self.gamma

    def allowance(self, signal: float, point_signal: float = 0.0) -> float:
        ""
        raw = self.raw_grant(signal)
        if self.micro_rho > 0.0 and point_signal > self.tau_point:
            raw += self.micro_rho
        remaining_hard = max(0.0, self.hard_total - self.spent)
        remaining_debt = max(0.0, self.debt_max - self.debt)
        return max(
            0.0,
            min(raw, self.rho_cap, remaining_hard, remaining_debt),
        )

    def charge(self, rho_spent: float) -> None:
        ""
        if rho_spent < 0.0:
            raise ValueError("rho_spent must be non-negative")
        self.spent += rho_spent

    def settle(self, rho_spent: float, realized_gain: float) -> None:
        ""
        if rho_spent < 0.0:
            raise ValueError("rho_spent must be non-negative")
        credit = max(0.0, realized_gain)
        self.debt = max(0.0, self.eta_debt * self.debt + rho_spent - credit)


@dataclass
class ProbePlanner:
    ""

    beta: float = 0.0
    budget: ProbeBudget = field(default_factory=ProbeBudget)

    def weights(
        self,
        intervals: Mapping[str, Sequence[tuple[float, float]]],
        importance: Mapping[str, Sequence[float]] | None = None,
    ) -> dict[str, float]:
        ""
        return weights_from_intervals(intervals, importance)

    def rho(self) -> float:
        ""
        return self.budget.allowance()

    def record(self, rho_spent: float) -> None:
        ""
        self.budget.charge(rho_spent)


def information_gain(coeffs: Sequence[float], realization: Sequence[float]) -> float:
    ""
    return sum(c * x for c, x in zip(coeffs, realization, strict=True))


__all__ = [
    "weights_from_intervals",
    "ProbeBudget",
    "SafetyBudgetLedger",
    "ProbePlanner",
    "information_gain",
]
