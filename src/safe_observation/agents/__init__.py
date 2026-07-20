""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from ..confidence import OpponentEvidenceStore
from ..probe import ProbeBudget, information_gain, weights_from_intervals
from ..sequence_form import compile as compile_game
from ..solvers import (
    confidence_sensitivity,
    fallback_mixture_repair,
    probe_coefficients,
    robust_safe_response,
    robust_safe_response_probe,
    robust_safe_response_public,
    safety_verifier,
    solve_blueprint,
)
from ..timing import StageTimer


@dataclass
class Decision:
    ""

    realization: tuple[float, ...]
    behavior: dict[str, list[float]]
    robust_value: float
    safety_value: float
    repaired: bool
    mean_ci_width: float
    ci_width_by_infoset: dict[str, float]

    info_gain: float = 0.0

    rho_granted: float = 0.0

    rho_spent: float = 0.0


class OnlineSafeExploitAgent:
    ""

    def __init__(
        self,
        game: str = "kuhn",
        delta: float = 0.05,
        eps_safe: float = 0.0,
        method: str = "hoeffding",
        monitoring: str = "full",
        timer: StageTimer | None = None,
        probing: bool = False,
        beta: float = 0.0,
        probe_budget: ProbeBudget | None = None,
        importance_mode: str = "uniform",
    ) -> None:
        if monitoring not in ("full", "public"):
            raise ValueError(
                f"unknown monitoring {monitoring!r}; expected 'full' or 'public'"
            )
        if probing and monitoring != "full":
            raise ValueError("probing requires monitoring='full'")
        if importance_mode not in ("uniform", "sensitivity"):
            raise ValueError(
                f"unknown importance_mode {importance_mode!r}; expected "
                "'uniform' or 'sensitivity'"
            )
        self.game = game
        self.delta = delta
        self.eps_safe = eps_safe
        self.method = method
        self.monitoring = monitoring
        self.timer = timer if timer is not None else StageTimer()
        self.probing = probing
        self.beta = beta
        self.probe_budget = probe_budget if probe_budget is not None else ProbeBudget()
        self.importance_mode = importance_mode

        self.blueprint = solve_blueprint(game, method="lp")
        self.v_ref = self.blueprint.value
        assert self.blueprint.realization is not None
        self.x_blueprint = self.blueprint.realization
        self._sf0 = compile_game(game, 0)
        self.evidence = OpponentEvidenceStore.for_game(game)

    def select(self) -> Decision:
        ""
        groups: dict[str, list[str]] = {}
        with self.timer.stage("confidence_build"):
            if self.monitoring == "public":
                groups = self.evidence.public_groups()
                intervals = self.evidence.public_intervals(
                    self.delta, method=self.method
                )
            else:
                intervals = self.evidence.intervals(self.delta, method=self.method)

        if self.probing:
            return self._select_probing(intervals)

        with self.timer.stage("robust_solve"):
            if self.monitoring == "public":
                response = robust_safe_response_public(
                    groups,
                    intervals,
                    v_ref=self.v_ref,
                    eps_safe=self.eps_safe,
                    game=self.game,
                )
            else:
                response = robust_safe_response(
                    intervals,
                    v_ref=self.v_ref,
                    eps_safe=self.eps_safe,
                    game=self.game,
                )
        x_t = response.realization

        ci_width_by_infoset = {
            key: sum(u - lo for lo, u in bounds) / len(bounds)
            for key, bounds in intervals.items()
        }
        mean_ci_width = sum(ci_width_by_infoset.values()) / len(ci_width_by_infoset)

        with self.timer.stage("safety_verify"):
            safety = safety_verifier(x_t, game=self.game)
        repaired = False
        if not safety.is_safe(self.v_ref, self.eps_safe):
            with self.timer.stage("repair"):
                x_t = fallback_mixture_repair(
                    self.x_blueprint, x_t, self.v_ref, self.eps_safe, game=self.game
                )
                safety = safety_verifier(x_t, game=self.game)
            repaired = True

        return Decision(
            realization=x_t,
            behavior=self._sf0.behavior_from_realization(x_t),
            robust_value=response.robust_value,
            safety_value=safety.value,
            repaired=repaired,
            mean_ci_width=mean_ci_width,
            ci_width_by_infoset=ci_width_by_infoset,
        )

    def _select_probing(
        self, intervals: Mapping[str, list[tuple[float, float]]]
    ) -> Decision:
        ""
        opp_behavior = {
            label: list(self.evidence.p_hat(label)) for label in self.evidence.labels
        }
        if self.importance_mode == "sensitivity":
            with self.timer.stage("sensitivity_solve"):
                importance = confidence_sensitivity(
                    intervals,
                    v_ref=self.v_ref,
                    eps_safe=self.eps_safe,
                    game=self.game,
                )
            weights = weights_from_intervals(intervals, importance)
        else:
            weights = weights_from_intervals(intervals)
        rho = self.probe_budget.allowance()

        with self.timer.stage("probe_solve"):
            response = robust_safe_response_probe(
                intervals,
                opp_behavior,
                weights,
                v_ref=self.v_ref,
                eps_safe=self.eps_safe,
                beta=self.beta,
                rho=rho,
                game=self.game,
            )
        x_t = response.realization

        ci_width_by_infoset = {
            key: sum(u - lo for lo, u in bounds) / len(bounds)
            for key, bounds in intervals.items()
        }
        mean_ci_width = sum(ci_width_by_infoset.values()) / len(ci_width_by_infoset)
        coeffs = probe_coefficients(opp_behavior, weights, game=self.game)

        with self.timer.stage("safety_verify"):
            safety = safety_verifier(x_t, game=self.game)
        repaired = False
        if safety.value < self.v_ref - self.eps_safe - rho - 1e-8:
            with self.timer.stage("repair"):
                x_t = fallback_mixture_repair(
                    self.x_blueprint,
                    x_t,
                    self.v_ref,
                    self.eps_safe + rho,
                    game=self.game,
                )
                safety = safety_verifier(x_t, game=self.game)
            repaired = True

        rho_spent = max(0.0, (self.v_ref - self.eps_safe) - safety.value)
        self.probe_budget.charge(rho_spent)

        return Decision(
            realization=x_t,
            behavior=self._sf0.behavior_from_realization(x_t),
            robust_value=response.robust_value,
            safety_value=safety.value,
            repaired=repaired,
            mean_ci_width=mean_ci_width,
            ci_width_by_infoset=ci_width_by_infoset,
            info_gain=information_gain(coeffs, x_t),
            rho_granted=rho,
            rho_spent=rho_spent,
        )

    def observe(self, p2_counts: Mapping[str, Sequence[int]]) -> None:
        ""
        for label, counts in p2_counts.items():
            self.evidence.record(label, counts)


__all__ = ["OnlineSafeExploitAgent", "Decision"]
