"""Public interfaces for solvers. See The Safe Observation-Capacity Frontier, Certified Value Recovery, and supplementary Certification at the Unbucketed River."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import os
from time import perf_counter

from .. import native
from ..sequence_form import SequenceForm, compile as compile_game

_TIMER_T0 = perf_counter()


def _timers_enabled() -> bool:
    """Compute timers enabled for the init workflow."""
    value = os.environ.get(
        "SAFE_OBSERVATION_PY_TIMERS", os.environ.get("SAFE_OBSERVATION_TIMERS", "0")
    )
    return value not in {"", "0", "false", "False", "FALSE"}


def _timer(label: str, start: float | None = None, **fields) -> float:
    """Emit an optional timing record and return a timestamp."""
    now = perf_counter()
    if _timers_enabled():
        parts = [f"[timer py-solver +{now - _TIMER_T0:8.3f}s]", label]
        if start is not None:
            parts.append(f"elapsed={now - start:.3f}s")
        parts.extend(f"{key}={value}" for key, value in fields.items())
        print(" ".join(parts), flush=True)
    return now


@dataclass
class BlueprintSolution:
    """Represent blueprint solution for the init workflow."""

    value: float
    strategy: dict[str, list[float]] = field(default_factory=dict)
    realization: tuple[float, ...] | None = None
    method: str = "lp"


@dataclass
class SafetyResult:
    """Store safety result values for the init workflow."""

    value: float
    best_response: tuple[float, ...]

    def is_safe(self, v_ref: float, eps_safe: float = 0.0, tol: float = 1e-8) -> bool:
        """Return whether safe for the init workflow."""
        return self.value >= v_ref - eps_safe - tol


@dataclass
class RobustSafeResponse:
    """Represent robust safe response for the init workflow."""

    robust_value: float
    realization: tuple[float, ...]
    v_ref: float
    eps_safe: float
    game: str = "kuhn"

    def safety_value(self) -> float:
        """Compute the safety value for the init workflow."""
        return safety_verifier(self.realization, game=self.game).value

    def is_safe(self, tol: float = 1e-8) -> bool:
        """Return whether safe for the init workflow."""
        return self.safety_value() >= self.v_ref - self.eps_safe - tol


@dataclass
class ProbeResponse:
    """Represent probe response for the init workflow."""

    robust_value: float
    realization: tuple[float, ...]
    v_ref: float
    eps_safe: float
    beta: float
    rho: float
    game: str = "kuhn"

    def safety_value(self) -> float:
        """Compute the safety value for the init workflow."""
        return safety_verifier(self.realization, game=self.game).value

    def rho_spent(self) -> float:
        """Compute rho spent for the init workflow."""
        return max(0.0, (self.v_ref - self.eps_safe) - self.safety_value())

    def is_within_budget(self, tol: float = 1e-8) -> bool:
        """Return whether within budget for the init workflow."""
        return self.safety_value() >= self.v_ref - self.eps_safe - self.rho - tol


def _behavior_from_realization(
    sf: SequenceForm, x: Sequence[float], tol: float = 1e-12
) -> dict[str, list[float]]:
    """Compute behavior from realization for the init workflow."""
    return sf.behavior_from_realization(x, tol=tol)


def solve_blueprint(
    game: str = "kuhn",
    method: str = "lp",
    iterations: int = 100_000,
    seed: int = 2026,
) -> BlueprintSolution:
    """Solve blueprint for the init workflow."""
    if method == "lp":
        t0 = _timer("solve_blueprint:lp_native:start", game=game)
        value, realization = native.blueprint_lp(game)
        _timer("solve_blueprint:lp_native:done", t0, game=game, value=f"{value:.6g}")
        t0 = _timer("solve_blueprint:behavior_from_realization:start", game=game)
        strategy = _behavior_from_realization(compile_game(game, 0), realization)
        _timer(
            "solve_blueprint:behavior_from_realization:done",
            t0,
            game=game,
            infosets=len(strategy),
        )
        return BlueprintSolution(
            value=value,
            strategy=strategy,
            realization=tuple(realization),
            method="lp",
        )
    if method == "cfr":
        if game != "kuhn":
            raise NotImplementedError(f"CFR solver is Kuhn-only, not {game!r}.")
        value, strategy = native.solve_kuhn(iterations)
        return BlueprintSolution(value=value, strategy=strategy, method="cfr")
    if method == "mccfr":
        t0 = _timer(
            "solve_blueprint:mccfr_native:start",
            game=game,
            iterations=iterations,
            seed=seed,
        )
        value, behavior = native.blueprint_mccfr(game, iterations, seed)
        _timer("solve_blueprint:mccfr_native:done", t0, game=game, value=f"{value:.6g}")
        t0 = _timer("solve_blueprint:mccfr_realization:start", game=game)
        realization = compile_game(game, 0).realization_from_behavior(
            {label: list(dist) for label, dist in behavior.items()}
        )
        _timer(
            "solve_blueprint:mccfr_realization:done",
            t0,
            game=game,
            sequences=len(realization),
        )
        return BlueprintSolution(
            value=value,
            strategy={label: list(dist) for label, dist in behavior.items()},
            realization=tuple(realization),
            method="mccfr",
        )
    raise ValueError(f"unknown method {method!r}; expected 'lp', 'cfr', or 'mccfr'")


def safety_verifier(x: Sequence[float], game: str = "kuhn") -> SafetyResult:
    """Compute safety verifier for the init workflow."""
    value, best_response = native.safety_verify(game, list(x))
    return SafetyResult(value=value, best_response=tuple(best_response))


@dataclass
class BestResponse:
    """Represent best response for the init workflow."""

    value: float
    realization: tuple[float, ...]


def best_response(y: Sequence[float], game: str = "kuhn") -> BestResponse:
    """Compute best response for the init workflow."""
    value, realization = native.best_response(game, list(y))
    return BestResponse(value=value, realization=tuple(realization))


@dataclass
class RestrictedNashResponse:
    """Represent restricted Nash response for the init workflow."""

    value: float
    realization: tuple[float, ...]
    p: float
    game: str = "kuhn"

    def safety_value(self) -> float:
        """Compute the safety value for the init workflow."""
        return safety_verifier(self.realization, game=self.game).value


def restricted_nash_response(
    y_fix: Sequence[float], p: float, game: str = "kuhn"
) -> RestrictedNashResponse:
    """Compute restricted Nash response for the init workflow."""
    value, realization = native.restricted_nash_response(game, list(y_fix), p)
    return RestrictedNashResponse(
        value=value, realization=tuple(realization), p=p, game=game
    )


@dataclass
class SafetyFilteredRNRResponse:
    """Represent safety filtered restricted Nash response response."""

    value: float
    realization: tuple[float, ...]
    p: float
    safety_value: float
    game: str = "kuhn"


def safety_filtered_restricted_nash_response(
    y_fix: Sequence[float],
    floor: float,
    game: str = "kuhn",
    iterations: int = 8,
    p_max: float = 1.0,
) -> SafetyFilteredRNRResponse:
    """Compute safety filtered restricted Nash response."""
    tol = 1e-9

    def solve(p: float) -> tuple[RestrictedNashResponse, float]:
        """Solve the configured optimization problem."""
        r = restricted_nash_response(y_fix, p, game=game)
        s = safety_verifier(r.realization, game=game).value
        return r, s

    best_r, best_s = solve(0.0)
    best_p = 0.0

    r_hi, s_hi = solve(p_max)
    if s_hi >= floor - tol:
        return SafetyFilteredRNRResponse(
            value=r_hi.value,
            realization=r_hi.realization,
            p=p_max,
            safety_value=s_hi,
            game=game,
        )
    # Safety is monotone along the restricted-response mixing path, so
    # bisection finds the most exploitative admissible mixture coefficient.
    lo, hi = 0.0, p_max
    for _ in range(iterations):
        mid = 0.5 * (lo + hi)
        r_m, s_m = solve(mid)
        if s_m >= floor - tol:
            lo, best_p, best_r, best_s = mid, mid, r_m, s_m
        else:
            hi = mid
    return SafetyFilteredRNRResponse(
        value=best_r.value,
        realization=best_r.realization,
        p=best_p,
        safety_value=best_s,
        game=game,
    )


@dataclass
class SafetyConstrainedBestResponse:
    """Represent safety constrained best response."""

    value: float
    realization: tuple[float, ...]
    v_ref: float
    eps_safe: float
    game: str = "kuhn"


def safety_constrained_best_response(
    opponent_behavior: Mapping[str, Sequence[float]],
    v_ref: float | None = None,
    eps_safe: float = 0.0,
    game: str = "kuhn",
) -> SafetyConstrainedBestResponse:
    """Compute safety constrained best response."""
    if v_ref is None:
        v_ref = solve_blueprint(game, method="lp").value
    sf1 = compile_game(game, 1)
    labels = {info.label for info in sf1.info_sets}
    if set(opponent_behavior) != labels:
        # Partial policies are routed through the robust point-set interface,
        # which retains unspecified fibers instead of inventing behavior.
        point_intervals = {
            label: [(p, p) for p in dist] for label, dist in opponent_behavior.items()
        }
        response = robust_safe_response(
            point_intervals, v_ref=v_ref, eps_safe=eps_safe, game=game
        )
        return SafetyConstrainedBestResponse(
            value=response.robust_value,
            realization=response.realization,
            v_ref=v_ref,
            eps_safe=eps_safe,
            game=game,
        )
    behavior = {label: list(dist) for label, dist in opponent_behavior.items()}
    t0 = _timer("safety_constrained_best_response:native:start", game=game)
    value, realization = native.safety_constrained_best_response(
        game, behavior, v_ref, eps_safe
    )
    _timer(
        "safety_constrained_best_response:native:done",
        t0,
        game=game,
        value=f"{value:.6g}",
    )
    return SafetyConstrainedBestResponse(
        value=value,
        realization=tuple(realization),
        v_ref=v_ref,
        eps_safe=eps_safe,
        game=game,
    )


def robust_safe_response(
    intervals: Mapping[str, Sequence[tuple[float, float]]],
    v_ref: float | None = None,
    eps_safe: float = 0.0,
    game: str = "kuhn",
) -> RobustSafeResponse:
    """Compute robust safe response for the init workflow."""
    if v_ref is None:
        v_ref = solve_blueprint(game, method="lp").value
    t0 = _timer("robust_safe_response:payload:start", game=game)
    payload = {label: [tuple(b) for b in bounds] for label, bounds in intervals.items()}
    _timer("robust_safe_response:payload:done", t0, game=game, keys=len(payload))
    t0 = _timer("robust_safe_response:native:start", game=game)
    value, realization = native.robust_safe_response(game, payload, v_ref, eps_safe)
    _timer("robust_safe_response:native:done", t0, game=game, value=f"{value:.6g}")
    return RobustSafeResponse(
        robust_value=value,
        realization=tuple(realization),
        v_ref=v_ref,
        eps_safe=eps_safe,
        game=game,
    )


def floor_shadow_price(
    intervals: Mapping[str, Sequence[tuple[float, float]]],
    v_ref: float | None = None,
    eps_safe: float = 0.0,
    delta_rho: float = 0.05,
    game: str = "kuhn",
) -> float:
    """Compute floor shadow price for the init workflow."""
    if v_ref is None:
        v_ref = solve_blueprint(game, method="lp").value
    j0 = robust_safe_response(
        intervals, v_ref=v_ref, eps_safe=eps_safe, game=game
    ).robust_value
    jd = robust_safe_response(
        intervals, v_ref=v_ref, eps_safe=eps_safe + delta_rho, game=game
    ).robust_value
    return (jd - j0) / delta_rho


def robust_safe_response_public(
    groups: Mapping[str, Sequence[str]],
    intervals: Mapping[str, Sequence[tuple[float, float]]],
    v_ref: float | None = None,
    eps_safe: float = 0.0,
    game: str = "kuhn",
    weights: Mapping[str, float] | None = None,
) -> RobustSafeResponse:
    """Compute robust safe response public."""
    if v_ref is None:
        v_ref = solve_blueprint(game, method="lp").value
    t0 = _timer("robust_safe_response_public:payload:start", game=game)
    grp = {key: list(labels) for key, labels in groups.items()}
    payload = {key: [tuple(b) for b in bounds] for key, bounds in intervals.items()}
    w = dict(weights) if weights is not None else None
    _timer(
        "robust_safe_response_public:payload:done",
        t0,
        game=game,
        groups=len(grp),
        intervals=len(payload),
        weights=0 if w is None else len(w),
    )
    t0 = _timer("robust_safe_response_public:native:start", game=game)
    value, realization = native.robust_safe_response_public(
        game, grp, payload, v_ref, eps_safe, w
    )
    _timer(
        "robust_safe_response_public:native:done", t0, game=game, value=f"{value:.6g}"
    )
    return RobustSafeResponse(
        robust_value=value,
        realization=tuple(realization),
        v_ref=v_ref,
        eps_safe=eps_safe,
        game=game,
    )


def robust_safe_response_public_cutting_plane(
    groups: Mapping[str, Sequence[str]],
    intervals: Mapping[str, Sequence[tuple[float, float]]],
    v_ref: float | None = None,
    eps_safe: float = 0.0,
    game: str = "kuhn",
    weights: Mapping[str, float] | None = None,
    max_iters: int = 64,
    tol: float = 1e-7,
) -> RobustSafeResponse:
    """Compute robust safe response public cutting plane."""
    if v_ref is None:
        v_ref = solve_blueprint(game, method="lp").value
    grp = {key: list(labels) for key, labels in groups.items()}
    payload = {key: [tuple(b) for b in bounds] for key, bounds in intervals.items()}
    w = dict(weights) if weights is not None else None
    t0 = _timer("robust_safe_response_public_cutting_plane:native:start", game=game)
    value, realization = native.robust_safe_response_public_cutting_plane(
        game, grp, payload, v_ref, eps_safe, w, int(max_iters), float(tol)
    )
    _timer(
        "robust_safe_response_public_cutting_plane:native:done",
        t0,
        game=game,
        value=f"{value:.6g}",
    )
    return RobustSafeResponse(
        robust_value=value,
        realization=tuple(realization),
        v_ref=v_ref,
        eps_safe=eps_safe,
        game=game,
    )


def robust_safe_response_envelope(
    groups: Mapping[str, Sequence[str]],
    public_intervals: Mapping[str, Sequence[tuple[float, float]]],
    box_intervals: Mapping[str, Sequence[tuple[float, float]]],
    v_ref: float | None = None,
    eps_safe: float = 0.0,
    game: str = "kuhn",
    weights: Mapping[str, float] | None = None,
) -> RobustSafeResponse:
    """Compute robust safe response envelope."""
    if v_ref is None:
        v_ref = solve_blueprint(game, method="lp").value
    grp = {key: list(labels) for key, labels in groups.items()}
    pub = {key: [tuple(b) for b in bounds] for key, bounds in public_intervals.items()}
    box = {key: [tuple(b) for b in bounds] for key, bounds in box_intervals.items()}
    w = dict(weights) if weights is not None else None
    value, realization = native.robust_safe_response_envelope(
        game, grp, pub, box, v_ref, eps_safe, w
    )
    return RobustSafeResponse(
        robust_value=value,
        realization=tuple(realization),
        v_ref=v_ref,
        eps_safe=eps_safe,
        game=game,
    )


def robust_safe_response_obs(
    groups: Mapping[str, Sequence[str]],
    public_intervals: Mapping[str, Sequence[tuple[float, float]]],
    box_intervals: Mapping[str, Sequence[tuple[float, float]]],
    v_ref: float | None = None,
    eps_safe: float = 0.0,
    game: str = "kuhn",
    weights: Mapping[str, float] | None = None,
) -> RobustSafeResponse:
    """Compute robust safe response obs."""
    if v_ref is None:
        v_ref = solve_blueprint(game, method="lp").value
    t0 = _timer("robust_safe_response_obs:payload:start", game=game)
    grp = {key: list(labels) for key, labels in groups.items()}
    pub = {key: [tuple(b) for b in bounds] for key, bounds in public_intervals.items()}
    box = {key: [tuple(b) for b in bounds] for key, bounds in box_intervals.items()}
    w = dict(weights) if weights is not None else None
    _timer(
        "robust_safe_response_obs:payload:done",
        t0,
        game=game,
        groups=len(grp),
        public=len(pub),
        boxes=len(box),
        weights=0 if w is None else len(w),
    )
    t0 = _timer("robust_safe_response_obs:native:start", game=game)
    value, realization = native.robust_safe_response_obs(
        game, grp, pub, box, v_ref, eps_safe, w
    )
    _timer("robust_safe_response_obs:native:done", t0, game=game, value=f"{value:.6g}")
    return RobustSafeResponse(
        robust_value=value,
        realization=tuple(realization),
        v_ref=v_ref,
        eps_safe=eps_safe,
        game=game,
    )


def robust_safe_response_linear(
    groups: Mapping[str, Sequence[str]],
    public_intervals: Mapping[str, Sequence[tuple[float, float]]],
    event_entries: Sequence[tuple[int, int, float]],
    event_h: Sequence[float],
    v_ref: float | None = None,
    eps_safe: float = 0.0,
    game: str = "kuhn",
    weights: Mapping[str, float] | None = None,
    row_meta: Sequence[tuple[str, int]] | None = None,
) -> RobustSafeResponse:
    """Compute robust safe response linear."""
    if v_ref is None:
        v_ref = solve_blueprint(game, method="lp").value
    t0 = _timer("robust_safe_response_linear:payload:start", game=game)
    grp = {key: list(labels) for key, labels in groups.items()}
    pub = {key: [tuple(b) for b in bounds] for key, bounds in public_intervals.items()}
    entries = [(int(r), int(c), float(v)) for r, c, v in event_entries]
    h = [float(v) for v in event_h]
    w = dict(weights) if weights is not None else None
    meta = list(row_meta) if row_meta is not None else None
    _timer(
        "robust_safe_response_linear:payload:done",
        t0,
        game=game,
        groups=len(grp),
        public=len(pub),
        rows=len(h),
        nnz=len(entries),
        weights=0 if w is None else len(w),
    )
    t0 = _timer("robust_safe_response_linear:native:start", game=game)
    value, realization = native.robust_safe_response_linear(
        game,
        grp,
        pub,
        entries,
        h,
        v_ref,
        eps_safe,
        w,
        meta,
    )
    _timer(
        "robust_safe_response_linear:native:done", t0, game=game, value=f"{value:.6g}"
    )
    return RobustSafeResponse(
        robust_value=value,
        realization=tuple(realization),
        v_ref=v_ref,
        eps_safe=eps_safe,
        game=game,
    )


def robust_safe_response_linear_cutting_plane(
    groups: Mapping[str, Sequence[str]],
    public_intervals: Mapping[str, Sequence[tuple[float, float]]],
    event_entries: Sequence[tuple[int, int, float]],
    event_h: Sequence[float],
    v_ref: float | None = None,
    eps_safe: float = 0.0,
    game: str = "kuhn",
    weights: Mapping[str, float] | None = None,
    row_meta: Sequence[tuple[str, int]] | None = None,
    max_iters: int = 64,
    tol: float = 1e-7,
) -> RobustSafeResponse:
    """Compute robust safe response linear cutting plane."""
    if v_ref is None:
        v_ref = solve_blueprint(game, method="lp").value
    grp = {key: list(labels) for key, labels in groups.items()}
    pub = {key: [tuple(b) for b in bounds] for key, bounds in public_intervals.items()}
    entries = [(int(r), int(c), float(v)) for r, c, v in event_entries]
    h = [float(v) for v in event_h]
    w = dict(weights) if weights is not None else None
    meta = list(row_meta) if row_meta is not None else None
    t0 = _timer("robust_safe_response_linear_cutting_plane:native:start", game=game)
    value, realization = native.robust_safe_response_linear_cutting_plane(
        game,
        grp,
        pub,
        entries,
        h,
        v_ref,
        eps_safe,
        w,
        meta,
        int(max_iters),
        float(tol),
    )
    _timer(
        "robust_safe_response_linear_cutting_plane:native:done",
        t0,
        game=game,
        value=f"{value:.6g}",
    )
    return RobustSafeResponse(
        robust_value=value,
        realization=tuple(realization),
        v_ref=v_ref,
        eps_safe=eps_safe,
        game=game,
    )


def opponent_reach_weights(
    x_agent: Sequence[float],
    game: str = "kuhn",
) -> dict[str, float]:
    """Compute weights for opponent reach."""
    return dict(native.opponent_reach_weights(game, list(x_agent)))


def agent_showdown_reach(
    x_agent: Sequence[float],
    game: str = "kuhn",
) -> dict[str, list[tuple[float, bool]]]:
    """Compute agent showdown reach for the init workflow."""
    return {
        label: [(float(w), bool(c)) for w, c in row]
        for label, row in native.agent_showdown_reach(game, list(x_agent)).items()
    }


@dataclass(frozen=True)
class ConfidenceGuardedResponse:
    """Represent confidence guarded response for the init workflow."""

    point_value: float
    realization: tuple[float, ...]
    j_rho: float
    kappa: float
    v_ref: float
    eps_safe: float
    rho: float
    game: str = "kuhn"

    def safety_value(self) -> float:
        """Compute the safety value for the init workflow."""
        return safety_verifier(self.realization, game=self.game).value


def confidence_guarded_point_probe(
    groups: Mapping[str, Sequence[str]],
    intervals: Mapping[str, Sequence[tuple[float, float]]],
    y_hat: Sequence[float],
    v_ref: float | None = None,
    eps_safe: float = 0.0,
    rho: float = 0.0,
    kappa: float = 0.0,
    game: str = "kuhn",
    weights: Mapping[str, float] | None = None,
) -> ConfidenceGuardedResponse:
    """Compute confidence guarded point probe."""
    if v_ref is None:
        v_ref = solve_blueprint(game, method="lp").value
    grp = {key: list(labels) for key, labels in groups.items()}
    payload = {key: [tuple(b) for b in bounds] for key, bounds in intervals.items()}
    w = dict(weights) if weights is not None else None
    j_rho = robust_safe_response_public(
        grp, payload, v_ref=v_ref, eps_safe=eps_safe + rho, game=game, weights=w
    ).robust_value
    # The point response may exploit the empirical model only while remaining
    # within kappa of the confidence-set certificate at the relaxed floor.
    guard_rhs = j_rho - kappa
    point_value, realization = native.confidence_guarded_point_probe(
        game, grp, payload, list(y_hat), v_ref, eps_safe, rho, guard_rhs, w
    )
    return ConfidenceGuardedResponse(
        point_value=point_value,
        realization=tuple(realization),
        j_rho=j_rho,
        kappa=kappa,
        v_ref=v_ref,
        eps_safe=eps_safe,
        rho=rho,
        game=game,
    )


def probe_coefficients(
    opp_behavior: Mapping[str, Sequence[float]],
    weights: Mapping[str, float],
    game: str = "kuhn",
) -> tuple[float, ...]:
    """Compute probe coefficients for the init workflow."""
    ob = {label: list(dist) for label, dist in opp_behavior.items()}
    w = {label: float(value) for label, value in weights.items()}
    return tuple(native.probe_coefficients(game, ob, w))


def confidence_sensitivity(
    intervals: Mapping[str, Sequence[tuple[float, float]]],
    v_ref: float | None = None,
    eps_safe: float = 0.0,
    game: str = "kuhn",
) -> dict[str, list[float]]:
    """Compute confidence sensitivity for the init workflow."""
    if v_ref is None:
        v_ref = solve_blueprint(game, method="lp").value
    payload = {label: [tuple(b) for b in bounds] for label, bounds in intervals.items()}
    return {
        label: list(row)
        for label, row in native.confidence_sensitivity(
            game, payload, v_ref, eps_safe
        ).items()
    }


def robust_safe_response_probe(
    intervals: Mapping[str, Sequence[tuple[float, float]]],
    opp_behavior: Mapping[str, Sequence[float]],
    weights: Mapping[str, float],
    v_ref: float | None = None,
    eps_safe: float = 0.0,
    beta: float = 0.0,
    rho: float = 0.0,
    game: str = "kuhn",
) -> ProbeResponse:
    """Compute robust safe response probe."""
    if v_ref is None:
        v_ref = solve_blueprint(game, method="lp").value
    iv = {label: [tuple(b) for b in bounds] for label, bounds in intervals.items()}
    ob = {label: list(dist) for label, dist in opp_behavior.items()}
    w = {label: float(value) for label, value in weights.items()}
    value, realization = native.robust_safe_response_probe(
        game, iv, ob, w, v_ref, eps_safe, beta, rho
    )
    return ProbeResponse(
        robust_value=value,
        realization=tuple(realization),
        v_ref=v_ref,
        eps_safe=eps_safe,
        beta=beta,
        rho=rho,
        game=game,
    )


def fallback_mixture_repair(
    x_blueprint: Sequence[float],
    x_candidate: Sequence[float],
    v_ref: float,
    eps_safe: float = 0.0,
    game: str = "kuhn",
    iterations: int = 50,
    tol: float = 1e-9,
) -> tuple[float, ...]:
    """Compute fallback mixture repair for the init workflow."""
    floor = v_ref - eps_safe

    def mix(alpha: float) -> tuple[float, ...]:
        """Mix the blueprint and candidate realization plans."""
        return tuple(
            (1.0 - alpha) * b + alpha * c
            for b, c in zip(x_blueprint, x_candidate, strict=True)
        )

    def safe(alpha: float) -> bool:
        """Return whether the candidate mixture satisfies the safety floor."""
        return safety_verifier(mix(alpha), game=game).value >= floor - tol

    if safe(1.0):
        return mix(1.0)
    # The blueprint endpoint is safe; bisection keeps the largest feasible
    # fraction of the candidate response.
    lo, hi = 0.0, 1.0
    for _ in range(iterations):
        mid = 0.5 * (lo + hi)
        if safe(mid):
            lo = mid
        else:
            hi = mid
    return mix(lo)


__all__ = [
    "BlueprintSolution",
    "SafetyResult",
    "BestResponse",
    "RestrictedNashResponse",
    "SafetyFilteredRNRResponse",
    "SafetyConstrainedBestResponse",
    "RobustSafeResponse",
    "ConfidenceGuardedResponse",
    "ProbeResponse",
    "solve_blueprint",
    "safety_verifier",
    "best_response",
    "restricted_nash_response",
    "safety_filtered_restricted_nash_response",
    "safety_constrained_best_response",
    "robust_safe_response",
    "floor_shadow_price",
    "robust_safe_response_public",
    "robust_safe_response_public_cutting_plane",
    "robust_safe_response_envelope",
    "robust_safe_response_obs",
    "robust_safe_response_linear",
    "robust_safe_response_linear_cutting_plane",
    "opponent_reach_weights",
    "confidence_guarded_point_probe",
    "probe_coefficients",
    "confidence_sensitivity",
    "robust_safe_response_probe",
    "fallback_mixture_repair",
]
