""

import json
import os
from pathlib import Path

from safe_observation import native
from safe_observation.confidence import (
    OpponentEvidenceStore,
    empirical_bernstein_halfwidth,
)
from safe_observation.experiments.online import (
    _censored_em_behavior,
    _public_point_behavior,
)
from safe_observation.opponents import holdem_structured_opponent_suite
from safe_observation.payoff import build as build_payoff
from safe_observation.sequence_form import compile as compile_game
from safe_observation.solvers import (
    opponent_reach_weights,
    robust_safe_response_envelope,
    robust_safe_response_public,
    safety_constrained_best_response,
    safety_verifier,
    solve_blueprint,
)

GAME = "holdem"
EPISODES = int(os.environ.get("DESIGN_EPISODES", "150000"))
SEED = 2026
DELTA = 0.1
METHOD = "empirical_bernstein"
RHO_CAP = 0.5

RADII = [0.0, 0.02, 0.05, 0.08, 0.12, 0.18, 0.25, 0.35, 0.5, 0.7, 1.0]

SCALES = [0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
OUT = Path("results/envelope_design_holdem.json")

FOCUS = [
    "board_public_call",
    "board_bluffcatcher_station",
    "ambiguous_fold_strong",
    "mixed_public_private",
    "low_reach_private",
    "board_polarized_maniac",
    "near_equilibrium",
]


def _population_stores(
    game: str,
    agent_behavior: dict[str, list[float]],
    opp_behavior: dict[str, list[float]],
) -> tuple[OpponentEvidenceStore, OpponentEvidenceStore]:
    ev_point = OpponentEvidenceStore.for_game(game)
    ev_public = OpponentEvidenceStore.for_game(game)
    _pay, show, fold = native.simulate_showdown(
        game, agent_behavior, opp_behavior, EPISODES, SEED
    )
    for label, c in show.items():
        ev_point.record(label, c)
        ev_public.record(label, c)
    for label, c in fold.items():
        ev_public.record(label, c)
    return ev_point, ev_public


def _adaptive_radius(p: float, n: int, scale: float) -> float:
    ""
    if n <= 0:
        return 1.0
    return scale * empirical_bernstein_halfwidth(n, p * (1.0 - p), DELTA)


def _box_uniform_all(
    center: dict[str, list[float]], radius: float
) -> dict[str, list[tuple[float, float]]]:
    ""
    return {
        label: [(max(0.0, p - radius), min(1.0, p + radius)) for p in dist]
        for label, dist in center.items()
    }


def _box_adaptive(
    center: dict[str, list[float]],
    labels: set[str],
    support: dict[str, int],
    scale: float,
) -> dict[str, list[tuple[float, float]]]:
    ""
    box: dict[str, list[tuple[float, float]]] = {}
    for label, dist in center.items():
        if label not in labels:
            continue
        n = support.get(label, 0)
        box[label] = [
            (
                max(0.0, p - _adaptive_radius(p, n, scale)),
                min(1.0, p + _adaptive_radius(p, n, scale)),
            )
            for p in dist
        ]
    return box


def _covers_uniform(
    center: dict[str, list[float]],
    behavior: dict[str, list[float]],
    omega: dict[str, float],
    radius: float,
) -> bool:
    ""
    for label, dist in behavior.items():
        if omega.get(label, 0.0) <= 0.0:
            continue
        ref = center.get(label)
        if ref is None:
            continue
        if any(abs(p - q) > radius + 1e-9 for p, q in zip(dist, ref, strict=True)):
            return False
    return True


def _covers_adaptive(
    center: dict[str, list[float]],
    behavior: dict[str, list[float]],
    omega: dict[str, float],
    support: dict[str, int],
    scale: float,
) -> bool:
    ""
    for label, dist in behavior.items():
        if omega.get(label, 0.0) <= 0.0:
            continue
        ref = center.get(label)
        if ref is None:
            continue
        n = support.get(label, 0)
        for p, q in zip(dist, ref, strict=True):
            if abs(p - q) > _adaptive_radius(q, n, scale) + 1e-9:
                return False
    return True


def _solve(game, groups, pub_intervals, box, omega, payoff, y_star, v_ref):
    ""
    try:
        resp = robust_safe_response_envelope(
            groups,
            pub_intervals,
            box,
            v_ref=v_ref,
            eps_safe=RHO_CAP,
            game=game,
            weights=omega,
        )
    except ValueError:
        return None
    x = list(resp.realization)
    return (
        resp.robust_value - v_ref,
        payoff.bilinear(x, y_star) - v_ref,
        safety_verifier(x, game=game).value,
    )


def main() -> None:
    game = GAME
    payoff = build_payoff(game)
    sf0 = compile_game(game, 0)
    blueprint = solve_blueprint(game, method="lp")
    assert blueprint.realization is not None
    x_blueprint = blueprint.realization
    v_ref = blueprint.value
    agent_behavior = sf0.behavior_from_realization(x_blueprint)
    omega = opponent_reach_weights(x_blueprint, game=game)
    floor = v_ref - RHO_CAP

    suite = holdem_structured_opponent_suite(game)
    rows: list[dict] = []
    print(
        f"# envelope DESIGN study  game={game}  v_ref={v_ref:+.5f}  floor={floor:+.4f}  "
        f"episodes={EPISODES}\n"
    )

    for name, opponent in suite.items():
        behavior = opponent.behavior
        y_star = list(opponent.realization())
        ev_point, ev_public = _population_stores(game, agent_behavior, behavior)
        groups = ev_public.public_groups()
        pub_intervals = ev_public.public_intervals(DELTA, method=METHOD)
        labels = {label for label in ev_public.labels if omega.get(label, 0.0) > 0.0}
        support = {label: ev_point.visits(label) for label in ev_point.labels}

        p_pub = _public_point_behavior(ev_public)
        p_em = _censored_em_behavior(game, ev_point, ev_public, omega)

        core = robust_safe_response_public(
            groups,
            pub_intervals,
            v_ref=v_ref,
            eps_safe=RHO_CAP,
            game=game,
            weights=omega,
        )
        core_ell = core.robust_value - v_ref
        core_real = payoff.bilinear(list(core.realization), y_star) - v_ref
        scbr_pub = safety_constrained_best_response(
            p_pub, v_ref=v_ref, eps_safe=RHO_CAP, game=game
        )
        pub_arm = payoff.bilinear(list(scbr_pub.realization), y_star) - v_ref
        scbr_em = safety_constrained_best_response(
            p_em, v_ref=v_ref, eps_safe=RHO_CAP, game=game
        )
        em_arm = payoff.bilinear(list(scbr_em.realization), y_star) - v_ref
        oracle = safety_constrained_best_response(
            behavior, v_ref=v_ref, eps_safe=RHO_CAP, game=game
        )
        oracle_gain = oracle.value - v_ref

        design_a = []
        for r in RADII:
            res = _solve(
                game,
                groups,
                pub_intervals,
                _box_uniform_all(p_pub, r),
                omega,
                payoff,
                y_star,
                v_ref,
            )
            cell = {"r": r, "feasible": res is not None}
            if res is not None:
                cell.update(
                    ell_gain=res[0],
                    real_gain=res[1],
                    min_safety=res[2],
                    covers=_covers_uniform(p_pub, behavior, omega, r),
                )
            design_a.append(cell)

        design_b: dict[str, list[dict]] = {}
        for fam, center in (("het", p_pub), ("em", p_em)):
            cells = []
            for s in SCALES:
                res = _solve(
                    game,
                    groups,
                    pub_intervals,
                    _box_adaptive(center, labels, support, s),
                    omega,
                    payoff,
                    y_star,
                    v_ref,
                )
                cell = {"scale": s, "feasible": res is not None}
                if res is not None:
                    cell.update(
                        ell_gain=res[0],
                        real_gain=res[1],
                        min_safety=res[2],
                        covers=_covers_adaptive(center, behavior, omega, support, s),
                    )
                cells.append(cell)
            design_b[fam] = cells

        rows.append(
            {
                "opponent": name,
                "oracle_gain": oracle_gain,
                "core_ell": core_ell,
                "core_real": core_real,
                "pub_arm": pub_arm,
                "em_arm": em_arm,
                "design_a": design_a,
                "design_b": design_b,
            }
        )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps(
            {
                "game": game,
                "v_ref": v_ref,
                "floor": floor,
                "episodes": EPISODES,
                "seed": SEED,
                "delta": DELTA,
                "method": METHOD,
                "rho_cap": RHO_CAP,
                "radii": RADII,
                "scales": SCALES,
                "rows": rows,
            },
            indent=2,
        )
    )
    _report(rows, floor)
    print(f"\nwrote {OUT}")


def _honest_cert(
    cells: list[dict], key: str
) -> tuple[float | None, float | None, float]:
    ""
    for c in cells:
        if c.get("feasible") and c.get("covers"):
            return c.get("ell_gain"), c.get("real_gain"), c[key]
    return None, None, float("nan")


def _report(rows: list[dict], floor: float) -> None:
    by = {r["opponent"]: r for r in rows}

    print(
        "=== DESIGN A (uniform, ALL infosets pinned; r=0 should == standalone point arm) ==="
    )
    print(
        f"{'opponent':<28}{'pub_arm':>8}{'A@r=0':>8}{'match?':>8}{'A_best':>8}{'@r':>6}{'core':>7}"
    )
    for name in FOCUS:
        r = by.get(name)
        if not r:
            continue
        a = r["design_a"]
        a0 = next((c for c in a if c["r"] == 0.0), None)
        a0v = a0.get("real_gain") if a0 and a0["feasible"] else None
        feas = [c for c in a if c["feasible"]]
        best = max(feas, key=lambda c: c["real_gain"], default=None)
        match = (
            "n/a"
            if a0v is None
            else ("YES" if abs(a0v - r["pub_arm"]) < 5e-3 else "NO")
        )
        a0s = "  inf" if a0v is None else f"{a0v:+.2f}"
        bests = "n/a" if best is None else f"{best['real_gain']:+.2f}"
        bestr = "n/a" if best is None else f"{best['r']:.2f}"
        print(
            f"{name:<28}{r['pub_arm']:>8.2f}{a0s:>8}{match:>8}{bests:>8}{bestr:>6}{r['core_ell']:>7.2f}"
        )

    print("\n=== DESIGN B (adaptive radius; HONEST certificate vs core) ===")
    print(
        f"{'opponent':<28}{'oracle':>7}{'core_ell':>9}{'B_honEll':>9}{'@scale':>7}{'B_honReal':>10}{'cert_gain':>10}"
    )
    print(
        "  (cert_gain = honest certified ell ABOVE core; >0 => certifiable continuum)"
    )
    for name in FOCUS:
        r = by.get(name)
        if not r:
            continue

        best = None
        for fam in ("het", "em"):
            ell, real, sc = _honest_cert(r["design_b"][fam], "scale")
            if ell is not None and (best is None or ell > best[0]):
                best = (ell, real, sc, fam)
        if best is None:
            print(
                f"{name:<28}{r['oracle_gain']:>7.2f}{r['core_ell']:>9.3f}{'  none':>9}"
            )
            continue
        cert_gain = best[0] - r["core_ell"]
        print(
            f"{name:<28}{r['oracle_gain']:>7.2f}{r['core_ell']:>9.3f}{best[0]:>9.3f}"
            f"{best[2]:>7.1f}{best[1]:>10.3f}{cert_gain:>+10.3f}  [{best[3]}]"
        )

    print("\n=== DECISION SUMMARY ===")
    cert_gains = []
    for name in FOCUS:
        r = by.get(name)
        if not r:
            continue
        best = max(
            (
                _honest_cert(r["design_b"][fam], "scale")[0] or -9.0
                for fam in ("het", "em")
            ),
        )
        cert_gains.append((name, best - r["core_ell"]))
    if cert_gains:
        mx = max(cert_gains, key=lambda t: t[1])
        print(f"max honest cert-above-core = {mx[1]:+.3f} on {mx[0]}")
        print("KNEE (continuum certifiable) if this is solidly >0 on value opponents;")
        print("CLIFF (3-arm selector) if it stays ~0 (certificate flat at core).")


if __name__ == "__main__":
    main()
