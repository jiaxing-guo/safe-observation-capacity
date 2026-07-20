""

import json
import math
import os
from pathlib import Path

from safe_observation import native
from safe_observation.confidence import OpponentEvidenceStore
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
EPISODES = int(os.environ.get("KNEE_EPISODES", "150000"))
SEED = 2026
DELTA = 0.1
METHOD = "empirical_bernstein"
RHO_CAP = 0.5

RADII = [0.0, 0.02, 0.05, 0.08, 0.12, 0.18, 0.25, 0.35, 0.5, 0.7, 1.0]
OUT = Path("results/envelope_knee_holdem.json")


FOCUS = {
    "ambiguous_fold_strong": "twin (gap 0.170)",
    "low_reach_private": "low-reach (gap 0.143)",
    "mixed_public_private": "mixed (gap 0.115)",
    "board_polarized_maniac": "EM-collapse stressor",
    "board_bluffcatcher_station": "private-het",
    "board_public_call": "public-homogeneous ref",
    "near_equilibrium": "control",
}


def _population_stores(
    game: str,
    agent_behavior: dict[str, list[float]],
    opp_behavior: dict[str, list[float]],
) -> tuple[OpponentEvidenceStore, OpponentEvidenceStore]:
    ""
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


def _box_intervals(
    center: dict[str, list[float]], radius: float, labels: set[str]
) -> dict[str, list[tuple[float, float]]]:
    ""
    box: dict[str, list[tuple[float, float]]] = {}
    for label, dist in center.items():
        if label not in labels:
            continue
        box[label] = [(max(0.0, p - radius), min(1.0, p + radius)) for p in dist]
    return box


def _covering_radius(
    center: dict[str, list[float]],
    behavior: dict[str, list[float]],
    omega: dict[str, float],
) -> float:
    ""
    d = 0.0
    for label, dist in behavior.items():
        if omega.get(label, 0.0) <= 0.0:
            continue
        ref = center.get(label)
        if ref is None:
            continue
        d = max(d, max(abs(p - q) for p, q in zip(dist, ref, strict=True)))
    return d


def _sweep_family(
    *,
    game: str,
    center: dict[str, list[float]],
    groups,
    pub_intervals,
    omega: dict[str, float],
    labels: set[str],
    v_ref: float,
    payoff,
    y_star: list[float],
    cover_r: float,
) -> list[dict]:
    cells: list[dict] = []
    for r in RADII:
        box = _box_intervals(center, r, labels)
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
            cells.append({"r": r, "feasible": False})
            continue
        x = list(resp.realization)
        cells.append(
            {
                "r": r,
                "feasible": True,
                "covers": r + 1e-9 >= cover_r,
                "ell_gain": resp.robust_value - v_ref,
                "real_gain": payoff.bilinear(x, y_star) - v_ref,
                "min_safety": safety_verifier(x, game=game).value,
            }
        )
    return cells


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
        f"# envelope knee/cliff  game={game}  v_ref={v_ref:+.5f}  floor={floor:+.4f}  "
        f"episodes={EPISODES}\n"
    )

    checks_ok = True
    for name, opponent in suite.items():
        behavior = opponent.behavior
        y_star = list(opponent.realization())
        ev_point, ev_public = _population_stores(game, agent_behavior, behavior)
        groups = ev_public.public_groups()
        pub_intervals = ev_public.public_intervals(DELTA, method=METHOD)
        labels = {label for label in ev_public.labels if omega.get(label, 0.0) > 0.0}

        p_pub = _public_point_behavior(ev_public)
        p_em = _censored_em_behavior(game, ev_point, ev_public, omega)

        cover_het = _covering_radius(p_pub, behavior, omega)
        cover_em = _covering_radius(p_em, behavior, omega)

        het = _sweep_family(
            game=game,
            center=p_pub,
            groups=groups,
            pub_intervals=pub_intervals,
            omega=omega,
            labels=labels,
            v_ref=v_ref,
            payoff=payoff,
            y_star=y_star,
            cover_r=cover_het,
        )
        em = _sweep_family(
            game=game,
            center=p_em,
            groups=groups,
            pub_intervals=pub_intervals,
            omega=omega,
            labels=labels,
            v_ref=v_ref,
            payoff=payoff,
            y_star=y_star,
            cover_r=cover_em,
        )

        x_core = robust_safe_response_public(
            groups,
            pub_intervals,
            v_ref=v_ref,
            eps_safe=RHO_CAP,
            game=game,
            weights=omega,
        )
        core_gain = payoff.bilinear(list(x_core.realization), y_star) - v_ref
        x_pub_pt = safety_constrained_best_response(
            p_pub, v_ref=v_ref, eps_safe=RHO_CAP, game=game
        )
        pub_pt_gain = payoff.bilinear(list(x_pub_pt.realization), y_star) - v_ref
        het_rinf = next(c for c in reversed(het) if c["feasible"])
        het_r0 = het[0]
        core_match = math.isclose(het_rinf["real_gain"], core_gain, abs_tol=2e-3)
        r0_minus_point = (
            (het_r0["real_gain"] - pub_pt_gain) if het_r0["feasible"] else None
        )
        checks_ok = checks_ok and core_match

        oracle = safety_constrained_best_response(
            behavior, v_ref=v_ref, eps_safe=RHO_CAP, game=game
        )
        oracle_gain = oracle.value - v_ref

        rows.append(
            {
                "opponent": name,
                "v_ref": v_ref,
                "floor": floor,
                "oracle_gain": oracle_gain,
                "cover_radius": {"het": cover_het, "em": cover_em},
                "lambda_min": {
                    "het": next((c["r"] for c in het if c["feasible"]), None),
                    "em": next((c["r"] for c in em if c["feasible"]), None),
                },
                "het": het,
                "em": em,
                "honest_cert": {
                    "het": _honest_cert(het, cover_het),
                    "em": _honest_cert(em, cover_em),
                },
                "endpoint_check": {
                    "core_match": core_match,
                    "core_envelope_gain": het_rinf["real_gain"],
                    "core_standalone_gain": core_gain,
                    "point_arm_gain": pub_pt_gain,
                    "r0_minus_point_arm": r0_minus_point,
                },
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
                "rows": rows,
            },
            indent=2,
        )
    )
    _print_focus(rows)
    print(
        f"\nendpoint self-check (r>=1 envelope == standalone core, exact): "
        f"{'PASS' if checks_ok else 'FAIL'}"
    )
    print(f"wrote {OUT}")


def _honest_cert(cells: list[dict], cover_r: float) -> dict:
    ""
    honest_r = None
    honest_ell = None
    for c in cells:
        if c["feasible"] and c.get("covers"):
            honest_r, honest_ell = c["r"], c["ell_gain"]
            break
    feas = [c for c in cells if c["feasible"]]
    best = max(feas, key=lambda c: c["real_gain"], default=None)
    return {
        "honest_r": honest_r,
        "honest_ell": honest_ell,
        "best_real": best["real_gain"] if best else None,
        "best_real_r": best["r"] if best else None,
    }


def _fmt_curve(cells: list[dict], key: str) -> str:
    parts = []
    for c in cells:
        if not c["feasible"]:
            parts.append("  inf ")
        else:
            mark = "*" if c.get("covers") else " "
            parts.append(f"{c[key]:+5.2f}{mark}")
    return " ".join(parts)


def _print_focus(rows: list[dict]) -> None:
    by_name = {r["opponent"]: r for r in rows}
    rstr = " ".join(f"{r:>5.2f} " for r in RADII)
    for name, note in FOCUS.items():
        r = by_name.get(name)
        if r is None:
            continue
        print(
            f"\n=== {name}  ({note})  oracle={r['oracle_gain']:+.3f}  "
            f"floor={r['floor']:+.3f} ==="
        )
        print(
            f"  r ->          {rstr}   (cover_het={r['cover_radius']['het']:.2f} "
            f"cover_em={r['cover_radius']['em']:.2f})"
        )
        print(f"  het ell:      {_fmt_curve(r['het'], 'ell_gain')}")
        print(f"  het real:     {_fmt_curve(r['het'], 'real_gain')}")
        print(f"  em  ell:      {_fmt_curve(r['em'], 'ell_gain')}")
        print(f"  em  real:     {_fmt_curve(r['em'], 'real_gain')}")
        mins = [c["min_safety"] for c in r["het"] + r["em"] if c["feasible"]]
        print(
            f"  min_safety over all feasible cells: {min(mins):+.4f} "
            f"(floor {r['floor']:+.4f})  '*'=box covers y*"
        )
        hc = r["honest_cert"]
        for fam in ("het", "em"):
            h = hc[fam]
            hr = "none" if h["honest_r"] is None else f"{h['honest_r']:.2f}"
            he = "  n/a" if h["honest_ell"] is None else f"{h['honest_ell']:+.3f}"
            br = "  n/a" if h["best_real"] is None else f"{h['best_real']:+.3f}"
            brr = "none" if h["best_real_r"] is None else f"{h['best_real_r']:.2f}"
            print(
                f"  {fam}: honest cert ell={he} @ r>={hr} (covers y*) | "
                f"best REALIZED={br} @ r={brr} (oracle-annotated)"
            )


if __name__ == "__main__":
    main()
