""

import json
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
    robust_safe_response_public,
    safety_constrained_best_response,
    safety_verifier,
    solve_blueprint,
)

GAME = "holdem"
EPISODES = int(os.environ.get("GEOM_EPISODES", "150000"))
SEED = 2026
DELTA = 0.1
METHOD = "empirical_bernstein"
RHO_CAP = 0.5
OUT = Path("results/envelope_geometry_holdem.json")


REGIME = {
    "equilibrium": "control",
    "near_equilibrium": "control",
    "board_public_fold": "public-homogeneous",
    "board_public_call": "public-homogeneous",
    "board_marginal_overfold": "private-heterogeneous",
    "board_bluffcatcher_station": "private-heterogeneous",
    "board_polarized_maniac": "private-heterogeneous",
    "board_ambiguous_fold_marginal": "ambiguity-twin",
    "ambiguous_fold_strong": "ambiguity-twin",
    "showdown_selection_trap": "low-support/EM-stress",
    "mixed_public_private": "mixed",
    "low_reach_private": "low-reach",
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


def _oracle_radii(
    behavior: dict[str, list[float]],
    centres: dict[str, dict[str, list[float]]],
    omega: dict[str, float],
) -> dict[str, dict[str, float]]:
    ""
    out: dict[str, dict[str, float]] = {
        name: {"max": 0.0, "reach_mean": 0.0} for name in centres
    }
    wsum = 0.0
    acc = {name: 0.0 for name in centres}
    for label, dist in behavior.items():
        w = omega.get(label, 0.0)
        if w <= 0.0:
            continue
        wsum += w
        for name, centre in centres.items():
            ref = centre.get(label)
            if ref is None:
                continue
            dev = max(abs(p - q) for p, q in zip(dist, ref, strict=True))
            out[name]["max"] = max(out[name]["max"], dev)
            acc[name] += w * dev
    if wsum > 0.0:
        for name in centres:
            out[name]["reach_mean"] = acc[name] / wsum
    return out


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

    suite = holdem_structured_opponent_suite(game)
    rows: list[dict] = []
    print(
        f"# envelope geometry diagnostic  game={game}  v_ref={v_ref:+.5f}  "
        f"floor=F-{RHO_CAP}={v_ref - RHO_CAP:+.4f}  episodes={EPISODES}\n"
    )
    header = (
        f"{'opponent':<31}{'regime':<22}"
        f"{'d_het':>7}{'d_em':>7}"
        f"{'g_orcl':>8}{'g_core':>8}{'g_pub':>8}{'g_em':>8}{'g_nv':>8}"
        f"{'gap_core':>9}{'gap_em':>8}"
    )
    print(header)
    print("-" * len(header))

    for name, opponent in suite.items():
        behavior = opponent.behavior
        y_star = list(opponent.realization())

        ev_point, ev_public = _population_stores(game, agent_behavior, behavior)
        groups = ev_public.public_groups()
        pub_intervals = ev_public.public_intervals(DELTA, method=METHOD)

        p_pub = _public_point_behavior(ev_public)
        p_em = _censored_em_behavior(game, ev_point, ev_public, omega)
        p_naive = {label: list(ev_point.p_hat(label)) for label in ev_point.labels}

        radii = _oracle_radii(
            behavior, {"het": p_pub, "em": p_em, "naive": p_naive}, omega
        )

        x_core = robust_safe_response_public(
            groups,
            pub_intervals,
            v_ref=v_ref,
            eps_safe=RHO_CAP,
            game=game,
            weights=omega,
        ).realization
        x_pub = safety_constrained_best_response(
            p_pub, v_ref=v_ref, eps_safe=RHO_CAP, game=game
        ).realization
        x_em = safety_constrained_best_response(
            p_em, v_ref=v_ref, eps_safe=RHO_CAP, game=game
        ).realization
        x_naive = safety_constrained_best_response(
            p_naive, v_ref=v_ref, eps_safe=RHO_CAP, game=game
        ).realization
        oracle = safety_constrained_best_response(
            behavior, v_ref=v_ref, eps_safe=RHO_CAP, game=game
        )

        u_core = payoff.bilinear(list(x_core), y_star)
        u_pub = payoff.bilinear(list(x_pub), y_star)
        u_em = payoff.bilinear(list(x_em), y_star)
        u_naive = payoff.bilinear(list(x_naive), y_star)
        u_oracle = oracle.value

        min_safety = min(
            safety_verifier(list(x), game=game).value
            for x in (x_core, x_pub, x_em, x_naive, list(oracle.realization))
        )

        row = {
            "opponent": name,
            "regime": REGIME.get(name, "?"),
            "v_ref": v_ref,
            "oracle_radius": {k: radii[k] for k in ("het", "em", "naive")},
            "gain": {
                "oracle": u_oracle - v_ref,
                "core_rinf": u_core - v_ref,
                "public_point_h0": u_pub - v_ref,
                "censored_em_l0": u_em - v_ref,
                "naive_point": u_naive - v_ref,
            },
            "oracle_gap": {
                "core_rinf": u_oracle - u_core,
                "public_point_h0": u_oracle - u_pub,
                "censored_em_l0": u_oracle - u_em,
                "naive_point": u_oracle - u_naive,
            },
            "min_safety_value": min_safety,
        }
        rows.append(row)

        print(
            f"{name:<31}{REGIME.get(name, '?'):<22}"
            f"{radii['het']['reach_mean']:>7.3f}{radii['em']['reach_mean']:>7.3f}"
            f"{u_oracle - v_ref:>8.3f}{u_core - v_ref:>8.3f}{u_pub - v_ref:>8.3f}"
            f"{u_em - v_ref:>8.3f}{u_naive - v_ref:>8.3f}"
            f"{u_oracle - u_core:>9.3f}{u_oracle - u_em:>8.3f}"
        )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps(
            {
                "game": game,
                "v_ref": v_ref,
                "floor": v_ref - RHO_CAP,
                "episodes": EPISODES,
                "seed": SEED,
                "delta": DELTA,
                "method": METHOD,
                "rho_cap": RHO_CAP,
                "rows": rows,
            },
            indent=2,
        )
    )
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
