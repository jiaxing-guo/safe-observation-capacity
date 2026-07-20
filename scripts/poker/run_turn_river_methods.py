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
from safe_observation.opponents import Opponent, holdem_equilibrium_opponent
from safe_observation.payoff import build as build_payoff
from safe_observation.sequence_form import compile as compile_game
from safe_observation.solvers import (
    opponent_reach_weights,
    robust_safe_response_public,
    safety_constrained_best_response,
    safety_verifier,
    solve_blueprint,
)

GAME = os.environ.get("TR_GAME", "holdem_tr_b4")
EPISODES = int(os.environ.get("TR_EPISODES", "300000"))
SEED = 2026
DELTA = 0.1
METHOD = "empirical_bernstein"
RHO_CAP = 0.5
OUT = Path(
    os.environ.get("TR_TABLE_OUT", f"results/turn_river_method_table_{GAME}.json")
)

_RANK = {c: i for i, c in enumerate("23456789TJQKA")}


def _top_rank(hole: str) -> int:
    return max(_RANK[hole[0]], _RANK[hole[2]])


def _perturb(base, actions, pick, target, weight):
    out = {label: list(dist) for label, dist in base.items()}
    for label, dist in base.items():
        hole, hist = label.split("|", 1)
        acts = actions[label]
        if target not in acts or not pick(hole, hist, acts):
            continue
        w = weight(hole, hist) if callable(weight) else weight
        if w <= 0.0:
            continue
        ti = acts.index(target)
        new = [p * (1.0 - w) for p in dist]
        new[ti] += w
        s = sum(new)
        out[label] = [p / s for p in new] if s > 1e-12 else list(dist)
    return out


def _build_opponents(game, actions):
    eq = holdem_equilibrium_opponent(game).behavior

    def river_fold(label):
        hist = label.split("|", 1)[1]
        return "/" in hist and "f" in actions[label]

    def turn_fold(label):
        hist = label.split("|", 1)[1]
        return "/" not in hist and "f" in actions[label]

    return {
        "tr_equilibrium": eq,
        "tr_river_overfold_uniform": _perturb(
            eq, actions, lambda h, hi, a: river_fold(f"{h}|{hi}"), "f", 0.5
        ),
        "tr_river_overfold_strong": _perturb(
            eq,
            actions,
            lambda h, hi, a: river_fold(f"{h}|{hi}"),
            "f",
            lambda h, hi: 0.6 if _top_rank(h) >= 9 else 0.0,
        ),
        "tr_turn_overfold": _perturb(
            eq, actions, lambda h, hi, a: turn_fold(f"{h}|{hi}"), "f", 0.5
        ),
    }


def main() -> None:
    game = GAME
    payoff = build_payoff(game)
    sf0 = compile_game(game, 0)
    sf1 = compile_game(game, 1)
    bp = solve_blueprint(game, method="lp")
    assert bp.realization is not None
    x_bp = bp.realization
    v_ref = bp.value
    agent_behavior = sf0.behavior_from_realization(x_bp)
    omega = opponent_reach_weights(x_bp, game=game)
    floor = v_ref - RHO_CAP
    actions = {info.label: [a for a, _ in info.children] for info in sf1.info_sets}

    opponents = _build_opponents(game, actions)
    rows = []
    print(
        f"# turn+river VALUE experiment  game={game}  v_ref={v_ref:+.5f}  "
        f"floor={floor:+.4f}  episodes={EPISODES}\n"
    )
    hdr = (
        f"{'opponent':<28}{'oracle':>8}{'core':>8}{'em':>8}{'pub':>8}{'naive':>8}"
        f" | {'core-em':>8}{'core_cert':>10}{'minSafe':>9}"
    )
    print(hdr)
    print("-" * len(hdr))

    for name, behavior in opponents.items():
        y_star = list(Opponent(name=name, behavior=behavior, game=game).realization())
        ev_point = OpponentEvidenceStore.for_game(game)
        ev_public = OpponentEvidenceStore.for_game(game)
        _p, show, fold = native.simulate_showdown(
            game, agent_behavior, behavior, EPISODES, SEED
        )
        for label, c in show.items():
            ev_point.record(label, c)
            ev_public.record(label, c)
        for label, c in fold.items():
            ev_public.record(label, c)
        groups = ev_public.public_groups()
        pub_intervals = ev_public.public_intervals(DELTA, method=METHOD)
        p_pub = _public_point_behavior(ev_public)
        p_em = _censored_em_behavior(game, ev_point, ev_public, omega)
        p_naive = {label: list(ev_point.p_hat(label)) for label in ev_point.labels}

        core = robust_safe_response_public(
            groups,
            pub_intervals,
            v_ref=v_ref,
            eps_safe=RHO_CAP,
            game=game,
            weights=omega,
        )
        em = safety_constrained_best_response(
            p_em, v_ref=v_ref, eps_safe=RHO_CAP, game=game
        )
        pub = safety_constrained_best_response(
            p_pub, v_ref=v_ref, eps_safe=RHO_CAP, game=game
        )
        naive = safety_constrained_best_response(
            p_naive, v_ref=v_ref, eps_safe=RHO_CAP, game=game
        )
        oracle = safety_constrained_best_response(
            behavior, v_ref=v_ref, eps_safe=RHO_CAP, game=game
        )

        def realized(resp, y=y_star):
            return payoff.bilinear(list(resp.realization), y) - v_ref

        g_core = realized(core)
        g_em = realized(em)
        g_pub = realized(pub)
        g_naive = realized(naive)
        g_oracle = realized(oracle)

        core_cert = core.robust_value - v_ref
        min_safety = min(
            safety_verifier(list(r.realization), game=game).value
            for r in (core, em, pub, naive)
        )
        rows.append(
            {
                "opponent": name,
                "v_ref": v_ref,
                "floor": floor,
                "oracle": g_oracle,
                "core": g_core,
                "em": g_em,
                "pub": g_pub,
                "naive": g_naive,
                "core_minus_em": g_core - g_em,
                "core_cert_exploit": core_cert,
                "min_safety": min_safety,
            }
        )
        print(
            f"{name:<28}{g_oracle:>8.3f}{g_core:>8.3f}{g_em:>8.3f}{g_pub:>8.3f}"
            f"{g_naive:>8.3f} | {g_core - g_em:>8.3f}{core_cert:>10.3f}{min_safety:>9.3f}"
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
                "rows": rows,
            },
            indent=2,
        )
    )
    print(
        "\ncore-em = core realized - em realized (>0 => robustness wins in VALUE "
        "= Case B anchor).\ncore_cert = core's CERTIFIED exploitation lower bound "
        "(worst case over C_pub); the point arms have none.\nminSafe must hold floor "
        f"{floor:+.4f} (safety decoupled)."
    )
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
