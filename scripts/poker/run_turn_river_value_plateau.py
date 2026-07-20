""

import json
import multiprocessing as mp
import os
from pathlib import Path
import time

from safe_observation import native
from safe_observation.confidence import OpponentEvidenceStore
from safe_observation.experiments.online import (
    _censored_em_behavior,
    _public_point_behavior,
)
from safe_observation.opponents import Opponent
from safe_observation.payoff import build as build_payoff
from safe_observation.sequence_form import compile as compile_game
from safe_observation.solvers import (
    opponent_reach_weights,
    robust_safe_response_public,
    safety_constrained_best_response,
    safety_verifier,
    solve_blueprint,
)
from scripts.poker.run_turn_river_methods import _build_opponents

GAME = os.environ.get("TR_GAME", "holdem_tr_b4")
OPPONENT = os.environ.get("TR_OPP", "tr_river_overfold_strong")
SEED = 2026
DELTA = 0.1
METHOD = "empirical_bernstein"
RHO_CAP = 0.5
EPISODE_GRID = [
    int(x)
    for x in os.environ.get("TR_GRID", "100000,300000,1000000,3000000").split(",")
]
WORKERS = int(os.environ.get("WORKERS", "4"))
OUT = Path(
    os.environ.get("TR_PLATEAU_OUT", f"results/turn_river_value_plateau_{GAME}.json")
)
PROGRESS = Path(
    os.environ.get(
        "TR_PLATEAU_PROGRESS", f"results/turn_river_value_plateau_{GAME}.jsonl"
    )
)

_G: dict = {}


def _init_worker(payload: dict) -> None:
    _G.update(payload)
    _G["payoffs"] = build_payoff(payload["game"])


def _rw_error(center, behavior_map, omega, fold_labels):
    num = den = 0.0
    for label, ref in behavior_map.items():
        w = omega.get(label, 0.0)
        if w <= 0.0 or label not in fold_labels or not ref:
            continue
        c = center.get(label)
        if c is None:
            continue
        num += w * max(abs(p - q) for p, q in zip(c, ref, strict=True))
        den += w
    return num / den if den > 0 else 0.0


def _solve_cell(episodes: int) -> dict:
    t0 = time.time()
    game = _G["game"]
    v_ref = _G["v_ref"]
    omega = _G["omega"]
    payoff = _G["payoffs"]
    behavior = _G["behavior"]
    y_star_vec = _G["y_star_vec"]
    behavior_map = _G["behavior_map"]
    fold_labels = _G["fold_labels"]

    ev_point = OpponentEvidenceStore.for_game(game)
    ev_public = OpponentEvidenceStore.for_game(game)
    _p, show, fold = native.simulate_showdown(
        game, _G["agent_behavior"], behavior, episodes, SEED
    )
    for label, c in show.items():
        ev_point.record(label, c)
        ev_public.record(label, c)
    for label, c in fold.items():
        ev_public.record(label, c)
    groups = ev_public.public_groups()
    pub_intervals = ev_public.public_intervals(DELTA, method=METHOD)
    p_em = _censored_em_behavior(game, ev_point, ev_public, omega)
    p_pub = _public_point_behavior(ev_public)

    core = robust_safe_response_public(
        groups, pub_intervals, v_ref=v_ref, eps_safe=RHO_CAP, game=game, weights=omega
    )
    em = safety_constrained_best_response(
        p_em, v_ref=v_ref, eps_safe=RHO_CAP, game=game
    )
    pub = safety_constrained_best_response(
        p_pub, v_ref=v_ref, eps_safe=RHO_CAP, game=game
    )
    oracle = safety_constrained_best_response(
        behavior, v_ref=v_ref, eps_safe=RHO_CAP, game=game
    )

    def realized(resp):
        return payoff.bilinear(list(resp.realization), y_star_vec) - v_ref

    g_core = realized(core)
    g_em = realized(em)
    return {
        "episodes": episodes,
        "core": g_core,
        "em": g_em,
        "pub": realized(pub),
        "oracle": realized(oracle),
        "core_minus_em": g_core - g_em,
        "d_em_fold": _rw_error(p_em, behavior_map, omega, fold_labels),
        "min_safety": min(
            safety_verifier(list(r.realization), game=game).value
            for r in (core, em, pub)
        ),
        "wall_s": time.time() - t0,
    }


def main() -> None:
    game = GAME
    sf0 = compile_game(game, 0)
    sf1 = compile_game(game, 1)
    bp = solve_blueprint(game, method="lp")
    assert bp.realization is not None
    x_bp = bp.realization
    v_ref = bp.value
    agent_behavior = sf0.behavior_from_realization(x_bp)
    omega = opponent_reach_weights(x_bp, game=game)
    actions = {info.label: [a for a, _ in info.children] for info in sf1.info_sets}
    opponents = _build_opponents(game, actions)
    behavior = opponents[OPPONENT]
    opp = Opponent(name=OPPONENT, behavior=behavior, game=game)
    y_star_vec = list(opp.realization())
    behavior_map = {
        info.label: list(behavior.get(info.label, [])) for info in sf1.info_sets
    }
    fold_labels = {label for label, acts in actions.items() if "f" in acts}

    print(
        f"# turn+river VALUE-PLATEAU (parallel x{WORKERS})  game={game}  "
        f"opp={OPPONENT}  v_ref={v_ref:+.5f}  grid={EPISODE_GRID}\n",
        flush=True,
    )

    done: dict[int, dict] = {}
    if PROGRESS.exists():
        for line in PROGRESS.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                done[r["episodes"]] = r
    todo = [n for n in EPISODE_GRID if n not in done]
    rows = list(done.values())

    if todo:
        payload = {
            "game": game,
            "v_ref": v_ref,
            "omega": omega,
            "agent_behavior": agent_behavior,
            "behavior": behavior,
            "y_star_vec": y_star_vec,
            "behavior_map": behavior_map,
            "fold_labels": fold_labels,
        }
        ctx = mp.get_context("spawn")
        with (
            ctx.Pool(
                processes=min(WORKERS, len(todo)),
                initializer=_init_worker,
                initargs=(payload,),
            ) as pool,
            PROGRESS.open("a") as fp,
        ):
            for r in pool.imap_unordered(_solve_cell, todo):
                fp.write(json.dumps(r) + "\n")
                fp.flush()
                rows.append(r)
                print(
                    f"  done N={r['episodes']:>8}  core={r['core']:+.3f} "
                    f"em={r['em']:+.3f} core-em={r['core_minus_em']:+.3f} "
                    f"d_em_fold={r['d_em_fold']:.3f} ({r['wall_s']:.0f}s)",
                    flush=True,
                )

    rows = sorted(rows, key=lambda r: r["episodes"])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps(
            {
                "game": game,
                "opponent": OPPONENT,
                "v_ref": v_ref,
                "grid": EPISODE_GRID,
                "rho_cap": RHO_CAP,
                "rows": rows,
            },
            indent=2,
        )
    )

    print(
        f"\n{'episodes':>10}{'core':>8}{'em':>8}{'pub':>8}{'oracle':>8}"
        f"{'core-em':>9}{'d_em_fold':>11}{'minSafe':>9}",
        flush=True,
    )
    for r in rows:
        print(
            f"{r['episodes']:>10}{r['core']:>8.3f}{r['em']:>8.3f}{r['pub']:>8.3f}"
            f"{r['oracle']:>8.3f}{r['core_minus_em']:>9.3f}{r['d_em_fold']:>11.3f}"
            f"{r['min_safety']:>9.3f}",
            flush=True,
        )
    if len(rows) >= 2:
        first, last = rows[0], rows[-1]
        verdict = (
            "VALUE PLATEAU (structural)"
            if last["core_minus_em"] > 0.5 * first["core_minus_em"]
            else "shrinking (cold-start?)"
        )
        print(
            f"\ncore-em: {first['core_minus_em']:+.3f} (N={first['episodes']}) -> "
            f"{last['core_minus_em']:+.3f} (N={last['episodes']})  = {verdict}",
            flush=True,
        )
    print(f"wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
