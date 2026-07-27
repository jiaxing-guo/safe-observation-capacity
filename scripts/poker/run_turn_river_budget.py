"""Run the turn river budget experiment. See Experiments and supplementary Certification at the Unbucketed River."""

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
EPISODES = int(os.environ.get("TR_EPISODES", "300000"))
SEED = 2026
DELTA = 0.1
METHOD = "empirical_bernstein"
RHO_GRID = [float(x) for x in os.environ.get("TR_RHOS", "0.05,0.1,0.25,0.5").split(",")]
WORKERS = int(os.environ.get("WORKERS", "12"))
OUT = Path(
    os.environ.get("TR_BUDGET_OUT", f"results/turn_river_budget_sweep_{GAME}.json")
)
PROGRESS = Path(
    os.environ.get(
        "TR_BUDGET_PROGRESS", f"results/turn_river_budget_sweep_{GAME}.jsonl"
    )
)


_G: dict = {}


def _init_worker(game: str, v_ref: float, omega: dict, payloads: dict) -> None:
    """Initialize process-local state for a parallel worker."""
    _G["game"] = game
    _G["v_ref"] = v_ref
    _G["omega"] = omega
    _G["payoffs"] = build_payoff(game)
    _G["payloads"] = payloads


def _solve_cell(task: tuple[str, float]) -> dict:
    """Solve cell for the run turn river budget workflow."""
    name, rho = task
    t0 = time.time()
    game = _G["game"]
    v_ref = _G["v_ref"]
    omega = _G["omega"]
    payoff = _G["payoffs"]
    pl = _G["payloads"][name]
    groups = pl["groups"]
    pub_intervals = pl["pub_intervals"]
    p_em = pl["p_em"]
    p_pub = pl["p_pub"]
    y_star = pl["y_star"]
    behavior = pl["behavior"]

    core = robust_safe_response_public(
        groups, pub_intervals, v_ref=v_ref, eps_safe=rho, game=game, weights=omega
    )
    em = safety_constrained_best_response(p_em, v_ref=v_ref, eps_safe=rho, game=game)
    pub = safety_constrained_best_response(p_pub, v_ref=v_ref, eps_safe=rho, game=game)
    oracle = safety_constrained_best_response(
        behavior, v_ref=v_ref, eps_safe=rho, game=game
    )

    def realized(resp):
        """Evaluate a realization plan against the selected opponent."""
        return payoff.bilinear(list(resp.realization), y_star) - v_ref

    g_core = realized(core)
    g_em = realized(em)
    g_pub = realized(pub)
    g_oracle = realized(oracle)
    min_safety = min(
        safety_verifier(list(r.realization), game=game).value for r in (core, em, pub)
    )
    return {
        "opponent": name,
        "rho": rho,
        "floor": v_ref - rho,
        "core": g_core,
        "em": g_em,
        "pub": g_pub,
        "oracle": g_oracle,
        "core_minus_em": g_core - g_em,
        "min_safety": min_safety,
        "wall_s": time.time() - t0,
    }


def main() -> None:
    """Run the command-line entry point."""
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

    print(
        f"# turn+river BUDGET sweep (parallel x{WORKERS})  game={game}  "
        f"v_ref={v_ref:+.5f}  episodes={EPISODES}  rhos={RHO_GRID}\n",
        flush=True,
    )

    payloads: dict[str, dict] = {}
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
        payloads[name] = {
            "behavior": behavior,
            "y_star": y_star,
            "groups": ev_public.public_groups(),
            "pub_intervals": ev_public.public_intervals(DELTA, method=METHOD),
            "p_em": _censored_em_behavior(game, ev_point, ev_public, omega),
            "p_pub": _public_point_behavior(ev_public),
        }
        print(f"  prepared payload: {name}", flush=True)

    tasks = [(name, rho) for name in opponents for rho in RHO_GRID]

    done: dict[tuple[str, float], dict] = {}
    if PROGRESS.exists():
        for line in PROGRESS.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            done[(r["opponent"], r["rho"])] = r
    todo = [t for t in tasks if t not in done]
    print(
        f"\n{len(done)} cells already done, {len(todo)} to solve "
        f"(stream + checkpoint to {PROGRESS})\n",
        flush=True,
    )

    rows = list(done.values())
    if todo:
        ctx = mp.get_context("spawn")
        with (
            ctx.Pool(
                processes=min(WORKERS, len(todo)),
                initializer=_init_worker,
                initargs=(game, v_ref, omega, payloads),
            ) as pool,
            PROGRESS.open("a") as fp,
        ):
            for r in pool.imap_unordered(_solve_cell, todo):
                fp.write(json.dumps(r) + "\n")
                fp.flush()
                rows.append(r)
                print(
                    f"  done {r['opponent']:<26} rho={r['rho']:.2f} "
                    f"core={r['core']:+.3f} em={r['em']:+.3f} "
                    f"core-em={r['core_minus_em']:+.3f} ({r['wall_s']:.0f}s)",
                    flush=True,
                )

    rows = sorted(rows, key=lambda r: (r["opponent"], r["rho"]))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps(
            {
                "game": game,
                "v_ref": v_ref,
                "episodes": EPISODES,
                "seed": SEED,
                "delta": DELTA,
                "method": METHOD,
                "rhos": RHO_GRID,
                "workers": WORKERS,
                "rows": rows,
            },
            indent=2,
        )
    )

    by_opp: dict[str, list[dict]] = {}
    for r in rows:
        by_opp.setdefault(r["opponent"], []).append(r)
    for name, cells in by_opp.items():
        print(f"\n=== {name} ===", flush=True)
        print(
            f"  {'rho':>5}{'floor':>9}{'core':>8}{'em':>8}{'pub':>8}{'oracle':>8}"
            f"{'core-em':>9}{'minSafe':>9}",
            flush=True,
        )
        for r in sorted(cells, key=lambda c: c["rho"]):
            print(
                f"  {r['rho']:>5.2f}{r['floor']:>9.3f}{r['core']:>8.3f}{r['em']:>8.3f}"
                f"{r['pub']:>8.3f}{r['oracle']:>8.3f}{r['core_minus_em']:>9.3f}"
                f"{r['min_safety']:>9.3f}",
                flush=True,
            )

    leaks = [r for r in rows if r["opponent"] != "tr_equilibrium"]
    ok = all(r["core_minus_em"] > 0 for r in leaks)
    print(
        f"\nORDERING core>EM on all leak×budget cells: {ok}  "
        f"(min core-em on leaks = {min(r['core_minus_em'] for r in leaks):+.3f})",
        flush=True,
    )
    print(f"wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
