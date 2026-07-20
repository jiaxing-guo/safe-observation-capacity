""

import json
import multiprocessing as mp
import os
from pathlib import Path
import statistics
import time

from safe_observation import native
from safe_observation.confidence import OpponentEvidenceStore
from safe_observation.experiments.online import _censored_em_behavior
from safe_observation.opponents import Opponent, holdem_equilibrium_opponent
from safe_observation.payoff import build as build_payoff
from safe_observation.sequence_form import compile as compile_game
from safe_observation.solvers import (
    opponent_reach_weights,
    restricted_nash_response,
    robust_safe_response_linear,
    robust_safe_response_public,
    safety_constrained_best_response,
    safety_filtered_restricted_nash_response,
    safety_verifier,
    solve_blueprint,
)
from scripts.poker.evaluate_public_routing import _build_gate_suite

GAME = os.environ.get("TR_GAME", "holdem_tr_b4")
RHO = float(os.environ.get("TR_RHO", "0.5"))
EPISODES = int(os.environ.get("TR_EPISODES", "100000"))
SEEDS = [
    int(s)
    for s in os.environ.get(
        "TR_SEEDS", ",".join(str(s) for s in range(2026, 2031))
    ).split(",")
    if s.strip()
]
DELTA = 0.1
METHOD = "empirical_bernstein"
WITH_CORE = os.environ.get("TR_WITH_CORE", "0") == "1"
OUT = Path(os.environ.get("TR_OUT", f"results/baseline_menu_{GAME}.json"))


LEAK_NAMES = [
    "tr_equilibrium",
    "river_overfold_w80",
    "turn_overfold_w70",
    "revealed_call_strong",
]


_WORKER: dict = {}


def _init_worker(game: str) -> None:
    ""
    payoff = build_payoff(game)
    sf0 = compile_game(game, 0)
    sf1 = compile_game(game, 1)
    bp = solve_blueprint(game, method="lp")
    assert bp.realization is not None
    x_bp = bp.realization
    v_ref = bp.value
    agent_behavior = sf0.behavior_from_realization(x_bp)
    omega = opponent_reach_weights(x_bp, game=game)
    actions = {info.label: [a for a, _ in info.children] for info in sf1.info_sets}
    eqb = holdem_equilibrium_opponent(game).behavior
    suite = _build_gate_suite(game, actions, eqb)
    _WORKER.update(
        game=game,
        payoff=payoff,
        sf1=sf1,
        v_ref=v_ref,
        floor=v_ref - RHO,
        agent_behavior=agent_behavior,
        omega=omega,
        suite=suite,
    )


def _dbr_rows(sf1, show, p_em, k_prior=10.0, p_cap=0.95):
    ""
    entries, h, meta = [], [], []
    for info in sf1.info_sets:
        counts = show.get(info.label)
        n = float(sum(counts)) if counts else 0.0
        if n <= 0:
            continue
        m = p_em.get(info.label)
        if m is None:
            continue
        p = min(n / (n + k_prior), p_cap)
        for ai, (_a, child) in enumerate(info.children):
            coef = p * m[ai]
            if coef <= 1e-9:
                continue
            row = len(h)
            entries.append((row, child, -1.0))
            entries.append((row, info.parent_seq, coef))
            h.append(0.0)
            meta.append((info.label, ai))
    return entries, h, meta


def _run_cell(task: tuple[str, int]) -> dict:
    ""
    name, seed = task
    g = _WORKER
    game, payoff, sf1 = g["game"], g["payoff"], g["sf1"]
    v_ref, floor, omega = g["v_ref"], g["floor"], g["omega"]
    agent_behavior, behavior = g["agent_behavior"], g["suite"][name]
    t0 = time.time()
    y_star = list(Opponent(name=name, behavior=behavior, game=game).realization())

    ev_point = OpponentEvidenceStore.for_game(game)
    ev_public = OpponentEvidenceStore.for_game(game)
    _p, show, fold = native.simulate_showdown(
        game, agent_behavior, behavior, EPISODES, seed
    )
    for label, c in show.items():
        ev_point.record(label, c)
        ev_public.record(label, c)
    for label, c in fold.items():
        ev_public.record(label, c)

    p_em = _censored_em_behavior(game, ev_point, ev_public, omega)
    y_em = list(sf1.realization_from_behavior(p_em))

    arms = {
        "safe_em": safety_constrained_best_response(
            p_em, v_ref=v_ref, eps_safe=RHO, game=game
        ),
        "safe_rnr": safety_filtered_restricted_nash_response(
            y_em, floor=floor, game=game
        ),
        "rnr_p1_brk": restricted_nash_response(y_em, p=1.0, game=game),
    }
    dbr_entries, dbr_h, dbr_meta = _dbr_rows(sf1, show, p_em)
    arms["safe_dbr"] = robust_safe_response_linear(
        ev_public.public_groups(),
        {},
        dbr_entries,
        dbr_h,
        v_ref=v_ref,
        eps_safe=RHO,
        game=game,
        weights=omega,
        row_meta=dbr_meta,
    )
    if WITH_CORE:
        groups = ev_public.public_groups()
        pub_intervals = ev_public.public_intervals(DELTA, method=METHOD)
        arms["core"] = robust_safe_response_public(
            groups, pub_intervals, v_ref=v_ref, eps_safe=RHO, game=game, weights=omega
        )

    cell: dict = {"opponent": name, "seed": seed, "arms": {}}
    for arm, resp in arms.items():
        real = payoff.bilinear(list(resp.realization), y_star) - v_ref
        worst = safety_verifier(list(resp.realization), game=game).value
        cell["arms"][arm] = {"real": real, "worst": worst}

    if seed == SEEDS[0]:
        oracle = safety_constrained_best_response(
            behavior, v_ref=v_ref, eps_safe=RHO, game=game
        )
        cell["oracle_gain"] = payoff.bilinear(list(oracle.realization), y_star) - v_ref
    cell["wall_s"] = time.time() - t0
    return cell


def main() -> None:
    t_start = time.time()
    game = GAME

    bp = solve_blueprint(game, method="lp")
    v_ref = bp.value
    floor = v_ref - RHO

    cells = [(name, seed) for name in LEAK_NAMES for seed in SEEDS]
    nproc = min(len(cells), 12)
    ckpt = OUT.with_suffix(".cells.jsonl")
    raw: list[dict] = []
    done: set[tuple[str, int]] = set()
    if ckpt.exists():
        for line in ckpt.open():
            if line.strip():
                r = json.loads(line)
                raw.append(r)
                done.add((r["opponent"], r["seed"]))
    todo = [c for c in cells if c not in done]

    print(
        f"# baseline menu  game={game}  v_ref={v_ref:+.5f}  rho={RHO}  "
        f"floor={floor:+.4f}  N={EPISODES}  cells={len(cells)} todo={len(todo)} "
        f"nproc={nproc} core={WITH_CORE}",
        flush=True,
    )

    if todo:
        ctx = mp.get_context("spawn")
        with (
            ckpt.open("a") as fh,
            ctx.Pool(nproc, initializer=_init_worker, initargs=(game,)) as pool,
        ):
            for cell in pool.imap_unordered(_run_cell, todo):
                fh.write(json.dumps(cell) + "\n")
                fh.flush()
                raw.append(cell)
                core = cell["arms"].get("core", {}).get("real", float("nan"))
                print(
                    f"  done {cell['opponent']:22s} seed={cell['seed']} "
                    f"core={core:+.3f} wall={cell['wall_s']:.0f}s"
                    f"{' [oracle]' if 'oracle_gain' in cell else ''}",
                    flush=True,
                )

    results: dict[str, dict] = {}
    for name in LEAK_NAMES:
        cs = [c for c in raw if c["opponent"] == name]
        oracle_gain = next(
            (c["oracle_gain"] for c in cs if "oracle_gain" in c), float("nan")
        )
        arm_names = list(cs[0]["arms"].keys())
        opp_summary: dict = {"oracle_gain": oracle_gain, "arms": {}}
        for arm in arm_names:
            reals = [c["arms"][arm]["real"] for c in cs]
            worsts = [c["arms"][arm]["worst"] for c in cs]
            n = len(reals)
            opp_summary["arms"][arm] = {
                "real_mean": statistics.mean(reals),
                "real_se": statistics.stdev(reals) / (n**0.5) if n > 1 else 0.0,
                "worst_safety_min": min(worsts),
                "holds_floor": min(worsts) >= floor - 1e-6,
            }
        results[name] = opp_summary

        print(f"\n## {name}   oracle gain = {oracle_gain:+.3f}")
        print(f"   {'arm':12s} {'real (mean±se)':>18s} {'worst-safety':>13s}  floor?")
        order = (["core"] if WITH_CORE else []) + [
            "safe_em",
            "safe_rnr",
            "safe_dbr",
            "rnr_p1_brk",
        ]
        for arm in order:
            a = opp_summary["arms"][arm]
            tag = "HOLDS" if a["holds_floor"] else "BREACH"
            print(
                f"   {arm:12s} {a['real_mean']:+8.3f} ± {a['real_se']:.3f}   "
                f"{a['worst_safety_min']:+8.3f}     [{tag}]"
            )

    payload = {
        "game": game,
        "v_ref": v_ref,
        "rho": RHO,
        "floor": floor,
        "episodes": EPISODES,
        "seeds": SEEDS,
        "with_core": WITH_CORE,
        "results": results,
        "wall_s": time.time() - t_start,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {OUT}   wall={payload['wall_s']:.1f}s")


if __name__ == "__main__":
    main()
