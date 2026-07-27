"""Run the turn river methods replicated experiment. See Experiments and supplementary Certification at the Unbucketed River."""

import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import statistics
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
SEEDS = [
    int(seed_text)
    for seed_text in os.environ.get(
        "TR_SEEDS", ",".join(str(seed) for seed in range(2026, 2026 + 25))
    ).split(",")
    if seed_text.strip()
]
DELTA = 0.1
METHOD = "empirical_bernstein"
RHO_GRID = [float(value) for value in os.environ.get("TR_RHOS", "0.5").split(",")]
WORKERS = int(os.environ.get("WORKERS", "10"))
INCLUDE_NAIVE = os.environ.get("TR_INCLUDE_NAIVE", "0") == "1"
OPPONENT_FILTER = {
    value.strip()
    for value in os.environ.get("TR_OPPONENTS", "").split(",")
    if value.strip()
}
OUT = Path(
    os.environ.get(
        "TR_TABLE_OUT",
        f"results/turn_river_method_table_{GAME}_{len(SEEDS)}s.json",
    )
)
PROGRESS = Path(
    os.environ.get(
        "TR_TABLE_PROGRESS",
        f"results/turn_river_method_table_{GAME}_{len(SEEDS)}s.jsonl",
    )
)

_WORKER_STATE: dict = {}


def _oracle_gain_task(
    task: tuple[str, float, str, float, dict, list[float]],
) -> tuple[str, float, float]:
    """Compute oracle gain task for the run turn river methods replicated workflow."""
    opponent_name, rho, game, v_ref, behavior, y_star = task
    payoff = build_payoff(game)
    oracle = safety_constrained_best_response(
        behavior, v_ref=v_ref, eps_safe=rho, game=game
    )
    oracle_gain = payoff.bilinear(list(oracle.realization), y_star) - v_ref
    return opponent_name, rho, oracle_gain


def _init_worker(payload: dict) -> None:
    """Initialize process-local state for a parallel worker."""
    _WORKER_STATE.update(payload)
    _WORKER_STATE["payoff"] = build_payoff(payload["game"])


def _solve_cell(task: tuple[int, str, float]) -> dict:
    """Solve cell for the run turn river methods replicated workflow."""
    seed, opponent_name, rho = task
    start_time = time.time()
    game = _WORKER_STATE["game"]
    v_ref = _WORKER_STATE["v_ref"]
    agent_behavior = _WORKER_STATE["agent_behavior"]
    omega = _WORKER_STATE["omega"]
    behavior = _WORKER_STATE["opponents"][opponent_name]
    y_star = _WORKER_STATE["y_stars"][opponent_name]
    oracle_gain = _WORKER_STATE["oracle_gains"][(opponent_name, rho)]
    payoff = _WORKER_STATE["payoff"]

    ev_point = OpponentEvidenceStore.for_game(game)
    ev_public = OpponentEvidenceStore.for_game(game)
    _payoff_samples, show_counts, fold_counts = native.simulate_showdown(
        game, agent_behavior, behavior, EPISODES, seed
    )
    for label, count in show_counts.items():
        ev_point.record(label, count)
        ev_public.record(label, count)
    for label, count in fold_counts.items():
        ev_public.record(label, count)

    groups = ev_public.public_groups()
    pub_intervals = ev_public.public_intervals(DELTA, method=METHOD)
    p_pub = _public_point_behavior(ev_public)
    p_em = _censored_em_behavior(game, ev_point, ev_public, omega)

    core = robust_safe_response_public(
        groups,
        pub_intervals,
        v_ref=v_ref,
        eps_safe=rho,
        game=game,
        weights=omega,
    )
    em = safety_constrained_best_response(p_em, v_ref=v_ref, eps_safe=rho, game=game)
    pub = safety_constrained_best_response(p_pub, v_ref=v_ref, eps_safe=rho, game=game)
    solved_arms = {"core": core, "em": em, "pub": pub}

    if INCLUDE_NAIVE:
        p_naive = {label: list(ev_point.p_hat(label)) for label in ev_point.labels}
        solved_arms["naive"] = safety_constrained_best_response(
            p_naive, v_ref=v_ref, eps_safe=rho, game=game
        )

    def realized_gain(response) -> float:
        """Compute realized gain for the run turn river methods replicated workflow."""
        return payoff.bilinear(list(response.realization), y_star) - v_ref

    row = {
        "seed": seed,
        "opponent": opponent_name,
        "rho": rho,
        "floor": v_ref - rho,
        "oracle": oracle_gain,
        "core": realized_gain(core),
        "em": realized_gain(em),
        "pub": realized_gain(pub),
        "core_cert_exploit": core.robust_value - v_ref,
        "core_cert_margin": core.robust_value - (v_ref - rho),
        "wall_s": time.time() - start_time,
    }
    if INCLUDE_NAIVE:
        row["naive"] = realized_gain(solved_arms["naive"])
    row["core_minus_em"] = row["core"] - row["em"]
    row["min_safety"] = min(
        safety_verifier(list(response.realization), game=game).value
        for response in solved_arms.values()
    )
    row["safety_margin"] = row["min_safety"] - row["floor"]
    return row


def _stats(values: list[float]) -> dict:
    """Compute summary statistics across independent replicates."""
    count = len(values)
    mean = sum(values) / count if count else None
    if count > 1:
        sd = statistics.stdev(values)
        se = sd / math.sqrt(count)
    else:
        sd = 0.0
        se = 0.0
    return {
        "n": count,
        "mean": mean,
        "sd": sd,
        "se": se,
        "ci95": 1.96 * se,
        "min": min(values) if values else None,
        "max": max(values) if values else None,
    }


def _aggregate(rows: list[dict]) -> list[dict]:
    """Aggregate the supplied records into summary statistics."""
    metric_names = [
        "oracle",
        "core",
        "em",
        "pub",
        "core_minus_em",
        "core_cert_exploit",
        "core_cert_margin",
        "min_safety",
        "safety_margin",
    ]
    if INCLUDE_NAIVE:
        metric_names.insert(4, "naive")

    grouped: dict[tuple[str, float], list[dict]] = {}
    for row in rows:
        grouped.setdefault((row["opponent"], row["rho"]), []).append(row)

    aggregate_rows = []
    for (opponent_name, rho), cells in sorted(grouped.items()):
        metrics = {
            metric: _stats([float(cell[metric]) for cell in cells])
            for metric in metric_names
        }
        aggregate_rows.append(
            {
                "opponent": opponent_name,
                "rho": rho,
                "n": len(cells),
                "metrics": metrics,
            }
        )
    return aggregate_rows


def _fmt_metric(stats: dict) -> str:
    """Compute fmt metric for the run turn river methods replicated workflow."""
    mean = stats["mean"]
    ci95 = stats["ci95"]
    if mean is None:
        return "n/a"
    return f"{mean:+.3f}+-{ci95:.3f}"


def _load_done() -> dict[tuple[int, str, float], dict]:
    """Load done for the run turn river methods replicated workflow."""
    done: dict[tuple[int, str, float], dict] = {}
    if not PROGRESS.exists():
        return done
    for line in PROGRESS.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        done[(row["seed"], row["opponent"], row["rho"])] = row
    return done


def main() -> None:
    """Run the command-line entry point."""
    game = GAME
    print(
        f"# starting replicated turn+river method table  game={game}  "
        f"seeds={SEEDS[0]}..{SEEDS[-1]} n={len(SEEDS)}  "
        f"episodes={EPISODES}  rhos={RHO_GRID}  workers={WORKERS}",
        flush=True,
    )
    sf0 = compile_game(game, 0)
    sf1 = compile_game(game, 1)
    blueprint = solve_blueprint(game, method="lp")
    assert blueprint.realization is not None
    x_blueprint = blueprint.realization
    v_ref = blueprint.value
    agent_behavior = sf0.behavior_from_realization(x_blueprint)
    omega = opponent_reach_weights(x_blueprint, game=game)
    actions = {
        info.label: [action for action, _ in info.children] for info in sf1.info_sets
    }
    opponents = _build_opponents(game, actions)
    if OPPONENT_FILTER:
        missing = sorted(OPPONENT_FILTER - set(opponents))
        if missing:
            raise ValueError(f"unknown TR_OPPONENTS entries: {missing}")
        opponents = {
            name: opponents[name] for name in opponents if name in OPPONENT_FILTER
        }

    y_stars: dict[str, list[float]] = {}
    print(
        f"# game compiled  v_ref={v_ref:+.5f}  opponents={list(opponents)}", flush=True
    )
    for opponent_name, behavior in opponents.items():
        y_star = list(
            Opponent(name=opponent_name, behavior=behavior, game=game).realization()
        )
        y_stars[opponent_name] = y_star

    oracle_tasks = [
        (opponent_name, rho, game, v_ref, behavior, y_stars[opponent_name])
        for opponent_name, behavior in opponents.items()
        for rho in RHO_GRID
    ]
    print(
        f"# precomputing {len(oracle_tasks)} oracle cells with "
        f"{min(WORKERS, len(oracle_tasks))} workers",
        flush=True,
    )
    oracle_gains: dict[tuple[str, float], float] = {}
    if oracle_tasks:
        context = mp.get_context("spawn")
        with context.Pool(processes=min(WORKERS, len(oracle_tasks))) as pool:
            for opponent_name, rho, oracle_gain in pool.imap_unordered(
                _oracle_gain_task, oracle_tasks
            ):
                oracle_gains[(opponent_name, rho)] = oracle_gain
                print(
                    f"  oracle {opponent_name:<26} rho={rho:.2f} gain={oracle_gain:+.3f}",
                    flush=True,
                )

    tasks = [
        (seed, opponent_name, rho)
        for seed in SEEDS
        for opponent_name in opponents
        for rho in RHO_GRID
    ]
    done = _load_done()
    todo = [task for task in tasks if task not in done]
    rows = list(done.values())
    print(
        f"{len(done)} cells already done, {len(todo)} to solve "
        f"(checkpoint: {PROGRESS})",
        flush=True,
    )

    if todo:
        payload = {
            "game": game,
            "v_ref": v_ref,
            "agent_behavior": agent_behavior,
            "omega": omega,
            "opponents": opponents,
            "y_stars": y_stars,
            "oracle_gains": oracle_gains,
        }
        context = mp.get_context("spawn")
        with (
            context.Pool(
                processes=min(WORKERS, len(todo)),
                initializer=_init_worker,
                initargs=(payload,),
            ) as pool,
            PROGRESS.open("a") as progress_file,
        ):
            for row in pool.imap_unordered(_solve_cell, todo):
                progress_file.write(json.dumps(row) + "\n")
                progress_file.flush()
                rows.append(row)
                print(
                    f"  seed={row['seed']} {row['opponent']:<26} rho={row['rho']:.2f} "
                    f"core={row['core']:+.3f} em={row['em']:+.3f} "
                    f"core-em={row['core_minus_em']:+.3f} "
                    f"cert={row['core_cert_exploit']:+.3f} "
                    f"safe_margin={row['safety_margin']:+.3e} ({row['wall_s']:.0f}s)",
                    flush=True,
                )

    rows = sorted(rows, key=lambda row: (row["opponent"], row["rho"], row["seed"]))
    aggregate_rows = _aggregate(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps(
            {
                "game": game,
                "v_ref": v_ref,
                "episodes": EPISODES,
                "seeds": SEEDS,
                "delta": DELTA,
                "method": METHOD,
                "rhos": RHO_GRID,
                "workers": WORKERS,
                "include_naive": INCLUDE_NAIVE,
                "rows": rows,
                "aggregate": aggregate_rows,
            },
            indent=2,
        )
    )

    print("\n=== aggregate (mean+-95% CI) ===", flush=True)
    header = (
        f"{'opponent':<28}{'rho':>6}{'n':>4}{'oracle':>14}{'core':>14}"
        f"{'em':>14}{'pub':>14}{'core-em':>14}{'cert':>14}{'safe':>14}"
    )
    print(header, flush=True)
    print("-" * len(header), flush=True)
    for aggregate_row in aggregate_rows:
        metrics = aggregate_row["metrics"]
        print(
            f"{aggregate_row['opponent']:<28}{aggregate_row['rho']:>6.2f}"
            f"{aggregate_row['n']:>4}"
            f"{_fmt_metric(metrics['oracle']):>14}"
            f"{_fmt_metric(metrics['core']):>14}"
            f"{_fmt_metric(metrics['em']):>14}"
            f"{_fmt_metric(metrics['pub']):>14}"
            f"{_fmt_metric(metrics['core_minus_em']):>14}"
            f"{_fmt_metric(metrics['core_cert_exploit']):>14}"
            f"{_fmt_metric(metrics['safety_margin']):>14}",
            flush=True,
        )

    leak_rows = [row for row in aggregate_rows if row["opponent"] != "tr_equilibrium"]
    min_leak_mean = min(row["metrics"]["core_minus_em"]["mean"] for row in leak_rows)
    min_leak_lcb = min(
        row["metrics"]["core_minus_em"]["mean"]
        - row["metrics"]["core_minus_em"]["ci95"]
        for row in leak_rows
    )
    worst_safety_margin = min(row["safety_margin"] for row in rows)
    print(
        f"\nleak rows: min mean core-em={min_leak_mean:+.3f}; "
        f"min 95% lower bound={min_leak_lcb:+.3f}; "
        f"worst safety margin={worst_safety_margin:+.3e}",
        flush=True,
    )
    print(f"wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
