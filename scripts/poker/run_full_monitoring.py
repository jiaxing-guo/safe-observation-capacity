"""Run the full monitoring experiment. See Experiments and supplementary Certification at the Unbucketed River."""

from dataclasses import dataclass
import json
import multiprocessing as mp
from pathlib import Path
from statistics import mean, pstdev

from safe_observation import native
from safe_observation.agents import solve_blueprint
from safe_observation.confidence import OpponentEvidenceStore
from safe_observation.experiments.online import _showdown_plan
from safe_observation.opponents import Opponent, leduc_opponent_suite
from safe_observation.payoff import build as build_payoff
from safe_observation.probe import SafetyBudgetLedger
from safe_observation.sequence_form import compile as compile_game
from safe_observation.solvers import (
    opponent_reach_weights,
    safety_constrained_best_response,
    safety_verifier,
)

CORE, POINT, SAD = "public_robust", "point_response", "safe_active_decensoring"
METHODS = (CORE, POINT, SAD)
MONITORINGS = ("full", "showdown")


ROUNDS, EPISODES = 60, 200
SEEDS = list(range(42, 47))
DELTA, METHOD = 0.1, "empirical_bernstein"
EPS_SAFE, RHO_CAP, KAPPA, DEBT_MAX, RNR_P = 0.0, 0.5, 0.2, 3.0, 0.5

OUT = Path("results/redundancy/full_vs_showdown_leduc.json")


def _run_one(
    method_name: str,
    opponent: Opponent,
    monitoring: str,
    v_ref: float,
    x_blueprint: tuple[float, ...],
    scbr_value: float,
    seed: int,
) -> dict:
    """Run the one experiment for the run full monitoring workflow."""
    game = opponent.game
    payoff = build_payoff(game)
    sf0 = compile_game(game, 0)
    sf1 = compile_game(game, 1)
    ev_point = OpponentEvidenceStore.for_game(game)
    ev_public = OpponentEvidenceStore.for_game(game)
    budget = SafetyBudgetLedger(
        rho_cap=RHO_CAP, hard_total=RHO_CAP * ROUNDS, debt_max=DEBT_MAX
    )
    y_star = list(opponent.realization())
    reach_accum: dict[str, float] = {}
    route_counter: dict[str, int] = {"core": 0, "point": 0}

    realized: list[float] = []
    min_safety = float("inf")
    cum_above_floor = 0.0
    for t in range(ROUNDS):
        x, spent, _believe = _showdown_plan(
            method_name,
            ev_point,
            ev_public,
            budget,
            game,
            v_ref,
            x_blueprint,
            sf1,
            payoff,
            DELTA,
            METHOD,
            EPS_SAFE,
            KAPPA,
            RNR_P,
            cum_above_floor,
            reach_accum,
            route_counter if method_name == SAD else None,
        )
        r = payoff.bilinear(x, y_star)
        realized.append(r)
        min_safety = min(min_safety, safety_verifier(x, game=game).value)
        budget.settle(spent, r - v_ref)
        cum_above_floor += r - v_ref
        behavior = sf0.behavior_from_realization(x)
        for label, w in opponent_reach_weights(x, game=game).items():
            reach_accum[label] = reach_accum.get(label, 0.0) + w
        _pay, show, fold = native.simulate_showdown(
            game, behavior, opponent.behavior, EPISODES, seed + t
        )
        for label, c in show.items():
            ev_point.record(label, c)
            ev_public.record(label, c)
        for label, c in fold.items():
            if monitoring == "full":
                ev_point.record(label, c)
            ev_public.record(label, c)

    tail = realized[max(0, len(realized) // 2) :]
    final_actual = mean(tail)
    n_route = route_counter["core"] + route_counter["point"]
    return {
        "exploitation_gain": final_actual - v_ref,
        "distance_from_scbr": scbr_value - final_actual,
        "min_safety_value": min_safety,
        "safety_violation": max(0.0, v_ref - min_safety),
        "core_selected_rate": route_counter["core"] / n_route if n_route else 0.0,
    }


@dataclass(frozen=True)
class _Cell:
    """Represent cell for the run full monitoring workflow."""

    opp_key: str
    opponent: Opponent
    method_name: str
    monitoring: str
    seed: int
    v_ref: float
    x_blueprint: tuple[float, ...]
    scbr_value: float


def _worker(cell: _Cell):
    """Execute one experiment cell in a worker process."""
    res = _run_one(
        cell.method_name,
        cell.opponent,
        cell.monitoring,
        cell.v_ref,
        cell.x_blueprint,
        cell.scbr_value,
        cell.seed,
    )
    return (cell.monitoring, cell.opp_key, cell.method_name, res)


def main() -> None:
    """Run the command-line entry point."""
    suite = leduc_opponent_suite()
    game = next(iter(suite.values())).game
    bp = solve_blueprint(game, method="lp")
    v_ref = bp.value
    assert bp.realization is not None
    x_blueprint = tuple(bp.realization)
    scbr = {
        name: safety_constrained_best_response(
            opp.behavior, v_ref=v_ref, eps_safe=EPS_SAFE, game=game
        ).value
        for name, opp in suite.items()
    }

    cells = [
        _Cell(opp_key, opp, m, mon, seed, v_ref, x_blueprint, scbr[opp_key])
        for opp_key, opp in suite.items()
        for m in METHODS
        for mon in MONITORINGS
        for seed in SEEDS
    ]
    print(
        f"{len(cells)} cells: {len(suite)} opp x {len(METHODS)} methods x "
        f"{len(MONITORINGS)} monitoring x {len(SEEDS)} seeds on {game}",
        flush=True,
    )

    with mp.Pool(processes=10) as pool:
        flat = pool.map(_worker, cells)

    agg: dict[tuple[str, str, str], list[dict]] = {}
    for mon, opp_key, method_name, res in flat:
        agg.setdefault((mon, opp_key, method_name), []).append(res)

    rows: dict[str, dict] = {}
    for mon in MONITORINGS:
        opp_out: dict[str, dict] = {}
        for opp_key in suite:
            cell_methods: dict[str, dict] = {}
            for m in METHODS:
                runs = agg[(mon, opp_key, m)]
                cell_methods[m] = {
                    "gain_mean": mean(r["exploitation_gain"] for r in runs),
                    "gain_std": (
                        pstdev([r["exploitation_gain"] for r in runs])
                        if len(runs) > 1
                        else 0.0
                    ),
                    "min_safety_value": min(r["min_safety_value"] for r in runs),
                    "safety_violation_max": max(r["safety_violation"] for r in runs),
                    "core_selected_rate": mean(r["core_selected_rate"] for r in runs),
                }

            cell_methods["core_minus_point"] = (
                cell_methods[CORE]["gain_mean"] - cell_methods[POINT]["gain_mean"]
            )
            opp_out[opp_key] = cell_methods
        rows[mon] = opp_out

    out = {
        "game": game,
        "game_value": v_ref,
        "floor": v_ref - RHO_CAP,
        "seeds": SEEDS,
        "rounds": ROUNDS,
        "episodes_per_round": EPISODES,
        "delta": DELTA,
        "methods": list(METHODS),
        "monitorings": list(MONITORINGS),
        "scbr_gain": {k: scbr[k] - v_ref for k in suite},
        "results": rows,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2))
    print(f"wrote {OUT}", flush=True)

    print("\n=== FULL vs SHOWDOWN: core / point gap (the redundancy claim) ===")
    hdr = (
        f"{'opponent':18}{'scbr':>7} | "
        f"{'core_F':>7}{'point_F':>8}{'gap_F':>7} | "
        f"{'core_S':>7}{'point_S':>8}{'gap_S':>7}"
    )
    print(hdr)
    for opp_key in suite:
        f = rows["full"][opp_key]
        s = rows["showdown"][opp_key]
        print(
            f"{opp_key:18}{scbr[opp_key] - v_ref:>+7.3f} | "
            f"{f[CORE]['gain_mean']:>+7.3f}{f[POINT]['gain_mean']:>+8.3f}"
            f"{f['core_minus_point']:>+7.3f} | "
            f"{s[CORE]['gain_mean']:>+7.3f}{s[POINT]['gain_mean']:>+8.3f}"
            f"{s['core_minus_point']:>+7.3f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
