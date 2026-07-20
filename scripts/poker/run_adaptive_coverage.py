""

from collections import defaultdict
import json
import math
import multiprocessing as mp
from pathlib import Path

from safe_observation import native
from safe_observation.agents import solve_blueprint
from safe_observation.confidence import OpponentEvidenceStore
from safe_observation.experiments.online import _showdown_plan
from safe_observation.opponents import holdem_showdown_opponent_suite
from safe_observation.payoff import build as build_payoff
from safe_observation.probe import SafetyBudgetLedger
from safe_observation.sequence_form import compile as compile_game
from safe_observation.solvers import opponent_reach_weights

OPPONENTS = ("equilibrium", "censored_fold", "overfold", "calling_station")
DELTAS = (0.05, 0.1, 0.2)
ROUNDS = 60
EPISODES = 200
SEEDS = tuple(range(42, 67))


def _inside(game, groups, intervals, weights, y_star) -> bool:
    grp = {k: list(v) for k, v in groups.items()}
    payload = {k: [tuple(b) for b in bd] for k, bd in intervals.items()}
    cset = native.ConfidenceSet.from_public(game, grp, payload, weights)
    return cset.nrows == 0 or cset.max_violation(y_star) <= 1e-9


def _one_seed(arg):
    ""
    game, opp_name, gval, xbp, seed = arg
    opp = holdem_showdown_opponent_suite(game)[opp_name]
    y_star = list(opp.realization())
    payoff = build_payoff(game)
    sf0 = compile_game(game, 0)
    sf1 = compile_game(game, 1)
    ev_point = OpponentEvidenceStore.for_game(game)
    ev_public = OpponentEvidenceStore.for_game(game)
    budget = SafetyBudgetLedger(rho_cap=0.5, hard_total=0.5 * ROUNDS, debt_max=3.0)
    reach_accum: dict[str, float] = {}

    violated = {d: False for d in DELTAS}
    per_round_hits = {d: 0 for d in DELTAS}
    cum_above = 0.0
    for t in range(1, ROUNDS + 1):
        x, spent, _believe = _showdown_plan(
            "safe_active_decensoring",
            ev_point,
            ev_public,
            budget,
            game,
            gval,
            xbp,
            sf1,
            payoff,
            0.1,
            "empirical_bernstein",
            0.0,
            0.2,
            0.5,
            cum_above,
            reach_accum,
            None,
        )
        r = payoff.bilinear(x, y_star)
        budget.settle(spent, r - gval)
        cum_above += r - gval

        for label, w in opponent_reach_weights(x, game=game).items():
            reach_accum[label] = reach_accum.get(label, 0.0) + w
        behavior = sf0.behavior_from_realization(x)
        _pay, show, fold = native.simulate_showdown(
            game, behavior, opp.behavior, EPISODES, seed + t
        )
        for label, c in show.items():
            ev_point.record(label, c)
            ev_public.record(label, c)
        for label, c in fold.items():
            ev_public.record(label, c)

        groups = ev_public.public_groups()
        for delta in DELTAS:
            delta_t = delta * 6.0 / (math.pi**2 * t * t)
            iv = ev_public.public_intervals(delta_t, union_bound=True)
            if _inside(game, groups, iv, reach_accum, y_star):
                per_round_hits[delta] += 1
            else:
                violated[delta] = True
    return opp_name, {d: (not violated[d], per_round_hits[d]) for d in DELTAS}


def main() -> None:
    game = "holdem"
    bp = solve_blueprint(game, method="lp")
    xbp = tuple(bp.realization)
    tasks = [
        (game, opp, bp.value, xbp, s * 100_000) for opp in OPPONENTS for s in SEEDS
    ]
    print(
        f"launching {len(tasks)} adaptive-coverage seeds on 10 workers...", flush=True
    )

    agg = defaultdict(lambda: {d: {"anytime": 0, "hits": 0} for d in DELTAS})
    done = 0
    with mp.Pool(processes=10) as pool:
        for opp_name, per_delta in pool.imap_unordered(_one_seed, tasks):
            for d, (clean, hits) in per_delta.items():
                agg[opp_name][d]["anytime"] += int(clean)
                agg[opp_name][d]["hits"] += hits
            done += 1
            if done % 20 == 0:
                print(f"... {done}/{len(tasks)}", flush=True)

    n_seeds = len(SEEDS)
    n_checks = n_seeds * ROUNDS
    out = {
        "game": game,
        "policy": "safe_active_decensoring",
        "rounds": ROUNDS,
        "episodes_per_round": EPISODES,
        "seeds": list(SEEDS),
        "deltas": list(DELTAS),
        "opponents": [],
    }
    print(
        f"\n{'opponent':<18}" + "".join(f"  d={d}: anytime/per-round" for d in DELTAS),
        flush=True,
    )
    for opp in OPPONENTS:
        cells = []
        row = f"{opp:<18}"
        for d in DELTAS:
            a = agg[opp][d]["anytime"] / n_seeds
            pr = agg[opp][d]["hits"] / n_checks
            cells.append(
                {
                    "delta": d,
                    "guarantee": 1 - d,
                    "anytime_coverage": a,
                    "per_round_coverage": pr,
                    "meets_guarantee": a >= 1 - d - 1e-9,
                }
            )
            row += f"   {a:.2f}/{pr:.3f}"
        out["opponents"].append({"opponent": opp, "cells": cells})
        print(row, flush=True)

    p = Path("results/adaptive_coverage_holdem.json")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {p}", flush=True)


if __name__ == "__main__":
    main()
