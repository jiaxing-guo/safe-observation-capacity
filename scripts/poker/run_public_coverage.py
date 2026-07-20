""

import json
import math
from pathlib import Path

from safe_observation import native
from safe_observation.confidence import OpponentEvidenceStore
from safe_observation.opponents import holdem_showdown_opponent_suite
from safe_observation.sequence_form import compile as compile_game
from safe_observation.solvers import opponent_reach_weights

ARMS = ("weighted", "unweighted", "no_union")


OPPONENTS = ("equilibrium", "censored_fold", "overfold", "calling_station")
DELTAS = (0.05, 0.1, 0.2)
ROUNDS = 60
EPISODES = 200
SEEDS = tuple(range(42, 67))


def _build_set(game, groups, intervals, weights):
    grp = {k: list(v) for k, v in groups.items()}
    payload = {k: [tuple(b) for b in bd] for k, bd in intervals.items()}
    return native.ConfidenceSet.from_public(game, grp, payload, weights)


def _inside(cset, y_star) -> bool:
    return cset.nrows == 0 or cset.max_violation(y_star) <= 1e-9


def run_opponent(game: str, opp, weights) -> dict:
    y_star = list(opp.realization())
    blueprint_behavior = compile_game(game, 0).behavior_from_realization(
        native.blueprint_realization(game, 0)
    )

    never_violated = {d: {a: 0 for a in ARMS} for d in DELTAS}
    per_round_hits = {d: {a: 0 for a in ARMS} for d in DELTAS}
    for seed in SEEDS:
        store = OpponentEvidenceStore.for_game(game)
        violated = {d: {a: False for a in ARMS} for d in DELTAS}
        for t in range(1, ROUNDS + 1):
            _pay, show, fold = native.simulate_showdown(
                game, blueprint_behavior, opp.behavior, EPISODES, seed * 100_000 + t
            )
            for label, c in show.items():
                store.record(label, c)
            for label, c in fold.items():
                store.record(label, c)
            groups = store.public_groups()
            for delta in DELTAS:
                delta_t = delta * 6.0 / (math.pi**2 * t * t)
                iv_union = store.public_intervals(delta_t, union_bound=True)
                iv_plain = store.public_intervals(delta, union_bound=False)
                arms = {
                    "weighted": _build_set(game, groups, iv_union, weights),
                    "unweighted": _build_set(game, groups, iv_union, None),
                    "no_union": _build_set(game, groups, iv_plain, weights),
                }
                for arm, cset in arms.items():
                    if _inside(cset, y_star):
                        per_round_hits[delta][arm] += 1
                    else:
                        violated[delta][arm] = True
        for delta in DELTAS:
            for arm in ARMS:
                if not violated[delta][arm]:
                    never_violated[delta][arm] += 1
    n_checks = len(SEEDS) * ROUNDS
    cells = [
        {
            "delta": delta,
            "guarantee": 1.0 - delta,
            "arms": {
                arm: {
                    "anytime_coverage": never_violated[delta][arm] / len(SEEDS),
                    "per_round_coverage": per_round_hits[delta][arm] / n_checks,
                    "meets_guarantee": (never_violated[delta][arm] / len(SEEDS))
                    >= 1.0 - delta - 1e-9,
                }
                for arm in ARMS
            },
        }
        for delta in DELTAS
    ]
    return {"opponent": opp.name, "cells": cells}


def main() -> None:
    game = "holdem"
    suite = holdem_showdown_opponent_suite(game=game)
    by_name = {o.name: o for o in suite.values()}

    weights = opponent_reach_weights(native.blueprint_realization(game, 0), game=game)
    results = {
        "game": game,
        "monitoring": "showdown",
        "rounds": ROUNDS,
        "episodes_per_round": EPISODES,
        "seeds": list(SEEDS),
        "deltas": list(DELTAS),
        "opponents": [],
    }
    for name in OPPONENTS:
        opp = by_name[name]
        print(f"=== coverage: {name} ===", flush=True)
        res = run_opponent(game, opp, weights)
        for cell in res["cells"]:
            a = cell["arms"]
            print(
                f"  delta={cell['delta']:.2f}  "
                f"weighted anytime={a['weighted']['anytime_coverage']:.3f} "
                f"per_round={a['weighted']['per_round_coverage']:.4f}  | "
                f"unweighted={a['unweighted']['anytime_coverage']:.3f}  "
                f"no_union={a['no_union']['anytime_coverage']:.3f}",
                flush=True,
            )
        results["opponents"].append(res)
    out = Path("results/public_coverage_holdem.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
