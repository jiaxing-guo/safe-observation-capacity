""

import json
from pathlib import Path
import sys

from safe_observation.experiments.online import run_showdown_comparison
from safe_observation.opponents import holdem_censored_fold_opponent

RHO_GRID = [0.0, 0.05, 0.1, 0.25, 0.5]
SEEDS = list(range(42, 67))
ROUNDS = 60
EPISODES = 200
WORKERS = 10
OUT = Path("results/rho_frontier_holdem.json")


TRACE = ["passive_public", "public_robust", "point_response", "safe_active_decensoring"]


def main() -> None:
    cells = []
    vref = None
    for rho in RHO_GRID:
        suite = {"censored_fold": holdem_censored_fold_opponent()}
        res = run_showdown_comparison(
            suite,
            rounds=ROUNDS,
            episodes_per_round=EPISODES,
            delta=0.1,
            eps_safe=0.0,
            method="empirical_bernstein",
            rho_cap=rho,
            kappa=0.2,
            safety_debt_max=3.0,
            rnr_p=0.5,
            seeds=SEEDS,
            out_dir=None,
            workers=WORKERS,
        )
        od = res["opponents"]["censored_fold"]
        vref = od["game_value"]
        m = od["methods"]
        cell = {
            "rho_max": rho,
            "certified_floor": vref - rho,
            "gain": {k: m[k]["exploitation_gain_mean"] for k in TRACE},
            "gain_std": {k: m[k]["exploitation_gain_std"] for k in TRACE},
            "min_safety_value": {k: m[k]["min_safety_value"] for k in TRACE},
            "budget_spent": {k: m[k].get("budget_spent_mean", 0.0) for k in TRACE},
            "core_selected_rate": m["safe_active_decensoring"].get(
                "core_selected_rate_mean"
            ),
            "guarantee_held": all(m[k]["guarantee_held"] for k in TRACE),
        }
        cells.append(cell)
        sad = cell["gain"]["safe_active_decensoring"]
        held = "OK" if cell["guarantee_held"] else "BREACH"
        print(
            f"rho_max={rho:<5} floor={vref - rho:+.3f}  "
            f"SAD_gain={sad:+.3f}  minS={min(cell['min_safety_value'].values()):+.3f}  {held}",
            flush=True,
        )

    out = {
        "game": "holdem",
        "opponent": "censored_fold",
        "seeds": SEEDS,
        "rounds": ROUNDS,
        "episodes_per_round": EPISODES,
        "game_value": vref,
        "rho_grid": RHO_GRID,
        "trace_methods": TRACE,
        "cells": cells,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {OUT}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
