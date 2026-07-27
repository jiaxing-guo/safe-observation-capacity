"""Run the multiboard experiment. See Experiments and supplementary Certification at the Unbucketed River."""

import json
import os
from pathlib import Path
import sys

from safe_observation.experiments.online import run_showdown_comparison
from safe_observation.opponents import holdem_showdown_opponent_suite

BOARDS = [
    "holdem",
    "holdem_paired",
    "holdem_dry",
    "holdem_wet",
    "holdem_low",
]


KEY_OPPONENTS = [
    "equilibrium",
    "overfold",
    "censored_fold",
    "calling_station",
    "maniac",
    "fold_and_call",
]
SEEDS = list(range(42, 57))
ROUNDS = 60
EPISODES = 200
WORKERS = int(os.environ.get("WORKERS", "10"))
OUT = Path("results/multiboard_holdem.json")

CORE, POINT, SAD = "public_robust", "point_response", "safe_active_decensoring"


def main() -> None:
    """Run the command-line entry point."""
    boards_out = []
    for game in BOARDS:
        full = holdem_showdown_opponent_suite(game)
        suite = {k: full[k] for k in KEY_OPPONENTS if k in full}
        res = run_showdown_comparison(
            suite,
            rounds=ROUNDS,
            episodes_per_round=EPISODES,
            delta=0.1,
            eps_safe=0.0,
            method="empirical_bernstein",
            rho_cap=0.5,
            kappa=0.2,
            safety_debt_max=3.0,
            rnr_p=0.5,
            seeds=SEEDS,
            out_dir=None,
            workers=WORKERS,
        )
        vref = next(iter(res["opponents"].values()))["game_value"]
        rows = []
        for opp, od in res["opponents"].items():
            m = od["methods"]
            vc = m[CORE]["exploitation_gain_mean"]
            vp = m[POINT]["exploitation_gain_mean"]
            vx = m[SAD]["exploitation_gain_mean"]
            held = all(m[k]["guarantee_held"] for k in (CORE, POINT, SAD))
            rows.append(
                {
                    "opponent": opp,
                    "core": vc,
                    "point": vp,
                    "sad": vx,
                    "oracle_gap": max(vc, vp) - vx,
                    "core_selected_rate": m[SAD].get("core_selected_rate_mean"),
                    "min_safety_value": min(
                        m[k]["min_safety_value"] for k in (CORE, POINT, SAD)
                    ),
                    "guarantee_held": held,
                }
            )
        boards_out.append({"game": game, "game_value": vref, "rows": rows})
        worst = min(r["min_safety_value"] for r in rows)
        maxgap = max(r["oracle_gap"] for r in rows)
        allheld = all(r["guarantee_held"] for r in rows)
        print(
            f"{game:<16} vref={vref:+.3f}  maxOracleGap={maxgap:+.3f}  "
            f"worstMinS={worst:+.3f}  safety={'OK' if allheld else 'BREACH'}",
            flush=True,
        )

    out = {
        "boards": BOARDS,
        "key_opponents": KEY_OPPONENTS,
        "seeds": SEEDS,
        "rounds": ROUNDS,
        "episodes_per_round": EPISODES,
        "results": boards_out,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {OUT}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
