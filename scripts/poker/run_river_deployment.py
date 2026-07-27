"""Run the river deployment experiment. See Experiments and supplementary Certification at the Unbucketed River."""

import json
import os
from pathlib import Path

from safe_observation.experiments.online import run_showdown_comparison
from safe_observation.opponents import holdem_structured_opponent_suite

SEEDS = list(range(42, 42 + int(os.environ.get("RIVER_DEPLOYMENT_SEEDS", "10"))))
ROUNDS = int(os.environ.get("RIVER_DEPLOYMENT_ROUNDS", "60"))
EPISODES = int(os.environ.get("RIVER_DEPLOYMENT_EPISODES", "200"))
WORKERS = int(os.environ.get("WORKERS", "10"))
OUT = Path(
    os.environ.get("RIVER_DEPLOYMENT_OUT", "results/river_deployment_holdem.json")
)


def main() -> None:
    """Run the command-line entry point."""
    suite = holdem_structured_opponent_suite("holdem")
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
        latch_threshold=None,
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=2))

    arms = [
        "public_robust",
        "public_point",
        "censored_em",
        "point_response",
        "safe_active_decensoring",
    ]
    print(f"wrote {OUT}  ({len(suite)} opp x {len(SEEDS)} seeds)\n")
    print(
        f"{'opponent':<31}{'core':>8}{'pub':>8}{'em':>8}{'naive':>8}{'sad':>8}{'oracle_gap':>11}"
    )
    for opp, od in res["opponents"].items():
        m = od["methods"]
        vals = [m[a]["exploitation_gain_mean"] for a in arms]
        print(
            f"{opp:<31}"
            + "".join(f"{v:>8.3f}" for v in vals)
            + f"{od['oracle_gap']:>11.4f}"
        )


if __name__ == "__main__":
    main()
