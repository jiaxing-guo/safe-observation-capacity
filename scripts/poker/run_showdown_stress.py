"""Run the showdown stress experiment. See Experiments and supplementary Certification at the Unbucketed River."""

import os

from safe_observation.experiments.online import (
    run_showdown_nonstationary,
    run_showdown_population,
)

SEEDS = list(range(42, 67))
ROUNDS = 60
EPISODES = 200
WORKERS = int(os.environ.get("WORKERS", "10"))
POP_N = int(os.environ.get("POP_N", "50"))
POP_SEED = int(os.environ.get("POP_SEED", "2026"))


def main() -> None:
    """Run the command-line entry point."""
    for kind in ("lure_then_strike", "drift"):
        run_showdown_nonstationary(
            kind=kind,
            lure="overfold",
            strike="calling_station",
            rounds=ROUNDS,
            episodes_per_round=EPISODES,
            delta=0.1,
            eps_safe=0.0,
            method="empirical_bernstein",
            rho_cap=0.5,
            kappa=0.2,
            safety_debt_max=3.0,
            rnr_p=0.5,
            game="holdem",
            seeds=SEEDS,
            workers=WORKERS,
            latch_threshold=None,
            out_dir="results",
        )
        print(f"  wrote results/showdown_nonstationary_{kind}_holdem.json", flush=True)

    run_showdown_population(
        n=POP_N,
        population_seed=POP_SEED,
        rounds=ROUNDS,
        episodes_per_round=EPISODES,
        delta=0.1,
        eps_safe=0.0,
        method="empirical_bernstein",
        rho_cap=0.5,
        kappa=0.2,
        safety_debt_max=3.0,
        rnr_p=0.5,
        game="holdem",
        seeds=SEEDS,
        latch_threshold=None,
        out_dir="results",
        workers=WORKERS,
    )
    print("  wrote results/showdown_population_holdem.json", flush=True)


if __name__ == "__main__":
    main()
