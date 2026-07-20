""

import argparse

from . import native
from .experiments import (
    run_config,
    run_kuhn_blueprint,
    run_online_adaptation,
    summarize,
)
from .opponents import opponent_suite
from .sequence_form import kuhn_sizes


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="safe-observation",
        description="Safe active de-censoring in imperfect-information games",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser(
        "info", help="Print native core version and Kuhn sequence-form sizes."
    )

    blueprint = sub.add_parser("blueprint", help="Solve the Kuhn blueprint.")
    blueprint.add_argument(
        "--method",
        choices=["lp", "cfr"],
        default="lp",
        help="Exact sequence-form LP (default) or CFR.",
    )
    blueprint.add_argument("--iterations", type=int, default=100_000)
    blueprint.add_argument("--out-dir", default="results")

    online = sub.add_parser(
        "online", help="Run online safe exploitation against an opponent."
    )
    online.add_argument(
        "--opponent",
        choices=sorted(opponent_suite()),
        default="static_biased",
        help="Opponent from the built-in suite.",
    )
    online.add_argument("--rounds", type=int, default=200)
    online.add_argument("--episodes-per-round", type=int, default=50)
    online.add_argument("--delta", type=float, default=0.05)
    online.add_argument("--eps-safe", type=float, default=0.0)
    online.add_argument(
        "--monitoring",
        choices=["full", "public"],
        default="full",
        help="Observability model: full per-card or public-state (Section 6).",
    )
    online.add_argument("--seed", type=int, default=2026)
    online.add_argument("--out-dir", default="results")

    runcmd = sub.add_parser(
        "run", help="Run an experiment from a TOML config (see configs/)."
    )
    runcmd.add_argument("config", help="Path to a TOML experiment config.")
    runcmd.add_argument(
        "--no-figures",
        action="store_true",
        help="Skip figure rendering even if the config requests it.",
    )

    args = parser.parse_args(argv)

    if args.command == "info":
        print(f"safe_observation_native version: {native.version()}")
        print(f"Kuhn sequence-form sizes: {kuhn_sizes()}")
        return 0

    if args.command == "blueprint":
        results = run_kuhn_blueprint(args.method, args.iterations, args.out_dir)
        line = (
            f"Kuhn blueprint value (player 1): {results['value_player1']:.6f} "
            f"(known {results['known_value']:.6f}, abs error {results['abs_error']:.2e})"
        )
        if "safety_floor" in results:
            line += f"; safety floor {results['safety_floor']:.6f}"
        print(line)
        return 0

    if args.command == "online":
        opponent = opponent_suite()[args.opponent]
        results = run_online_adaptation(
            opponent,
            rounds=args.rounds,
            episodes_per_round=args.episodes_per_round,
            delta=args.delta,
            eps_safe=args.eps_safe,
            monitoring=args.monitoring,
            seed=args.seed,
            out_dir=args.out_dir,
        )
        print(
            f"Online vs {results['opponent']}: "
            f"final actual value {results['final_actual_value']:.4f} "
            f"(game value {results['game_value']:.4f}, "
            f"gain {results['exploitation_gain']:+.4f}); "
            f"min safety {results['min_safety_value']:.4f}, "
            f"safe={results['safety_preserved']}"
        )
        return 0

    if args.command == "run":
        run = run_config(args.config, figures=False if args.no_figures else None)
        print(summarize(run))
        for fig_name, path in run.figures.items():
            print(f"  figure {fig_name} -> {path}")
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
