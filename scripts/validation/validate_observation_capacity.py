"""Validate observation capacity. See Safe Active De-censoring and supplementary Algorithms."""

import json
import statistics
import tempfile

from safe_observation import native
from safe_observation.experiments.online import (
    _parallel_map,
    _run_showdown_cell,
    _ShowdownCell,
)
from safe_observation.opponents import holdem_showdown_opponent_suite
from safe_observation.sequence_form import compile as compile_game
from safe_observation.solvers import safety_constrained_best_response, solve_blueprint

GAME = "holdem"
ROUNDS = 60
EPISODES = 200
DELTA = 0.1
METHOD = "empirical_bernstein"
EPS_SAFE = 0.0
RHO_CAP = 0.5
DEBT_MAX = 3.0
RNR_P = 0.5
SEEDS = list(range(42, 52))


KAPPAS_GUARD = [0.1, 0.2, 0.5, 1.0, 2.0, 10.0]
KAPPAS = [0.0, *KAPPAS_GUARD]
OPPS = [
    "censored_fold",
    "overfold",
    "low_reach_leak",
    "near_equilibrium",
    "calling_station",
    "maniac",
]
WORKERS = 10


def main() -> None:
    """Run the command-line entry point."""
    suite = holdem_showdown_opponent_suite()
    bp = solve_blueprint(GAME, method="lp")
    v_ref = bp.value
    assert bp.realization is not None
    x_bp = bp.realization
    sf0 = compile_game(GAME, 0)
    bp_behavior = sf0.behavior_from_realization(list(x_bp))

    cens: dict[str, dict[str, float]] = {}
    for name in OPPS:
        opp = suite[name]
        _pay, show, fold = native.simulate_showdown(
            GAME, bp_behavior, opp.behavior, 20000, 999
        )

        s = float(sum(sum(v) for v in show.values()))
        f = float(sum(sum(v) for v in fold.values()))
        cens[name] = {
            "showdown": s,
            "fold": f,
            "showdown_rate": s / (s + f) if (s + f) > 0 else 0.0,
        }

    scbr = {
        name: safety_constrained_best_response(
            suite[name].behavior, v_ref=v_ref, eps_safe=EPS_SAFE, game=GAME
        ).value
        for name in OPPS
    }
    cells = []
    for name in OPPS:
        for seed in SEEDS:
            cells.append(
                _ShowdownCell(
                    opp_key=f"{name}|k=0.0",
                    opponent=suite[name],
                    method_name="public_robust",
                    seed=seed,
                    scbr_value=scbr[name],
                    rounds=ROUNDS,
                    episodes_per_round=EPISODES,
                    v_ref=v_ref,
                    x_blueprint=x_bp,
                    delta=DELTA,
                    method=METHOD,
                    eps_safe=EPS_SAFE,
                    rho_cap=RHO_CAP,
                    kappa=0.0,
                    safety_debt_max=DEBT_MAX,
                    rnr_p=RNR_P,
                )
            )
        for kap in KAPPAS_GUARD:
            for seed in SEEDS:
                cells.append(
                    _ShowdownCell(
                        opp_key=f"{name}|k={kap}",
                        opponent=suite[name],
                        method_name="confidence_guarded",
                        seed=seed,
                        scbr_value=scbr[name],
                        rounds=ROUNDS,
                        episodes_per_round=EPISODES,
                        v_ref=v_ref,
                        x_blueprint=x_bp,
                        delta=DELTA,
                        method=METHOD,
                        eps_safe=EPS_SAFE,
                        rho_cap=RHO_CAP,
                        kappa=kap,
                        safety_debt_max=DEBT_MAX,
                        rnr_p=RNR_P,
                    )
                )

    flat = _parallel_map(_run_showdown_cell, cells, WORKERS)
    grouped: dict[str, list[float]] = {}
    for opp_key, _m, _seed, res in flat:
        grouped.setdefault(opp_key, []).append(res["exploitation_gain"])

    sweep: dict[str, dict[str, dict[str, float]]] = {}
    for name in OPPS:
        sweep[name] = {}
        for kap in KAPPAS:
            gains = grouped[f"{name}|k={kap}"]
            sweep[name][str(kap)] = {
                "mean": statistics.mean(gains),
                "std": statistics.pstdev(gains),
            }

    out = {
        "censoring": cens,
        "kappas": KAPPAS,
        "sweep": sweep,
        "v_ref": v_ref,
        "seeds": SEEDS,
    }
    with tempfile.NamedTemporaryFile(
        mode="w", prefix="capacity_validation_", suffix=".json", delete=False
    ) as fh:
        json.dump(out, fh, indent=2)
        report_path = fh.name

    print(f"saved {report_path}")

    hdr = "opponent           s_rate  " + "  ".join(f"k={k:>4}" for k in KAPPAS)
    print(hdr)
    print("-" * len(hdr))
    for name in OPPS:
        row = "  ".join(f"{sweep[name][str(k)]['mean']:+5.2f}" for k in KAPPAS)
        print(f"{name:18s} {cens[name]['showdown_rate']:.3f}   {row}")
    print()
    print("argmax-kappa per opponent (which kappa maximizes gain):")
    for name in OPPS:
        best = max(KAPPAS, key=lambda k: sweep[name][str(k)]["mean"])
        print(
            f"  {name:18s} best_kappa={best:>5}  "
            f"(gain {sweep[name][str(best)]['mean']:+.2f}, "
            f"s_rate {cens[name]['showdown_rate']:.3f})"
        )


if __name__ == "__main__":
    main()
