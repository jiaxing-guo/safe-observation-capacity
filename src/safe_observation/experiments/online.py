"""Online primitives for safe-observation experiments. See Safe Active De-censoring, Experiments, and supplementary Game Instances and Experimental Setup."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import cache
from math import sqrt
from pathlib import Path
from statistics import mean, pstdev, stdev
from time import perf_counter
from typing import Any

from .. import evaluation, native
from ..agents import OnlineSafeExploitAgent
from ..confidence import OpponentEvidenceStore, build_confidence_set, public_key
from ..opponents import (
    Opponent,
    equilibrium_opponent,
    holdem_equilibrium_opponent,
    leduc_equilibrium_opponent,
    leduc_opponent_suite,
    opponent_from_spec,
    opponent_suite,
)
from ..payoff import build as build_payoff
from ..probe import ProbeBudget, SafetyBudgetLedger, weights_from_intervals
from ..sequence_form import compile as compile_game
from ..solvers import (
    best_response,
    confidence_guarded_point_probe,
    fallback_mixture_repair,
    floor_shadow_price,
    opponent_reach_weights,
    restricted_nash_response,
    robust_safe_response,
    robust_safe_response_probe,
    robust_safe_response_public,
    safety_constrained_best_response,
    safety_filtered_restricted_nash_response,
    safety_verifier,
    solve_blueprint,
)
from ..timing import StageTimer

KNOWN_KUHN_VALUE = -1.0 / 18.0


def run_online_adaptation(
    opponent: Opponent,
    rounds: int = 200,
    episodes_per_round: int = 50,
    delta: float = 0.05,
    eps_safe: float = 0.0,
    method: str = "hoeffding",
    monitoring: str = "full",
    probing: bool = False,
    beta: float = 0.0,
    probe_budget_total: float = 0.0,
    probe_per_round: float = float("inf"),
    importance_mode: str = "uniform",
    seed: int = 2026,
    out_dir: str | Path | None = "results",
) -> dict[str, Any]:
    """Run the online adaptation experiment."""
    game = opponent.game
    timer = StageTimer()
    budget = ProbeBudget(total=probe_budget_total, per_round=probe_per_round)
    with timer.stage("blueprint"):
        agent = OnlineSafeExploitAgent(
            game=game,
            delta=delta,
            eps_safe=eps_safe,
            method=method,
            monitoring=monitoring,
            timer=timer,
            probing=probing,
            beta=beta,
            probe_budget=budget,
            importance_mode=importance_mode,
        )
    game_value = agent.v_ref
    payoff = build_payoff(game)
    sf1 = compile_game(game, 1)
    y_star = list(opponent.realization())

    scbr = safety_constrained_best_response(
        opponent.behavior, v_ref=game_value, eps_safe=eps_safe, game=game
    )
    scbr_value = scbr.value

    rounds_log: list[dict[str, Any]] = []
    min_safety = float("inf")
    # Selection precedes observation in every round, preventing the current
    # batch from leaking into the policy used to collect it.
    for t in range(rounds):
        decision = agent.select()
        actual_value = payoff.bilinear(decision.realization, y_star)
        min_safety = min(min_safety, decision.safety_value)

        with timer.stage("empirical_br"):
            y_hat = sf1.realization_from_behavior(
                {
                    label: list(agent.evidence.p_hat(label))
                    for label in agent.evidence.labels
                }
            )
            empirical_br_value = best_response(y_hat, game=game).value

        with timer.stage("simulate"):
            total_payoff, p2_counts = native.simulate(
                game,
                decision.behavior,
                opponent.behavior,
                episodes_per_round,
                seed + t,
            )
        agent.observe(p2_counts)

        rounds_log.append(
            {
                "round": t,
                "actual_value": actual_value,
                "robust_value": decision.robust_value,
                "safety_value": decision.safety_value,
                "empirical_br_value": empirical_br_value,
                "mean_ci_width": decision.mean_ci_width,
                "ci_width_by_infoset": dict(decision.ci_width_by_infoset),
                "avg_payoff": total_payoff / episodes_per_round,
                "info_gain": decision.info_gain,
                "rho_granted": decision.rho_granted,
                "rho_spent": decision.rho_spent,
                "budget_spent": budget.spent,
                "scbr_gap": scbr_value - actual_value,
            }
        )

    # Report steady-state performance on the latter half of the trajectory.
    tail = rounds_log[max(0, len(rounds_log) // 2) :]
    final_actual = sum(r["actual_value"] for r in tail) / len(tail)
    final_scbr_gap = sum(r["scbr_gap"] for r in tail) / len(tail)

    budgeted_floor = game_value - eps_safe - probe_budget_total
    results: dict[str, Any] = {
        "game": game,
        "opponent": opponent.name,
        "rounds": rounds,
        "episodes_per_round": episodes_per_round,
        "delta": delta,
        "eps_safe": eps_safe,
        "method": method,
        "monitoring": monitoring,
        "probing": probing,
        "beta": beta,
        "probe_budget_total": probe_budget_total,
        "probe_per_round": probe_per_round,
        "importance_mode": importance_mode,
        "seed": seed,
        "game_value": game_value,
        "scbr_value": scbr_value,
        "scbr_gain": scbr_value - game_value,
        "final_scbr_gap": final_scbr_gap,
        "final_actual_value": final_actual,
        "exploitation_gain": final_actual - game_value,
        "exploitation_above_scbr": final_actual - scbr_value,
        "min_safety_value": min_safety,
        "safety_floor": game_value - eps_safe,
        "safety_preserved": min_safety >= game_value - eps_safe - 1e-8,
        "budget_spent": budget.spent,
        "budget_respected": budget.spent <= probe_budget_total + 1e-8,
        "budgeted_floor": budgeted_floor,
        "budget_floor_respected": min_safety >= budgeted_floor - 1e-8,
        "timings": timer.as_dict(),
        "rounds_log": rounds_log,
    }
    if out_dir is not None:
        prefix = "online" if game == "kuhn" else f"online_{game}"
        suffix = "_probe" if probing else ""
        evaluation.save_results(
            results, Path(out_dir) / f"{prefix}_{opponent.name}{suffix}.json"
        )
    return results


def run_suite(
    rounds: int = 200,
    episodes_per_round: int = 50,
    out_dir: str | Path | None = "results",
    **kwargs: Any,
) -> dict[str, dict[str, Any]]:
    """Run the suite experiment for the online workflow."""
    return {
        name: run_online_adaptation(
            opponent,
            rounds=rounds,
            episodes_per_round=episodes_per_round,
            out_dir=out_dir,
            **kwargs,
        )
        for name, opponent in opponent_suite().items()
    }


DEFAULT_SEEDS: tuple[int, ...] = (42, 43, 44, 45, 46)


_GIFT_GUARANTEE_TOL = 1e-6


_SAFETY_GUARANTEE_TOL = 1e-6

_SERIES_KEYS = (
    "actual_value",
    "robust_value",
    "safety_value",
    "empirical_br_value",
    "mean_ci_width",
    "avg_payoff",
    "scbr_gap",
)


def _aggregate_rounds(
    per_seed_logs: list[list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    """Aggregate rounds for the online workflow."""
    n_rounds = min(len(log) for log in per_seed_logs)
    labels = list(per_seed_logs[0][0]["ci_width_by_infoset"])
    aggregated: list[dict[str, Any]] = []
    for t in range(n_rounds):
        row: dict[str, Any] = {"round": t}
        for key in _SERIES_KEYS:
            vals = [log[t][key] for log in per_seed_logs]
            row[f"{key}_mean"] = mean(vals)
            row[f"{key}_std"] = pstdev(vals) if len(vals) > 1 else 0.0
        row["ci_width_by_infoset_mean"] = {
            label: mean([log[t]["ci_width_by_infoset"][label] for log in per_seed_logs])
            for label in labels
        }
        aggregated.append(row)
    return aggregated


def run_online_adaptation_replicated(
    opponent: Opponent,
    rounds: int = 300,
    episodes_per_round: int = 50,
    delta: float = 0.05,
    eps_safe: float = 0.0,
    method: str = "hoeffding",
    monitoring: str = "full",
    seeds: Sequence[int] = DEFAULT_SEEDS,
    out_dir: str | Path | None = "results",
) -> dict[str, Any]:
    """Run the online adaptation replicated experiment."""
    per_seed = [
        run_online_adaptation(
            opponent,
            rounds=rounds,
            episodes_per_round=episodes_per_round,
            delta=delta,
            eps_safe=eps_safe,
            method=method,
            monitoring=monitoring,
            seed=seed,
            out_dir=None,
        )
        for seed in seeds
    ]
    gains = [r["exploitation_gain"] for r in per_seed]
    game = opponent.game
    game_value = per_seed[0]["game_value"]
    scbr_value = per_seed[0]["scbr_value"]
    timer = StageTimer()
    for r in per_seed:
        for name, st in r["timings"].items():
            timer.add_totals(name, st["seconds"], int(st["calls"]))
    results: dict[str, Any] = {
        "game": game,
        "opponent": opponent.name,
        "seeds": list(seeds),
        "rounds": rounds,
        "episodes_per_round": episodes_per_round,
        "delta": delta,
        "eps_safe": eps_safe,
        "method": method,
        "monitoring": monitoring,
        "game_value": game_value,
        "scbr_value": scbr_value,
        "scbr_gain": scbr_value - game_value,
        "exploitation_gain_mean": mean(gains),
        "exploitation_gain_std": pstdev(gains) if len(gains) > 1 else 0.0,
        "min_safety_value": min(r["min_safety_value"] for r in per_seed),
        "safety_floor": game_value - eps_safe,
        "safety_preserved": all(r["safety_preserved"] for r in per_seed),
        "timings": timer.as_dict(),
        "per_seed": [
            {
                "seed": r["seed"],
                "final_actual_value": r["final_actual_value"],
                "exploitation_gain": r["exploitation_gain"],
                "min_safety_value": r["min_safety_value"],
                "safety_preserved": r["safety_preserved"],
            }
            for r in per_seed
        ],
        "aggregated_rounds": _aggregate_rounds([r["rounds_log"] for r in per_seed]),
    }
    if out_dir is not None:
        prefix = "online" if game == "kuhn" else f"online_{game}"
        evaluation.save_results(
            results, Path(out_dir) / f"{prefix}_{opponent.name}_replicated.json"
        )
    return results


def _target_ci_series(
    rounds_log: list[dict[str, Any]], target_labels: Sequence[str]
) -> list[float]:
    """Compute target confidence interval series."""
    targets = [label for label in target_labels]
    series = []
    for row in rounds_log:
        widths = row["ci_width_by_infoset"]
        vals = [widths[label] for label in targets if label in widths]
        series.append(sum(vals) / len(vals) if vals else 0.0)
    return series


def _mean_std_series(per_seed: list[list[float]]) -> list[dict[str, float]]:
    """Compute mean std series for the online workflow."""
    n = min(len(s) for s in per_seed)
    out = []
    for t in range(n):
        vals = [s[t] for s in per_seed]
        out.append({"mean": mean(vals), "std": pstdev(vals) if len(vals) > 1 else 0.0})
    return out


def run_probing_comparison(
    opponent: Opponent,
    arms: Mapping[str, Mapping[str, Any]] | None = None,
    rounds: int = 60,
    episodes_per_round: int = 200,
    delta: float = 0.05,
    eps_safe: float = 0.0,
    method: str = "hoeffding",
    seeds: Sequence[int] = DEFAULT_SEEDS,
    target_labels: Sequence[str] | None = None,
    out_dir: str | Path | None = "results",
) -> dict[str, Any]:
    """Run the probing comparison experiment."""
    game = opponent.game
    if arms is None:
        arms = {
            "passive": {"probing": False},
            "probe_hard_safe": {
                "probing": True,
                "beta": 2.0,
                "probe_budget_total": 0.0,
            },
            "probe_budget_only": {
                "probing": True,
                "beta": 0.0,
                "probe_budget_total": 0.5 * rounds,
                "probe_per_round": 0.5,
            },
            "probe_budgeted": {
                "probing": True,
                "beta": 2.0,
                "probe_budget_total": 0.5 * rounds,
                "probe_per_round": 0.5,
            },
        }
    if target_labels is None and game == "leduc":
        from ..opponents import _leduc_cap_facing_labels

        target_labels = _leduc_cap_facing_labels()
    targets = list(target_labels) if target_labels is not None else []

    arm_results: dict[str, Any] = {}
    game_value = 0.0
    scbr_value = 0.0
    for name, overrides in arms.items():
        per_seed = [
            run_online_adaptation(
                opponent,
                rounds=rounds,
                episodes_per_round=episodes_per_round,
                delta=delta,
                eps_safe=eps_safe,
                method=method,
                seed=seed,
                out_dir=None,
                **overrides,
            )
            for seed in seeds
        ]
        game_value = per_seed[0]["game_value"]
        scbr_value = per_seed[0]["scbr_value"]
        gains = [r["exploitation_gain"] for r in per_seed]
        actual_series = [
            [row["actual_value"] for row in r["rounds_log"]] for r in per_seed
        ]
        ig_series = [[row["info_gain"] for row in r["rounds_log"]] for r in per_seed]
        budget_series = [
            [row["budget_spent"] for row in r["rounds_log"]] for r in per_seed
        ]
        ci_series = [_target_ci_series(r["rounds_log"], targets) for r in per_seed]
        arm_results[name] = {
            **{k: overrides.get(k) for k in overrides},
            "exploitation_gain_mean": mean(gains),
            "exploitation_gain_std": pstdev(gains) if len(gains) > 1 else 0.0,
            "exploitation_above_scbr_mean": mean(
                r["exploitation_above_scbr"] for r in per_seed
            ),
            "min_safety_value": min(r["min_safety_value"] for r in per_seed),
            "budget_spent_mean": mean(r["budget_spent"] for r in per_seed),
            "budget_respected": all(r["budget_respected"] for r in per_seed),
            "budget_floor_respected": all(
                r["budget_floor_respected"] for r in per_seed
            ),
            "actual_value_by_round": _mean_std_series(actual_series),
            "info_gain_by_round": _mean_std_series(ig_series),
            "budget_spent_by_round": _mean_std_series(budget_series),
            "target_ci_by_round": _mean_std_series(ci_series),
        }

    results: dict[str, Any] = {
        "game": game,
        "opponent": opponent.name,
        "seeds": list(seeds),
        "rounds": rounds,
        "episodes_per_round": episodes_per_round,
        "delta": delta,
        "eps_safe": eps_safe,
        "method": method,
        "game_value": game_value,
        "scbr_value": scbr_value,
        "scbr_gain": scbr_value - game_value,
        "n_target_labels": len(targets),
        "arms": arm_results,
    }
    if out_dir is not None:
        prefix = "probing" if game == "kuhn" else f"probing_{game}"
        evaluation.save_results(
            results, Path(out_dir) / f"{prefix}_{opponent.name}.json"
        )
    return results


def run_budget_frontier(
    opponent: Opponent,
    budgets: Sequence[float] = (0.0, 0.1, 0.25, 0.5, 1.0, 2.0),
    beta: float = 2.0,
    rounds: int = 60,
    episodes_per_round: int = 200,
    delta: float = 0.05,
    eps_safe: float = 0.0,
    method: str = "hoeffding",
    seeds: Sequence[int] = DEFAULT_SEEDS,
    out_dir: str | Path | None = "results",
) -> dict[str, Any]:
    """Run the budget frontier experiment."""
    game = opponent.game
    cells: list[dict[str, Any]] = []
    scbr_value = 0.0
    game_value = 0.0
    for rho in budgets:
        per_seed = [
            run_online_adaptation(
                opponent,
                rounds=rounds,
                episodes_per_round=episodes_per_round,
                delta=delta,
                eps_safe=eps_safe,
                method=method,
                probing=rho > 0.0,
                beta=beta,
                probe_budget_total=rho * rounds,
                probe_per_round=rho,
                seed=seed,
                out_dir=None,
            )
            for seed in seeds
        ]
        game_value = per_seed[0]["game_value"]
        scbr_value = per_seed[0]["scbr_value"]
        gains = [r["exploitation_gain"] for r in per_seed]
        above = [r["exploitation_above_scbr"] for r in per_seed]
        spent = [r["budget_spent"] for r in per_seed]
        spent_mean = mean(spent)
        above_mean = mean(above)
        cells.append(
            {
                "per_round_budget": rho,
                "budget_total": rho * rounds,
                "exploitation_gain_mean": mean(gains),
                "exploitation_gain_std": pstdev(gains) if len(gains) > 1 else 0.0,
                "exploitation_above_scbr_mean": above_mean,
                "budget_spent_mean": spent_mean,
                "min_safety_value": min(r["min_safety_value"] for r in per_seed),
                "budget_floor_respected": all(
                    r["budget_floor_respected"] for r in per_seed
                ),
                "value_per_budget": (above_mean / spent_mean)
                if spent_mean > 1e-9
                else 0.0,
            }
        )

    results: dict[str, Any] = {
        "game": game,
        "opponent": opponent.name,
        "seeds": list(seeds),
        "rounds": rounds,
        "episodes_per_round": episodes_per_round,
        "delta": delta,
        "eps_safe": eps_safe,
        "method": method,
        "beta": beta,
        "game_value": game_value,
        "scbr_value": scbr_value,
        "scbr_gain": scbr_value - game_value,
        "budgets": list(budgets),
        "cells": cells,
    }
    if out_dir is not None:
        prefix = "budget_frontier" if game == "kuhn" else f"budget_frontier_{game}"
        evaluation.save_results(
            results, Path(out_dir) / f"{prefix}_{opponent.name}.json"
        )
    return results


def run_importance_comparison(
    opponent: Opponent,
    budgets: Sequence[float] = (0.05, 0.1, 0.25, 0.5),
    beta: float = 2.0,
    rounds: int = 60,
    episodes_per_round: int = 200,
    delta: float = 0.05,
    eps_safe: float = 0.0,
    method: str = "hoeffding",
    seeds: Sequence[int] = DEFAULT_SEEDS,
    out_dir: str | Path | None = "results",
) -> dict[str, Any]:
    """Run the importance comparison experiment."""
    game = opponent.game
    cells: list[dict[str, Any]] = []
    scbr_value = 0.0
    game_value = 0.0
    for rho in budgets:
        modes: dict[str, Any] = {}
        for mode in ("uniform", "sensitivity"):
            per_seed = [
                run_online_adaptation(
                    opponent,
                    rounds=rounds,
                    episodes_per_round=episodes_per_round,
                    delta=delta,
                    eps_safe=eps_safe,
                    method=method,
                    probing=True,
                    beta=beta,
                    probe_budget_total=rho * rounds,
                    probe_per_round=rho,
                    importance_mode=mode,
                    seed=seed,
                    out_dir=None,
                )
                for seed in seeds
            ]
            game_value = per_seed[0]["game_value"]
            scbr_value = per_seed[0]["scbr_value"]
            gains = [r["exploitation_gain"] for r in per_seed]
            above = [r["exploitation_above_scbr"] for r in per_seed]
            spent = [r["budget_spent"] for r in per_seed]
            spent_mean = mean(spent)
            above_mean = mean(above)
            modes[mode] = {
                "exploitation_gain_mean": mean(gains),
                "exploitation_gain_std": pstdev(gains) if len(gains) > 1 else 0.0,
                "exploitation_above_scbr_mean": above_mean,
                "budget_spent_mean": spent_mean,
                "min_safety_value": min(r["min_safety_value"] for r in per_seed),
                "budget_floor_respected": all(
                    r["budget_floor_respected"] for r in per_seed
                ),
                "value_per_budget": (
                    above_mean / spent_mean if spent_mean > 1e-9 else 0.0
                ),
            }
        cells.append(
            {
                "per_round_budget": rho,
                "uniform": modes["uniform"],
                "sensitivity": modes["sensitivity"],
                "sensitivity_advantage": (
                    modes["sensitivity"]["exploitation_gain_mean"]
                    - modes["uniform"]["exploitation_gain_mean"]
                ),
            }
        )

    results: dict[str, Any] = {
        "game": game,
        "opponent": opponent.name,
        "seeds": list(seeds),
        "rounds": rounds,
        "episodes_per_round": episodes_per_round,
        "delta": delta,
        "eps_safe": eps_safe,
        "method": method,
        "beta": beta,
        "game_value": game_value,
        "scbr_value": scbr_value,
        "scbr_gain": scbr_value - game_value,
        "budgets": list(budgets),
        "cells": cells,
    }
    if out_dir is not None:
        prefix = "importance" if game == "kuhn" else f"importance_{game}"
        evaluation.save_results(
            results, Path(out_dir) / f"{prefix}_{opponent.name}.json"
        )
    return results


def run_probing_suite(
    rounds: int = 60,
    episodes_per_round: int = 200,
    beta: float = 2.0,
    per_round_budget: float = 0.5,
    delta: float = 0.05,
    eps_safe: float = 0.0,
    method: str = "hoeffding",
    seeds: Sequence[int] = DEFAULT_SEEDS,
    opponents: Mapping[str, Opponent] | None = None,
    out_dir: str | Path | None = "results",
) -> dict[str, Any]:
    """Run the probing suite experiment."""
    from ..opponents import leduc_opponent_suite

    suite = dict(opponents) if opponents is not None else leduc_opponent_suite()
    arms = {
        "passive": {"probing": False},
        "probe_budgeted": {
            "probing": True,
            "beta": beta,
            "probe_budget_total": per_round_budget * rounds,
            "probe_per_round": per_round_budget,
        },
    }

    rows: dict[str, Any] = {}
    for opp_name, opponent in suite.items():
        arm_out: dict[str, Any] = {}
        scbr_value = 0.0
        game_value = 0.0
        for arm_name, overrides in arms.items():
            per_seed = [
                run_online_adaptation(
                    opponent,
                    rounds=rounds,
                    episodes_per_round=episodes_per_round,
                    delta=delta,
                    eps_safe=eps_safe,
                    method=method,
                    seed=seed,
                    out_dir=None,
                    **overrides,
                )
                for seed in seeds
            ]
            game_value = per_seed[0]["game_value"]
            scbr_value = per_seed[0]["scbr_value"]
            gains = [r["exploitation_gain"] for r in per_seed]
            arm_out[arm_name] = {
                "exploitation_gain_mean": mean(gains),
                "exploitation_gain_std": pstdev(gains) if len(gains) > 1 else 0.0,
                "min_safety_value": min(r["min_safety_value"] for r in per_seed),
                "budget_spent_mean": mean(r["budget_spent"] for r in per_seed),
                "budget_floor_respected": all(
                    r["budget_floor_respected"] for r in per_seed
                ),
            }
        rows[opp_name] = {
            "game_value": game_value,
            "scbr_value": scbr_value,
            "scbr_gain": scbr_value - game_value,
            "arms": arm_out,
            "probing_helps": (
                arm_out["probe_budgeted"]["exploitation_gain_mean"]
                > arm_out["passive"]["exploitation_gain_mean"] + 1e-3
            ),
        }

    results: dict[str, Any] = {
        "game": "leduc",
        "seeds": list(seeds),
        "rounds": rounds,
        "episodes_per_round": episodes_per_round,
        "delta": delta,
        "eps_safe": eps_safe,
        "method": method,
        "beta": beta,
        "per_round_budget": per_round_budget,
        "opponents": rows,
    }
    if out_dir is not None:
        evaluation.save_results(results, Path(out_dir) / "probing_suite_leduc.json")
    return results


_COVERAGE_ARMS = ("no_union", "spatial", "time_uniform")


def _build_intervals_for_arm(
    store: OpponentEvidenceStore, arm: str, delta: float, method: str, round_index: int
):
    """Build intervals for arm for the online workflow."""
    if arm == "no_union":
        return store.intervals(delta, method=method, union_bound=False)
    if arm == "spatial":
        return store.intervals(delta, method=method, union_bound=True)
    if arm == "time_uniform":
        return store.intervals(
            delta, method=method, union_bound=True, round_index=round_index
        )
    raise ValueError(f"unknown coverage arm {arm!r}")


def run_coverage_experiment(
    opponent: Opponent,
    deltas: Sequence[float] = (0.05, 0.1, 0.2),
    rounds: int = 40,
    episodes_per_round: int = 50,
    method: str = "hoeffding",
    seeds: Sequence[int] = DEFAULT_SEEDS,
    out_dir: str | Path | None = "results",
) -> dict[str, Any]:
    """Run the coverage experiment experiment."""
    game = opponent.game
    y_star = list(opponent.realization())
    blueprint_behavior = compile_game(game, 0).behavior_from_realization(
        native.blueprint_realization(game, 0)
    )

    cells: list[dict[str, Any]] = []
    for delta in deltas:
        never_violated = {arm: 0 for arm in _COVERAGE_ARMS}
        per_round_hits = {arm: 0 for arm in _COVERAGE_ARMS}
        for seed in seeds:
            store = OpponentEvidenceStore.for_game(game)
            violated = {arm: False for arm in _COVERAGE_ARMS}
            for t in range(1, rounds + 1):
                _, p2_counts = native.simulate(
                    game,
                    blueprint_behavior,
                    opponent.behavior,
                    episodes_per_round,
                    seed * 100_000 + t,
                )
                for label, counts in p2_counts.items():
                    store.record(label, counts)
                for arm in _COVERAGE_ARMS:
                    intervals = _build_intervals_for_arm(store, arm, delta, method, t)
                    inside = build_confidence_set(game, intervals).contains(y_star)
                    per_round_hits[arm] += int(inside)
                    if not inside:
                        violated[arm] = True
            for arm in _COVERAGE_ARMS:
                if not violated[arm]:
                    never_violated[arm] += 1
        n_checks = len(seeds) * rounds
        cells.append(
            {
                "delta": delta,
                "guarantee": 1.0 - delta,
                "arms": {
                    arm: {
                        "anytime_coverage": never_violated[arm] / len(seeds),
                        "per_round_coverage": per_round_hits[arm] / n_checks,
                        "meets_guarantee": (never_violated[arm] / len(seeds))
                        >= 1.0 - delta - 1e-9,
                    }
                    for arm in _COVERAGE_ARMS
                },
            }
        )

    results: dict[str, Any] = {
        "game": game,
        "opponent": opponent.name,
        "seeds": list(seeds),
        "rounds": rounds,
        "episodes_per_round": episodes_per_round,
        "method": method,
        "deltas": list(deltas),
        "cells": cells,
    }
    if out_dir is not None:
        prefix = "coverage" if game == "kuhn" else f"coverage_{game}"
        evaluation.save_results(
            results, Path(out_dir) / f"{prefix}_{opponent.name}.json"
        )
    return results


def run_finite_sample_gap(
    opponent: Opponent,
    rounds: int = 60,
    episodes_per_round: int = 100,
    delta: float = 0.05,
    eps_safe: float = 0.0,
    method: str = "hoeffding",
    seeds: Sequence[int] = DEFAULT_SEEDS,
    out_dir: str | Path | None = "results",
) -> dict[str, Any]:
    """Run the finite sample gap experiment."""
    rep = run_online_adaptation_replicated(
        opponent,
        rounds=rounds,
        episodes_per_round=episodes_per_round,
        delta=delta,
        eps_safe=eps_safe,
        method=method,
        seeds=seeds,
        out_dir=None,
    )
    series: list[dict[str, Any]] = []
    ratios: list[float] = []
    for row in rep["aggregated_rounds"]:
        radius = row["mean_ci_width_mean"] / 2.0
        gap = row["scbr_gap_mean"]
        ratio = gap / radius if radius > 1e-9 else 0.0
        if radius > 1e-9:
            ratios.append(ratio)
        series.append(
            {
                "round": row["round"],
                "radius": radius,
                "scbr_gap": gap,
                "ratio": ratio,
            }
        )
    half = max(1, len(series) // 2)
    early_gap = mean(r["scbr_gap"] for r in series[:half])
    late_gap = mean(r["scbr_gap"] for r in series[half:])
    results: dict[str, Any] = {
        "game": opponent.game,
        "opponent": opponent.name,
        "seeds": list(seeds),
        "rounds": rounds,
        "episodes_per_round": episodes_per_round,
        "delta": delta,
        "eps_safe": eps_safe,
        "method": method,
        "scbr_gain": rep["scbr_gain"],
        "early_gap_mean": early_gap,
        "late_gap_mean": late_gap,
        "gap_decreased": late_gap <= early_gap + 1e-9,
        "bound_constant": max(ratios) if ratios else 0.0,
        "series": series,
    }
    if out_dir is not None:
        prefix = (
            "finite_gap" if opponent.game == "kuhn" else f"finite_gap_{opponent.game}"
        )
        evaluation.save_results(
            results, Path(out_dir) / f"{prefix}_{opponent.name}.json"
        )
    return results


def _empirical_realization(evidence: OpponentEvidenceStore, sf1) -> list[float]:
    """Compute empirical realization for the online workflow."""
    return list(
        sf1.realization_from_behavior(
            {label: list(evidence.p_hat(label)) for label in evidence.labels}
        )
    )


class _Method:
    """Represent method for the online workflow."""

    name = "method"
    guarantee = "none"
    adaptive = True

    def select(self, t: int, evidence: OpponentEvidenceStore) -> list[float]:
        """Select the next floor-safe response."""
        raise NotImplementedError

    def observe(self, realized_value: float, v_ref: float) -> None:
        """Update state from newly observed opponent actions."""

    def extra(self) -> dict[str, Any]:
        """Return auxiliary diagnostics for the current policy."""
        return {}


def _baseline_methods(
    game: str,
    v_ref: float,
    x_blueprint: tuple[float, ...],
    delta: float,
    method: str,
    eps_safe: float,
    rnr_ps: Sequence[float],
    beta: float,
    probe_per_round: float,
    rounds: int,
    sf0,
    sf1,
    shadow_tau: float = 0.3,
    safety_debt_max: float = 3.0,
    safety_signal_scale: float = 1.0,
    safety_budget_gamma: float = 1.0,
    micro_probe_budget: float = 0.0,
    micro_probe_threshold: float = 0.5,
) -> list[tuple[str, str, Any]]:
    """Compute baseline methods for the online workflow."""

    class Blueprint(_Method):
        """Represent blueprint for the online workflow."""

        name = "blueprint"
        guarantee = "hard_safe"
        adaptive = False

        def select(self, t, evidence):
            """Select the next floor-safe response."""
            return list(x_blueprint)

    class EmpiricalBR(_Method):
        """Represent empirical br for the online workflow."""

        name = "empirical_br"

        def select(self, t, evidence):
            """Select the next floor-safe response."""
            y_hat = _empirical_realization(evidence, sf1)
            return list(best_response(y_hat, game=game).realization)

    class RNR(_Method):
        """Represent restricted Nash response for the online workflow."""

        def __init__(self, p):
            """Initialize the restricted Nash response."""
            self.p = p
            self.name = f"rnr_p{p:g}"

        def select(self, t, evidence):
            """Select the next floor-safe response."""
            y_hat = _empirical_realization(evidence, sf1)
            return list(restricted_nash_response(y_hat, self.p, game=game).realization)

    class Passive(_Method):
        """Represent passive for the online workflow."""

        name = "passive"
        guarantee = "hard_safe"

        def select(self, t, evidence):
            """Select the next floor-safe response."""
            intervals = evidence.intervals(delta, method=method)
            return list(
                robust_safe_response(
                    intervals, v_ref=v_ref, eps_safe=eps_safe, game=game
                ).realization
            )

    class ProbeBudgeted(_Method):
        """Represent probe budgeted for the online workflow."""

        name = "probe_budgeted"
        guarantee = "certified_budget"

        def __init__(self):
            """Initialize the probe budgeted for the online workflow."""
            self.budget = ProbeBudget(
                total=probe_per_round * rounds, per_round=probe_per_round
            )

        def select(self, t, evidence):
            """Select the next floor-safe response."""
            intervals = evidence.intervals(delta, method=method)
            opp_behavior = {
                label: list(evidence.p_hat(label)) for label in evidence.labels
            }
            weights = weights_from_intervals(intervals)
            rho = self.budget.allowance()
            resp = robust_safe_response_probe(
                intervals,
                opp_behavior,
                weights,
                v_ref=v_ref,
                eps_safe=eps_safe,
                beta=beta,
                rho=rho,
                game=game,
            )
            x = resp.realization
            safety = safety_verifier(x, game=game).value
            self.budget.charge(max(0.0, (v_ref - eps_safe) - safety))
            return list(x)

        def extra(self):
            """Return auxiliary diagnostics for the current policy."""
            return {"budget_spent": self.budget.spent}

    class SafetyFilteredRNR(_Method):
        """Represent safety filtered restricted Nash response."""

        name = "safety_filtered_rnr"
        guarantee = "certified_budget"

        def __init__(self):
            """Initialize the safety filtered restricted Nash response."""
            self.budget = ProbeBudget(
                total=probe_per_round * rounds, per_round=probe_per_round
            )
            self._p_sum = 0.0
            self._rounds = 0

        def select(self, t, evidence):
            """Select the next floor-safe response."""
            y_hat = _empirical_realization(evidence, sf1)
            rho = self.budget.allowance()
            resp = safety_filtered_restricted_nash_response(
                y_hat, floor=v_ref - eps_safe - rho, game=game
            )
            self.budget.charge(max(0.0, (v_ref - eps_safe) - resp.safety_value))
            self._p_sum += resp.p
            self._rounds += 1
            return list(resp.realization)

        def extra(self):
            """Return auxiliary diagnostics for the current policy."""
            return {
                "budget_spent": self.budget.spent,
                "mean_p_star": self._p_sum / max(1, self._rounds),
            }

    class BudgetedEmpiricalBR(_Method):
        """Represent budgeted empirical br for the online workflow."""

        name = "budgeted_empirical_br"
        guarantee = "certified_budget"

        def __init__(self):
            """Initialize the budgeted empirical br."""
            self.budget = ProbeBudget(
                total=probe_per_round * rounds, per_round=probe_per_round
            )

        def select(self, t, evidence):
            """Select the next floor-safe response."""
            rho = self.budget.allowance()
            opp_behavior = {
                label: list(evidence.p_hat(label)) for label in evidence.labels
            }
            resp = safety_constrained_best_response(
                opp_behavior, v_ref=v_ref, eps_safe=eps_safe + rho, game=game
            )
            x = resp.realization
            safety = safety_verifier(x, game=game).value
            self.budget.charge(max(0.0, (v_ref - eps_safe) - safety))
            return list(x)

        def extra(self):
            """Return auxiliary diagnostics for the current policy."""
            return {"budget_spent": self.budget.spent}

    class ValueAwareBR(_Method):
        """Represent value aware br for the online workflow."""

        name = "value_aware_br"
        guarantee = "certified_budget"

        def __init__(self):
            """Initialize the value aware br."""
            self.budget = ProbeBudget(
                total=probe_per_round * rounds, per_round=probe_per_round
            )
            self._gate_sum = 0.0
            self._shadow_sum = 0.0
            self._rounds = 0

        def select(self, t, evidence):
            """Select the next floor-safe response."""
            intervals = evidence.intervals(delta, method=method)
            shadow = floor_shadow_price(
                intervals, v_ref=v_ref, eps_safe=eps_safe, game=game
            )
            self._shadow_sum += shadow
            self._rounds += 1
            rho = self.budget.allowance() if shadow > shadow_tau else 0.0
            if rho <= 0.0:
                return list(x_blueprint)
            self._gate_sum += 1.0
            opp_behavior = {
                label: list(evidence.p_hat(label)) for label in evidence.labels
            }
            resp = safety_constrained_best_response(
                opp_behavior, v_ref=v_ref, eps_safe=eps_safe + rho, game=game
            )
            x = resp.realization
            safety = safety_verifier(x, game=game).value
            self.budget.charge(max(0.0, (v_ref - eps_safe) - safety))
            return list(x)

        def extra(self):
            """Return auxiliary diagnostics for the current policy."""
            return {
                "budget_spent": self.budget.spent,
                "gate_rate": self._gate_sum / max(1, self._rounds),
                "mean_shadow": self._shadow_sum / max(1, self._rounds),
            }

    class PointResponse(_Method):
        """Represent point response for the online workflow."""

        name = "point_response"
        guarantee = "certified_budget"

        def __init__(self):
            """Initialize the point response for the online workflow."""
            self.budget = SafetyBudgetLedger(
                rho_cap=probe_per_round,
                hard_total=probe_per_round * rounds,
                debt_max=safety_debt_max,
                s_scale=safety_signal_scale,
                gamma=safety_budget_gamma,
                micro_rho=micro_probe_budget,
                tau_point=micro_probe_threshold,
            )
            self._pending_spend = 0.0
            self._shadow_sum = 0.0
            self._spend_rounds = 0
            self._micro_rounds = 0
            self._rounds = 0

        def select(self, t, evidence):
            """Select the next floor-safe response."""
            intervals = evidence.intervals(delta, method=method)
            shadow = floor_shadow_price(
                intervals, v_ref=v_ref, eps_safe=eps_safe, game=game
            )
            self._shadow_sum += shadow
            self._rounds += 1

            point_signal = 0.0
            if (
                self.budget.micro_rho > 0.0
                and self.budget.raw_grant(shadow) < self.budget.rho_cap - 1e-9
            ):
                y_hat = _empirical_realization(evidence, sf1)
                point_signal = best_response(y_hat, game=game).value - v_ref
                if point_signal > self.budget.tau_point:
                    self._micro_rounds += 1
            rho = self.budget.allowance(shadow, point_signal)
            if rho <= 0.0:
                self._pending_spend = 0.0
                return list(x_blueprint)
            self._spend_rounds += 1
            opp_behavior = {
                label: list(evidence.p_hat(label)) for label in evidence.labels
            }
            resp = safety_constrained_best_response(
                opp_behavior, v_ref=v_ref, eps_safe=eps_safe + rho, game=game
            )
            x = resp.realization
            spent = max(0.0, (v_ref - eps_safe) - safety_verifier(x, game=game).value)
            self.budget.charge(spent)
            self._pending_spend = spent
            return list(x)

        def observe(self, realized_value, v_ref):
            """Update state from newly observed opponent actions."""
            self.budget.settle(self._pending_spend, realized_value - v_ref)
            self._pending_spend = 0.0

        def extra(self):
            """Return auxiliary diagnostics for the current policy."""
            return {
                "budget_spent": self.budget.spent,
                "safety_debt_final": self.budget.debt,
                "safety_spend_rate": self._spend_rounds / max(1, self._rounds),
                "micro_probe_rate": self._micro_rounds / max(1, self._rounds),
                "mean_shadow": self._shadow_sum / max(1, self._rounds),
            }

    class GiftBased(_Method):
        """Represent gift based for the online workflow."""

        name = "gift_based"
        guarantee = "gift_funded"

        def __init__(self):
            """Initialize the gift based for the online workflow."""
            self.bankroll = 0.0
            self.min_bankroll = 0.0

        def select(self, t, evidence):
            """Select the next floor-safe response."""
            if self.bankroll <= 1e-12:
                return list(x_blueprint)
            y_hat = _empirical_realization(evidence, sf1)
            x_br = best_response(y_hat, game=game).realization

            x = fallback_mixture_repair(
                x_blueprint, x_br, v_ref, eps_safe=self.bankroll, game=game
            )
            return list(x)

        def observe(self, realized_value, v_ref):
            """Update state from newly observed opponent actions."""
            self.bankroll += realized_value - v_ref
            self.min_bankroll = min(self.min_bankroll, self.bankroll)

        def extra(self):
            """Return auxiliary diagnostics for the current policy."""
            return {
                "gift_bankroll_final": self.bankroll,
                "gift_min_bankroll": self.min_bankroll,
            }

    lineup: list[tuple[str, str, Any]] = [
        ("blueprint", "hard_safe", Blueprint),
        ("empirical_br", "none", EmpiricalBR),
    ]
    lineup += [(f"rnr_p{p:g}", "none", (lambda p=p: RNR(p))) for p in rnr_ps]
    lineup += [
        ("gift_based", "gift_funded", GiftBased),
        ("passive", "hard_safe", Passive),
        ("probe_budgeted", "certified_budget", ProbeBudgeted),
        ("safety_filtered_rnr", "certified_budget", SafetyFilteredRNR),
        ("budgeted_empirical_br", "certified_budget", BudgetedEmpiricalBR),
        ("value_aware_br", "certified_budget", ValueAwareBR),
        ("point_response", "certified_budget", PointResponse),
    ]
    return lineup


def _run_method_against(
    opponent: Opponent,
    builder,
    rounds: int,
    episodes_per_round: int,
    v_ref: float,
    scbr_value: float,
    seed: int,
) -> dict[str, Any]:
    """Run the method against experiment."""
    game = opponent.game
    payoff = build_payoff(game)
    sf0 = compile_game(game, 0)
    y_star = list(opponent.realization())
    evidence = OpponentEvidenceStore.for_game(game)
    meth = builder()

    actuals: list[float] = []
    min_safety = float("inf")
    total_payoff = 0.0
    for t in range(rounds):
        x = meth.select(t, evidence)
        realized = payoff.bilinear(x, y_star)
        actuals.append(realized)
        meth.observe(realized, v_ref)
        min_safety = min(min_safety, safety_verifier(x, game=game).value)
        behavior = sf0.behavior_from_realization(x)
        pay, counts = native.simulate(
            game, behavior, opponent.behavior, episodes_per_round, seed + t
        )
        total_payoff += pay
        for label, c in counts.items():
            evidence.record(label, c)

    tail = actuals[max(0, len(actuals) // 2) :]
    final_actual = mean(tail)
    return {
        "final_actual_value": final_actual,
        "exploitation_gain": final_actual - v_ref,
        "distance_from_scbr": scbr_value - final_actual,
        "min_safety_value": min_safety,
        "safety_violation": max(0.0, v_ref - min_safety),
        "avg_payoff": total_payoff / (rounds * episodes_per_round),
        **meth.extra(),
    }


@dataclass(frozen=True)
class _ComparisonCell:
    """Represent comparison cell for the online workflow."""

    opp_key: str
    opponent: Opponent
    method_name: str
    seed: int
    scbr_value: float
    game: str
    v_ref: float
    x_blueprint: tuple[float, ...]
    delta: float
    method: str
    eps_safe: float
    rnr_ps: tuple[float, ...]
    beta: float
    probe_per_round: float
    rounds: int
    episodes_per_round: int


def _run_comparison_cell(
    cell: _ComparisonCell,
) -> tuple[str, str, int, dict[str, Any]]:
    """Run the comparison cell experiment."""
    sf0 = compile_game(cell.game, 0)
    sf1 = compile_game(cell.game, 1)
    lineup = _baseline_methods(
        cell.game,
        cell.v_ref,
        cell.x_blueprint,
        cell.delta,
        cell.method,
        cell.eps_safe,
        cell.rnr_ps,
        cell.beta,
        cell.probe_per_round,
        cell.rounds,
        sf0,
        sf1,
    )
    factory = next(f for name, _g, f in lineup if name == cell.method_name)
    result = _run_method_against(
        cell.opponent,
        factory,
        cell.rounds,
        cell.episodes_per_round,
        cell.v_ref,
        cell.scbr_value,
        cell.seed,
    )
    return (cell.opp_key, cell.method_name, cell.seed, result)


def _parallel_map(fn, args, workers: int | None):
    """Compute parallel map for the online workflow."""
    args = list(args)
    if workers is None or workers <= 1 or len(args) <= 1:
        return [fn(a) for a in args]
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing as mp

    try:
        ctx = mp.get_context("forkserver")
    except ValueError:
        ctx = mp.get_context()
    with ProcessPoolExecutor(
        max_workers=min(workers, len(args)), mp_context=ctx
    ) as pool:
        return list(pool.map(fn, args))


def run_baseline_comparison(
    opponents: Mapping[str, Opponent] | None = None,
    rounds: int = 60,
    episodes_per_round: int = 200,
    delta: float = 0.05,
    eps_safe: float = 0.0,
    method: str = "hoeffding",
    rnr_ps: Sequence[float] = (0.5,),
    beta: float = 2.0,
    probe_per_round: float = 0.5,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    out_dir: str | Path | None = "results",
    workers: int | None = None,
) -> dict[str, Any]:
    """Run the baseline comparison experiment."""
    suite = dict(opponents) if opponents is not None else leduc_opponent_suite()
    game = next(iter(suite.values())).game
    blueprint = solve_blueprint(game, method="lp")
    v_ref = blueprint.value
    assert blueprint.realization is not None
    x_blueprint = blueprint.realization
    sf0 = compile_game(game, 0)
    sf1 = compile_game(game, 1)

    lineup = _baseline_methods(
        game,
        v_ref,
        x_blueprint,
        delta,
        method,
        eps_safe,
        rnr_ps,
        beta,
        probe_per_round,
        rounds,
        sf0,
        sf1,
    )
    method_order: list[tuple[str, str]] = [(name, g) for name, g, _ in lineup]

    scbr = {
        opp_name: safety_constrained_best_response(
            opponent.behavior, v_ref=v_ref, eps_safe=eps_safe, game=game
        ).value
        for opp_name, opponent in suite.items()
    }

    cells = [
        _ComparisonCell(
            opp_key=opp_name,
            opponent=opponent,
            method_name=name,
            seed=seed,
            scbr_value=scbr[opp_name],
            game=game,
            v_ref=v_ref,
            x_blueprint=tuple(x_blueprint),
            delta=delta,
            method=method,
            eps_safe=eps_safe,
            rnr_ps=tuple(rnr_ps),
            beta=beta,
            probe_per_round=probe_per_round,
            rounds=rounds,
            episodes_per_round=episodes_per_round,
        )
        for opp_name, opponent in suite.items()
        for name, _g in method_order
        for seed in seeds
    ]

    flat = _parallel_map(_run_comparison_cell, cells, workers)
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for opp_key, method_name, _seed, result in flat:
        grouped.setdefault((opp_key, method_name), []).append(result)

    rows: dict[str, Any] = {}
    method_names: list[str] = [name for name, _g in method_order]
    for opp_name in suite:
        scbr_value = scbr[opp_name]
        per_method: dict[str, Any] = {}
        for name, guarantee in method_order:
            runs = grouped[(opp_name, name)]
            violation_max = max(r["safety_violation"] for r in runs)

            if guarantee == "hard_safe":
                guarantee_held = violation_max <= 1e-8
            elif guarantee == "certified_budget":
                guarantee_held = violation_max <= probe_per_round + 1e-8
            elif guarantee == "gift_funded":
                guarantee_held = all(
                    r.get("gift_min_bankroll", 0.0) >= -_GIFT_GUARANTEE_TOL
                    for r in runs
                )
            else:
                guarantee_held = True
            per_method[name] = {
                "guarantee": guarantee,
                "guarantee_held": guarantee_held,
                "exploitation_gain_mean": mean(r["exploitation_gain"] for r in runs),
                "exploitation_gain_std": (
                    pstdev([r["exploitation_gain"] for r in runs])
                    if len(runs) > 1
                    else 0.0
                ),
                "final_actual_value_mean": mean(r["final_actual_value"] for r in runs),
                "min_safety_value": min(r["min_safety_value"] for r in runs),
                "safety_violation_max": violation_max,
                "distance_from_scbr_mean": mean(r["distance_from_scbr"] for r in runs),
                "avg_payoff_mean": mean(r["avg_payoff"] for r in runs),
            }

            extra_keys = {
                k
                for r in runs
                for k, v in r.items()
                if k
                not in {
                    "exploitation_gain",
                    "final_actual_value",
                    "distance_from_scbr",
                    "min_safety_value",
                    "safety_violation",
                    "avg_payoff",
                }
                and isinstance(v, (int, float))
            }
            for k in extra_keys:
                vals = [r[k] for r in runs if k in r]
                if vals:
                    per_method[name][f"{k}_mean"] = mean(vals)
        rows[opp_name] = {
            "game_value": v_ref,
            "scbr_value": scbr_value,
            "scbr_gain": scbr_value - v_ref,
            "methods": per_method,
        }

    results: dict[str, Any] = {
        "game": game,
        "seeds": list(seeds),
        "rounds": rounds,
        "episodes_per_round": episodes_per_round,
        "delta": delta,
        "eps_safe": eps_safe,
        "method": method,
        "rnr_ps": list(rnr_ps),
        "method_names": method_names,
        "opponents": rows,
    }
    if out_dir is not None:
        evaluation.save_results(
            results, Path(out_dir) / f"baseline_comparison_{game}.json"
        )
    return results


_SHOWDOWN_METHODS: tuple[str, ...] = (
    "blueprint",
    "passive_public",
    "rnr",
    "gift_based",
    "public_point",
    "censored_em",
    "public_robust",
    "point_response",
    "confidence_guarded",
    "safe_active_decensoring",
)
_SHOWDOWN_GUARANTEE: dict[str, str] = {
    "blueprint": "hard_safe",
    "passive_public": "hard_safe",
    "rnr": "none",
    "gift_based": "gift_funded",
    "public_point": "certified_budget",
    "censored_em": "certified_budget",
    "public_robust": "certified_budget",
    "point_response": "certified_budget",
    "confidence_guarded": "certified_budget",
    "safe_active_decensoring": "certified_budget",
}


_T_CRIT_975: dict[int, float] = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    9: 2.262,
    14: 2.145,
    24: 2.064,
}


def _ci95_halfwidth(values: Sequence[float]) -> float:
    """Compute ci95 halfwidth for the online workflow."""
    n = len(values)
    if n <= 1:
        return 0.0
    se = stdev(values) / sqrt(n)
    return _T_CRIT_975.get(n - 1, 1.96) * se


def _public_floor_shadow(
    groups: Mapping[str, Sequence[str]],
    intervals: Mapping[str, Sequence[tuple[float, float]]],
    v_ref: float,
    eps_safe: float,
    game: str,
    delta_rho: float = 0.05,
    weights: Mapping[str, float] | None = None,
) -> float:
    """Compute public floor shadow for the online workflow."""
    j0 = robust_safe_response_public(
        groups, intervals, v_ref=v_ref, eps_safe=eps_safe, game=game, weights=weights
    ).robust_value
    jd = robust_safe_response_public(
        groups,
        intervals,
        v_ref=v_ref,
        eps_safe=eps_safe + delta_rho,
        game=game,
        weights=weights,
    ).robust_value
    return (jd - j0) / delta_rho


@cache
def _p2_fold_indices(game: str) -> tuple[tuple[str, int], ...]:
    """Compute player-two fold indices for the online workflow."""
    sf1 = compile_game(game, 1)
    out: list[tuple[str, int]] = []
    for info in sf1.info_sets:
        for i, (char, _child) in enumerate(info.children):
            if char == "f":
                out.append((info.label, i))
                break
    return tuple(out)


def _game_equilibrium_opponent(game: str) -> Opponent:
    """Construct the game equilibrium opponent policy."""
    if game == "holdem" or game.startswith("holdem_"):
        return holdem_equilibrium_opponent(game)
    if game == "leduc":
        return leduc_equilibrium_opponent()
    if game == "kuhn":
        return equilibrium_opponent()
    raise ValueError(f"no equilibrium opponent registered for game {game!r}")


@cache
def _cf_fold_data(
    game: str,
) -> tuple[tuple[float, ...], tuple[tuple[str, tuple[tuple[str, int, int], ...]], ...]]:
    """Compute cf fold data for the online workflow."""
    sf1 = compile_game(game, 1)
    y_eq = tuple(_game_equilibrium_opponent(game).realization())
    fold_idx = dict(_p2_fold_indices(game))
    by_key: dict[str, list[tuple[str, int, int]]] = {}
    for info in sf1.info_sets:
        fi = fold_idx.get(info.label)
        if fi is None or fi >= len(info.children):
            continue
        key = public_key(game, info.label)
        by_key.setdefault(key, []).append(
            (info.label, info.parent_seq, info.children[fi][1])
        )
    groups = tuple((k, tuple(v)) for k, v in by_key.items())
    return y_eq, groups


def _counterfactual_fold_reference(
    game: str, weights: Mapping[str, float]
) -> dict[str, float]:
    """Compute counterfactual fold reference for the online workflow."""
    y_eq, groups = _cf_fold_data(game)
    ref: dict[str, float] = {}
    for key, members in groups:
        num = sum(weights.get(lab, 0.0) * y_eq[fseq] for lab, _p, fseq in members)
        den = sum(weights.get(lab, 0.0) * y_eq[pseq] for lab, pseq, _f in members)
        ref[key] = num / den if den > 1e-12 else 0.0
    return ref


def _adaptive_fold_gate(
    game: str,
    ev_public: OpponentEvidenceStore,
    ref_fold: Mapping[str, float],
) -> float:
    """Compute adaptive fold gate for the online workflow."""
    counts = ev_public._public_counts()
    groups = ev_public.public_groups()
    fold_idx = dict(_p2_fold_indices(game))
    total = sum(float(sum(c)) for c in counts.values())
    if total <= 0.0:
        return 1.0
    gate = 0.0
    for key, c in counts.items():
        member = groups[key][0]
        fi = fold_idx.get(member)
        if fi is None or fi >= len(c):
            continue
        mass = float(sum(c))
        if mass <= 0.0:
            continue
        fold_rate = c[fi] / mass
        reach = mass / total
        gate += (fold_rate - ref_fold.get(key, 0.0)) * reach
    return gate


def _public_point_behavior(
    ev_public: OpponentEvidenceStore,
) -> dict[str, list[float]]:
    """Compute public point behavior for the online workflow."""
    counts = ev_public._public_counts()
    groups = ev_public.public_groups()
    behavior: dict[str, list[float]] = {}
    for key, c in counts.items():
        total = float(sum(c))
        if total > 0.0:
            dist = [ci / total for ci in c]
        else:
            k = len(c)
            dist = [1.0 / k] * k
        for label in groups[key]:
            behavior[label] = list(dist)
    return behavior


def _censored_em_behavior(
    game: str,
    ev_point: OpponentEvidenceStore,
    ev_public: OpponentEvidenceStore,
    pub_weights: Mapping[str, float] | None,
    iters: int = 20,
) -> dict[str, list[float]]:
    """Compute censored em behavior for the online workflow."""
    weights = pub_weights or {}
    fold_idx = dict(_p2_fold_indices(game))
    groups = ev_public.public_groups()
    public_counts = ev_public._public_counts()
    point_public = ev_point._public_counts()

    behavior: dict[str, list[float]] = {}
    for key, members in groups.items():
        fi = next(
            (fold_idx.get(m) for m in members if fold_idx.get(m) is not None), None
        )
        if fi is None:
            for m in members:
                behavior[m] = list(ev_point.p_hat(m))
            continue
        pub = public_counts.get(key, [])
        pt = point_public.get(key, [0] * len(pub))

        fold_total = max(0.0, float(pub[fi]) - float(pt[fi])) if fi < len(pub) else 0.0
        s = {m: list(ev_point.counts(m)) for m in members}
        sd_total = {m: float(sum(s[m])) for m in members}
        denom0 = fold_total + sum(sd_total.values())
        init = fold_total / denom0 if denom0 > 1e-12 else 0.0
        pfold = {m: init for m in members}
        if fold_total > 0.0:
            for _ in range(iters):
                wnum = {m: weights.get(m, 1.0) * pfold[m] for m in members}
                z = sum(wnum.values())
                if z <= 1e-12:
                    imputed = {m: fold_total / len(members) for m in members}
                else:
                    imputed = {m: fold_total * wnum[m] / z for m in members}
                for m in members:
                    n = sd_total[m] + imputed[m]
                    pfold[m] = imputed[m] / n if n > 1e-12 else 0.0
        for m in members:
            row = s[m]
            k = len(row)
            mfi = fold_idx.get(m, fi)
            pf = pfold[m]
            sd = sd_total[m]
            dist = [0.0] * k
            if sd > 1e-12:
                for a in range(k):
                    if a != mfi:
                        dist[a] = (row[a] / sd) * (1.0 - pf)
            else:
                nonfold = [a for a in range(k) if a != mfi]
                for a in nonfold:
                    dist[a] = (1.0 - pf) / max(1, len(nonfold))
            if 0 <= mfi < k:
                dist[mfi] = pf
            tot = sum(dist)
            behavior[m] = [d / tot for d in dist] if tot > 1e-12 else [1.0 / k] * k
    for label in ev_point.labels:
        behavior.setdefault(label, list(ev_point.p_hat(label)))
    return behavior


def _showdown_plan(
    method_name: str,
    ev_point: OpponentEvidenceStore,
    ev_public: OpponentEvidenceStore,
    budget: SafetyBudgetLedger,
    game: str,
    v_ref: float,
    x_blueprint: tuple[float, ...],
    sf1,
    payoff,
    delta: float,
    method: str,
    eps_safe: float,
    kappa: float,
    rnr_p: float,
    cum_above_floor: float,
    pub_weights: Mapping[str, float] | None = None,
    route_counter: dict[str, int] | None = None,
    latch_threshold: float | None = None,
) -> tuple[list[float], float, float]:
    """Compute showdown plan for the online workflow."""
    if method_name == "blueprint":
        return list(x_blueprint), 0.0, 0.0
    groups = ev_public.public_groups()
    pub_intervals = ev_public.public_intervals(delta, method=method)
    if method_name == "passive_public":
        resp = robust_safe_response_public(
            groups,
            pub_intervals,
            v_ref=v_ref,
            eps_safe=eps_safe,
            game=game,
            weights=pub_weights,
        )
        return list(resp.realization), 0.0, resp.robust_value - v_ref

    if method_name == "rnr":
        yhat_real = _empirical_realization(ev_point, sf1)
        resp = restricted_nash_response(yhat_real, rnr_p, game=game)
        x = list(resp.realization)
        return x, 0.0, payoff.bilinear(x, yhat_real) - v_ref

    if method_name == "gift_based":
        if cum_above_floor <= 1e-12:
            return list(x_blueprint), 0.0, 0.0
        yhat_real = _empirical_realization(ev_point, sf1)
        x_br = best_response(yhat_real, game=game).realization
        x = list(
            fallback_mixture_repair(
                x_blueprint, x_br, v_ref, eps_safe=cum_above_floor, game=game
            )
        )
        return x, 0.0, payoff.bilinear(x, yhat_real) - v_ref

    shadow = _public_floor_shadow(
        groups, pub_intervals, v_ref, eps_safe, game, weights=pub_weights
    )
    rho = budget.allowance(shadow)
    yhat_behavior = {label: list(ev_point.p_hat(label)) for label in ev_point.labels}
    if method_name == "public_robust":
        resp = robust_safe_response_public(
            groups,
            pub_intervals,
            v_ref=v_ref,
            eps_safe=eps_safe + rho,
            game=game,
            weights=pub_weights,
        )
        x = list(resp.realization)
        believe = resp.robust_value - v_ref
    elif method_name == "public_point":
        resp = safety_constrained_best_response(
            _public_point_behavior(ev_public),
            v_ref=v_ref,
            eps_safe=eps_safe + rho,
            game=game,
        )
        x = list(resp.realization)
        believe = resp.value - v_ref
    elif method_name == "censored_em":
        resp = safety_constrained_best_response(
            _censored_em_behavior(game, ev_point, ev_public, pub_weights),
            v_ref=v_ref,
            eps_safe=eps_safe + rho,
            game=game,
        )
        x = list(resp.realization)
        believe = resp.value - v_ref
    elif method_name == "point_response":
        resp = safety_constrained_best_response(
            yhat_behavior, v_ref=v_ref, eps_safe=eps_safe + rho, game=game
        )
        x = list(resp.realization)
        believe = resp.value - v_ref
    elif method_name == "confidence_guarded":
        yhat_real = list(sf1.realization_from_behavior(yhat_behavior))
        resp = confidence_guarded_point_probe(
            groups,
            pub_intervals,
            yhat_real,
            v_ref=v_ref,
            eps_safe=eps_safe,
            rho=rho,
            kappa=kappa,
            game=game,
            weights=pub_weights,
        )
        x = list(resp.realization)
        believe = resp.point_value - v_ref
    elif method_name == "safe_active_decensoring":
        ref_cf = _counterfactual_fold_reference(game, pub_weights or {})
        gate = _adaptive_fold_gate(game, ev_public, ref_cf)

        if (
            latch_threshold is not None
            and route_counter is not None
            and budget.debt > latch_threshold
        ):
            route_counter["latched"] = 1
        latched = route_counter is not None and route_counter.get("latched", 0) == 1
        if latched or gate > 0.0:
            if route_counter is not None:
                route_counter["core"] += 1
            resp = robust_safe_response_public(
                groups,
                pub_intervals,
                v_ref=v_ref,
                eps_safe=eps_safe + rho,
                game=game,
                weights=pub_weights,
            )
            x = list(resp.realization)
            believe = resp.robust_value - v_ref
        else:
            if route_counter is not None:
                route_counter["point"] += 1
            resp = safety_constrained_best_response(
                yhat_behavior, v_ref=v_ref, eps_safe=eps_safe + rho, game=game
            )
            x = list(resp.realization)
            believe = resp.value - v_ref
    else:
        raise ValueError(f"unknown showdown method {method_name!r}")
    spent = max(0.0, (v_ref - eps_safe) - safety_verifier(x, game=game).value)
    budget.charge(spent)
    return x, spent, believe


def _run_showdown_method(
    method_name: str,
    opponent: Opponent,
    rounds: int,
    episodes_per_round: int,
    v_ref: float,
    x_blueprint: tuple[float, ...],
    delta: float,
    method: str,
    eps_safe: float,
    rho_cap: float,
    kappa: float,
    safety_debt_max: float,
    rnr_p: float,
    scbr_value: float,
    seed: int,
    latch_threshold: float | None = None,
) -> dict[str, Any]:
    """Run the showdown method experiment."""
    game = opponent.game
    payoff = build_payoff(game)
    sf0 = compile_game(game, 0)
    sf1 = compile_game(game, 1)
    ev_point = OpponentEvidenceStore.for_game(game)
    ev_public = OpponentEvidenceStore.for_game(game)
    budget = SafetyBudgetLedger(
        rho_cap=rho_cap, hard_total=rho_cap * rounds, debt_max=safety_debt_max
    )
    y_star = list(opponent.realization())

    uses_public = method_name in {
        "passive_public",
        "public_point",
        "censored_em",
        "public_robust",
        "point_response",
        "confidence_guarded",
        "safe_active_decensoring",
    }
    reach_accum: dict[str, float] = {}

    route_counter: dict[str, int] = {"core": 0, "point": 0, "latched": 0}

    realized: list[float] = []
    believes: list[float] = []
    min_safety = float("inf")
    total_payoff = 0.0
    spend_rounds = 0
    cum_above_floor = 0.0
    min_cum_above_floor = 0.0
    for t in range(rounds):
        x, spent, believe = _showdown_plan(
            method_name,
            ev_point,
            ev_public,
            budget,
            game,
            v_ref,
            x_blueprint,
            sf1,
            payoff,
            delta,
            method,
            eps_safe,
            kappa,
            rnr_p,
            cum_above_floor,
            reach_accum if uses_public else None,
            route_counter if method_name == "safe_active_decensoring" else None,
            latch_threshold if method_name == "safe_active_decensoring" else None,
        )
        r = payoff.bilinear(x, y_star)
        realized.append(r)
        believes.append(believe)
        if spent > 1e-12:
            spend_rounds += 1
        min_safety = min(min_safety, safety_verifier(x, game=game).value)
        budget.settle(spent, r - v_ref)
        cum_above_floor += r - v_ref
        min_cum_above_floor = min(min_cum_above_floor, cum_above_floor)
        behavior = sf0.behavior_from_realization(x)

        if uses_public:
            for label, w in opponent_reach_weights(x, game=game).items():
                reach_accum[label] = reach_accum.get(label, 0.0) + w
        pay, show, fold = native.simulate_showdown(
            game, behavior, opponent.behavior, episodes_per_round, seed + t
        )
        total_payoff += pay
        for label, c in show.items():
            ev_point.record(label, c)
            ev_public.record(label, c)
        for label, c in fold.items():
            ev_public.record(label, c)

    tail = realized[max(0, len(realized) // 2) :]
    final_actual = mean(tail)
    belief_tail = believes[max(0, len(believes) // 2) :]
    believe_mean = mean(belief_tail) if belief_tail else 0.0
    return {
        "final_actual_value": final_actual,
        "exploitation_gain": final_actual - v_ref,
        "distance_from_scbr": scbr_value - final_actual,
        "min_safety_value": min_safety,
        "safety_violation": max(0.0, v_ref - min_safety),
        "avg_payoff": total_payoff / (rounds * episodes_per_round),
        "budget_spent": budget.spent,
        "safety_debt_final": budget.debt,
        "safety_spend_rate": spend_rounds / max(1, rounds),
        "core_selected_rate": (
            route_counter["core"] / (route_counter["core"] + route_counter["point"])
            if (route_counter["core"] + route_counter["point"]) > 0
            else 0.0
        ),
        "latched": route_counter["latched"],
        "point_belief_gain": believe_mean,
        "phantom_gap": believe_mean - (final_actual - v_ref),
        "gift_bankroll_final": cum_above_floor,
        "gift_min_bankroll": min_cum_above_floor,
    }


@dataclass(frozen=True)
class _ShowdownCell:
    """Represent showdown cell for the online workflow."""

    opp_key: str
    opponent: Opponent
    method_name: str
    seed: int
    scbr_value: float
    rounds: int
    episodes_per_round: int
    v_ref: float
    x_blueprint: tuple[float, ...]
    delta: float
    method: str
    eps_safe: float
    rho_cap: float
    kappa: float
    safety_debt_max: float
    rnr_p: float
    latch_threshold: float | None = None


def _run_showdown_cell(
    cell: _ShowdownCell,
) -> tuple[str, str, int, dict[str, Any]]:
    """Run the showdown cell experiment."""
    _t0 = perf_counter()
    result = _run_showdown_method(
        cell.method_name,
        cell.opponent,
        cell.rounds,
        cell.episodes_per_round,
        cell.v_ref,
        cell.x_blueprint,
        cell.delta,
        cell.method,
        cell.eps_safe,
        cell.rho_cap,
        cell.kappa,
        cell.safety_debt_max,
        cell.rnr_p,
        cell.scbr_value,
        cell.seed,
        cell.latch_threshold,
    )
    result["wall_time"] = perf_counter() - _t0
    return (cell.opp_key, cell.method_name, cell.seed, result)


def run_showdown_comparison(
    opponents: Mapping[str, Opponent] | None = None,
    rounds: int = 60,
    episodes_per_round: int = 200,
    delta: float = 0.1,
    eps_safe: float = 0.0,
    method: str = "hoeffding",
    rho_cap: float = 0.5,
    kappa: float = 0.2,
    safety_debt_max: float = 3.0,
    rnr_p: float = 0.5,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    out_dir: str | Path | None = "results",
    workers: int | None = None,
    latch_threshold: float | None = None,
) -> dict[str, Any]:
    """Run the showdown comparison experiment."""
    suite = dict(opponents) if opponents is not None else leduc_opponent_suite()
    game = next(iter(suite.values())).game
    blueprint = solve_blueprint(game, method="lp")
    v_ref = blueprint.value
    assert blueprint.realization is not None
    x_blueprint = blueprint.realization
    scbr = {
        opp_name: safety_constrained_best_response(
            opponent.behavior, v_ref=v_ref, eps_safe=eps_safe, game=game
        ).value
        for opp_name, opponent in suite.items()
    }

    cells = [
        _ShowdownCell(
            opp_key=opp_name,
            opponent=opponent,
            method_name=name,
            seed=seed,
            scbr_value=scbr[opp_name],
            rounds=rounds,
            episodes_per_round=episodes_per_round,
            v_ref=v_ref,
            x_blueprint=tuple(x_blueprint),
            delta=delta,
            method=method,
            eps_safe=eps_safe,
            rho_cap=rho_cap,
            kappa=kappa,
            safety_debt_max=safety_debt_max,
            rnr_p=rnr_p,
            latch_threshold=latch_threshold,
        )
        for opp_name, opponent in suite.items()
        for name in _SHOWDOWN_METHODS
        for seed in seeds
    ]

    flat = _parallel_map(_run_showdown_cell, cells, workers)
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for opp_key, method_name, _seed, result in flat:
        grouped.setdefault((opp_key, method_name), []).append(result)

    timing_by_method: dict[str, float] = {name: 0.0 for name in _SHOWDOWN_METHODS}
    for (_opp, method_name), method_runs in grouped.items():
        timing_by_method[method_name] += sum(r["wall_time"] for r in method_runs)

    rows: dict[str, Any] = {}
    for opp_name in suite:
        per_method: dict[str, Any] = {}
        for name in _SHOWDOWN_METHODS:
            guarantee = _SHOWDOWN_GUARANTEE[name]
            runs = grouped[(opp_name, name)]
            violation_max = max(r["safety_violation"] for r in runs)
            if guarantee == "hard_safe":
                guarantee_held = violation_max <= _SAFETY_GUARANTEE_TOL
            elif guarantee == "certified_budget":
                guarantee_held = violation_max <= rho_cap + _SAFETY_GUARANTEE_TOL
            elif guarantee == "gift_funded":
                guarantee_held = all(
                    r["gift_min_bankroll"] >= -_GIFT_GUARANTEE_TOL for r in runs
                )
            else:
                guarantee_held = True
            per_method[name] = {
                "guarantee": guarantee,
                "guarantee_held": guarantee_held,
                "exploitation_gain_mean": mean(r["exploitation_gain"] for r in runs),
                "exploitation_gain_std": (
                    pstdev([r["exploitation_gain"] for r in runs])
                    if len(runs) > 1
                    else 0.0
                ),
                "exploitation_gain_ci95": _ci95_halfwidth(
                    [r["exploitation_gain"] for r in runs]
                ),
                "final_actual_value_mean": mean(r["final_actual_value"] for r in runs),
                "min_safety_value": min(r["min_safety_value"] for r in runs),
                "safety_violation_max": violation_max,
                "distance_from_scbr_mean": mean(r["distance_from_scbr"] for r in runs),
                "avg_payoff_mean": mean(r["avg_payoff"] for r in runs),
                "wall_time_mean": mean(r["wall_time"] for r in runs),
                "wall_time_total": sum(r["wall_time"] for r in runs),
            }
            for key in (
                "budget_spent",
                "safety_debt_final",
                "safety_spend_rate",
                "core_selected_rate",
                "point_belief_gain",
                "phantom_gap",
                "gift_bankroll_final",
                "gift_min_bankroll",
            ):
                per_method[name][f"{key}_mean"] = mean(r[key] for r in runs)

        oracle_gain = max(
            per_method["public_robust"]["exploitation_gain_mean"],
            per_method["point_response"]["exploitation_gain_mean"],
        )
        rows[opp_name] = {
            "game_value": v_ref,
            "scbr_value": scbr[opp_name],
            "scbr_gain": scbr[opp_name] - v_ref,
            "oracle_gain": oracle_gain,
            "oracle_gap": oracle_gain
            - per_method["safe_active_decensoring"]["exploitation_gain_mean"],
            "methods": per_method,
        }

    results: dict[str, Any] = {
        "game": game,
        "monitoring": "showdown",
        "seeds": list(seeds),
        "rounds": rounds,
        "episodes_per_round": episodes_per_round,
        "delta": delta,
        "eps_safe": eps_safe,
        "method": method,
        "kappa": kappa,
        "rho_cap": rho_cap,
        "rnr_p": rnr_p,
        "latch_threshold": latch_threshold,
        "method_names": list(_SHOWDOWN_METHODS),
        "wall_time_by_method": timing_by_method,
        "opponents": rows,
    }
    if out_dir is not None:
        evaluation.save_results(
            results, Path(out_dir) / f"showdown_comparison_{game}.json"
        )
    return results


def _run_method_vs_adversary(
    factory,
    game: str,
    rounds: int,
    episodes_per_round: int,
    v_ref: float,
    seed: int,
) -> dict[str, Any]:
    """Run the method vs adversary experiment."""
    sf0 = compile_game(game, 0)
    sf1 = compile_game(game, 1)
    evidence = OpponentEvidenceStore.for_game(game)
    meth = factory()

    realized: list[float] = []
    for t in range(rounds):
        x = meth.select(t, evidence)

        safety = safety_verifier(x, game=game)
        realized.append(safety.value)
        meth.observe(safety.value, v_ref)
        adversary_behavior = sf1.behavior_from_realization(safety.best_response)
        agent_behavior = sf0.behavior_from_realization(x)
        _, counts = native.simulate(
            game, agent_behavior, adversary_behavior, episodes_per_round, seed + t
        )
        for label, c in counts.items():
            evidence.record(label, c)

    tail = realized[max(0, len(realized) // 2) :]
    return {
        "realized_value_mean": mean(tail),
        "realized_value_worst": min(realized),
    }


def run_adversarial_stress(
    rounds: int = 60,
    episodes_per_round: int = 200,
    delta: float = 0.05,
    eps_safe: float = 0.0,
    method: str = "hoeffding",
    rnr_ps: Sequence[float] = (0.5,),
    beta: float = 2.0,
    probe_per_round: float = 0.5,
    game: str = "leduc",
    seeds: Sequence[int] = DEFAULT_SEEDS,
    out_dir: str | Path | None = "results",
) -> dict[str, Any]:
    """Run the adversarial stress experiment."""
    blueprint = solve_blueprint(game, method="lp")
    v_ref = blueprint.value
    assert blueprint.realization is not None
    x_blueprint = blueprint.realization
    sf0 = compile_game(game, 0)
    sf1 = compile_game(game, 1)
    floor = v_ref - eps_safe

    lineup = _baseline_methods(
        game,
        v_ref,
        x_blueprint,
        delta,
        method,
        eps_safe,
        rnr_ps,
        beta,
        probe_per_round,
        rounds,
        sf0,
        sf1,
    )
    per_method: dict[str, Any] = {}
    for name, guarantee, factory in lineup:
        runs = [
            _run_method_vs_adversary(
                factory, game, rounds, episodes_per_round, v_ref, seed
            )
            for seed in seeds
        ]
        worst = min(r["realized_value_worst"] for r in runs)
        mean_realized = mean(r["realized_value_mean"] for r in runs)

        realized_loss = max(0.0, floor - worst)
        if guarantee == "hard_safe":
            guarantee_held = worst >= floor - 1e-8
        elif guarantee == "certified_budget":
            guarantee_held = worst >= floor - probe_per_round - 1e-8
        elif guarantee == "gift_funded":
            guarantee_held = worst >= floor - _GIFT_GUARANTEE_TOL
        else:
            guarantee_held = True
        per_method[name] = {
            "guarantee": guarantee,
            "guarantee_held": guarantee_held,
            "realized_value_mean": mean_realized,
            "realized_value_worst": worst,
            "realized_loss_below_floor": realized_loss,
        }

    results: dict[str, Any] = {
        "game": game,
        "seeds": list(seeds),
        "rounds": rounds,
        "episodes_per_round": episodes_per_round,
        "delta": delta,
        "eps_safe": eps_safe,
        "method": method,
        "rnr_ps": list(rnr_ps),
        "game_value": v_ref,
        "safety_floor": floor,
        "method_names": list(per_method),
        "methods": per_method,
    }
    if out_dir is not None:
        evaluation.save_results(
            results, Path(out_dir) / f"adversarial_stress_{game}.json"
        )
    return results


def _strike_behavior(x: Sequence[float], sf1, game: str) -> dict[str, list[float]]:
    """Compute strike behavior for the online workflow."""
    best_response_realization = safety_verifier(x, game=game).best_response
    return sf1.behavior_from_realization(best_response_realization)


def _lure_then_strike_schedule(
    lure_behavior: Mapping[str, Sequence[float]], strike_round: int, sf1, game: str
):
    """Compute lure then strike schedule."""
    lure = {label: list(dist) for label, dist in lure_behavior.items()}

    def schedule(t: int, x: Sequence[float], rounds: int) -> dict[str, list[float]]:
        """Construct the round-by-round probing schedule."""
        if t < strike_round:
            return lure
        return _strike_behavior(x, sf1, game)

    return schedule


def _drift_schedule(
    lure_behavior: Mapping[str, Sequence[float]], sf1, game: str, power: float = 1.0
):
    """Compute drift schedule for the online workflow."""
    lure = {label: list(dist) for label, dist in lure_behavior.items()}

    def schedule(t: int, x: Sequence[float], rounds: int) -> dict[str, list[float]]:
        """Construct the round-by-round probing schedule."""
        w = (t / (rounds - 1)) ** power if rounds > 1 else 1.0
        strike = _strike_behavior(x, sf1, game)
        return {
            label: [
                (1.0 - w) * p + w * q
                for p, q in zip(lure[label], strike[label], strict=True)
            ]
            for label in lure
        }

    return schedule


def _run_method_vs_schedule(
    factory,
    game: str,
    schedule,
    rounds: int,
    episodes_per_round: int,
    v_ref: float,
    seed: int,
) -> dict[str, Any]:
    """Run the method vs schedule experiment."""
    payoff = build_payoff(game)
    sf0 = compile_game(game, 0)
    sf1 = compile_game(game, 1)
    evidence = OpponentEvidenceStore.for_game(game)
    meth = factory()

    realized: list[float] = []
    running = 0.0
    aggregate_min = 0.0
    aggregate_max = 0.0
    for t in range(rounds):
        x = meth.select(t, evidence)
        opp_behavior = schedule(t, x, rounds)
        y_t = list(sf1.realization_from_behavior(opp_behavior))
        rv = payoff.bilinear(x, y_t)
        realized.append(rv)
        meth.observe(rv, v_ref)
        running += rv - v_ref
        aggregate_min = min(aggregate_min, running)
        aggregate_max = max(aggregate_max, running)
        agent_behavior = sf0.behavior_from_realization(x)
        _, counts = native.simulate(
            game, agent_behavior, opp_behavior, episodes_per_round, seed + t
        )
        for label, c in counts.items():
            evidence.record(label, c)

    return {
        "realized_value_mean": mean(realized),
        "realized_value_worst": min(realized),
        "cumulative_above_floor": running,
        "peak_above_floor": aggregate_max,
        "aggregate_min": aggregate_min,
    }


def run_nonstationary_stress(
    kind: str = "lure_then_strike",
    lure: str = "static_biased",
    strike_round: int | None = None,
    drift_power: float = 1.0,
    rounds: int = 60,
    episodes_per_round: int = 200,
    delta: float = 0.05,
    eps_safe: float = 0.0,
    method: str = "hoeffding",
    rnr_ps: Sequence[float] = (0.5,),
    beta: float = 2.0,
    probe_per_round: float = 0.5,
    game: str = "leduc",
    seeds: Sequence[int] = DEFAULT_SEEDS,
    out_dir: str | Path | None = "results",
) -> dict[str, Any]:
    """Run the nonstationary stress experiment."""
    if strike_round is None:
        strike_round = rounds // 2
    blueprint = solve_blueprint(game, method="lp")
    v_ref = blueprint.value
    assert blueprint.realization is not None
    x_blueprint = blueprint.realization
    sf0 = compile_game(game, 0)
    sf1 = compile_game(game, 1)
    floor = v_ref - eps_safe
    lure_behavior = opponent_from_spec({"game": game, "type": lure}).behavior

    if kind == "lure_then_strike":
        schedule = _lure_then_strike_schedule(lure_behavior, strike_round, sf1, game)
    elif kind == "drift":
        schedule = _drift_schedule(lure_behavior, sf1, game, power=drift_power)
    else:
        raise ValueError(
            f"unknown schedule kind {kind!r}; expected 'lure_then_strike' or 'drift'"
        )

    lineup = _baseline_methods(
        game,
        v_ref,
        x_blueprint,
        delta,
        method,
        eps_safe,
        rnr_ps,
        beta,
        probe_per_round,
        rounds,
        sf0,
        sf1,
    )
    per_method: dict[str, Any] = {}
    for name, guarantee, factory in lineup:
        runs = [
            _run_method_vs_schedule(
                factory, game, schedule, rounds, episodes_per_round, v_ref, seed
            )
            for seed in seeds
        ]
        worst = min(r["realized_value_worst"] for r in runs)
        mean_realized = mean(r["realized_value_mean"] for r in runs)
        cumulative = mean(r["cumulative_above_floor"] for r in runs)
        peak = mean(r["peak_above_floor"] for r in runs)
        aggregate_min = min(r["aggregate_min"] for r in runs)
        realized_loss = max(0.0, floor - worst)
        if guarantee == "hard_safe":
            guarantee_held = worst >= floor - 1e-8
        elif guarantee == "certified_budget":
            guarantee_held = worst >= floor - probe_per_round - 1e-8
        elif guarantee == "gift_funded":
            guarantee_held = aggregate_min >= -_GIFT_GUARANTEE_TOL
        else:
            guarantee_held = True
        per_method[name] = {
            "guarantee": guarantee,
            "guarantee_held": guarantee_held,
            "realized_value_mean": mean_realized,
            "realized_value_worst": worst,
            "cumulative_above_floor": cumulative,
            "peak_above_floor": peak,
            "aggregate_min": aggregate_min,
            "realized_loss_below_floor": realized_loss,
        }

    results: dict[str, Any] = {
        "game": game,
        "kind": kind,
        "lure": lure,
        "strike_round": strike_round,
        "drift_power": drift_power,
        "seeds": list(seeds),
        "rounds": rounds,
        "episodes_per_round": episodes_per_round,
        "delta": delta,
        "eps_safe": eps_safe,
        "method": method,
        "rnr_ps": list(rnr_ps),
        "game_value": v_ref,
        "safety_floor": floor,
        "method_names": list(per_method),
        "methods": per_method,
    }
    if out_dir is not None:
        evaluation.save_results(
            results, Path(out_dir) / f"nonstationary_{kind}_{game}.json"
        )
    return results


def _run_showdown_method_vs_schedule(
    method_name: str,
    game: str,
    schedule,
    rounds: int,
    episodes_per_round: int,
    v_ref: float,
    x_blueprint: tuple[float, ...],
    delta: float,
    method: str,
    eps_safe: float,
    rho_cap: float,
    kappa: float,
    safety_debt_max: float,
    rnr_p: float,
    seed: int,
    latch_threshold: float | None,
) -> dict[str, Any]:
    """Run the showdown method vs schedule experiment."""
    payoff = build_payoff(game)
    sf0 = compile_game(game, 0)
    sf1 = compile_game(game, 1)
    ev_point = OpponentEvidenceStore.for_game(game)
    ev_public = OpponentEvidenceStore.for_game(game)
    budget = SafetyBudgetLedger(
        rho_cap=rho_cap, hard_total=rho_cap * rounds, debt_max=safety_debt_max
    )
    uses_public = method_name in {
        "passive_public",
        "public_point",
        "censored_em",
        "public_robust",
        "point_response",
        "confidence_guarded",
        "safe_active_decensoring",
    }
    reach_accum: dict[str, float] = {}
    route_counter = {"core": 0, "point": 0, "latched": 0}
    realized: list[float] = []
    min_safety = float("inf")
    running = 0.0
    aggregate_min = 0.0
    cum_above_floor = 0.0
    for t in range(rounds):
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
            delta,
            method,
            eps_safe,
            kappa,
            rnr_p,
            cum_above_floor,
            reach_accum if uses_public else None,
            route_counter if method_name == "safe_active_decensoring" else None,
            latch_threshold if method_name == "safe_active_decensoring" else None,
        )
        opp_behavior = schedule(t, x, rounds)
        y_t = list(sf1.realization_from_behavior(opp_behavior))
        rv = payoff.bilinear(x, y_t)
        realized.append(rv)
        min_safety = min(min_safety, safety_verifier(x, game=game).value)
        budget.settle(spent, rv - v_ref)
        cum_above_floor += rv - v_ref
        running += rv - v_ref
        aggregate_min = min(aggregate_min, running)
        behavior = sf0.behavior_from_realization(x)
        if uses_public:
            for label, w in opponent_reach_weights(x, game=game).items():
                reach_accum[label] = reach_accum.get(label, 0.0) + w
        _pay, show, fold = native.simulate_showdown(
            game, behavior, opp_behavior, episodes_per_round, seed + t
        )
        for label, c in show.items():
            ev_point.record(label, c)
            ev_public.record(label, c)
        for label, c in fold.items():
            ev_public.record(label, c)

    tail = realized[max(0, len(realized) // 2) :]
    return {
        "exploitation_gain": mean(tail) - v_ref,
        "realized_value_worst": min(realized),
        "cumulative_above_floor": running,
        "aggregate_min": aggregate_min,
        "min_safety_value": min_safety,
        "safety_violation": max(0.0, v_ref - min_safety),
    }


@dataclass(frozen=True)
class _ShowdownScheduleCell:
    """Represent showdown schedule cell for the online workflow."""

    method_name: str
    seed: int
    game: str
    kind: str
    lure_behavior: dict[str, tuple[float, ...]]
    strike_behavior: dict[str, tuple[float, ...]]
    strike_round: int
    drift_power: float
    rounds: int
    episodes_per_round: int
    v_ref: float
    x_blueprint: tuple[float, ...]
    delta: float
    method: str
    eps_safe: float
    rho_cap: float
    kappa: float
    safety_debt_max: float
    rnr_p: float
    latch_threshold: float | None


def _run_showdown_schedule_cell(
    cell: _ShowdownScheduleCell,
) -> tuple[str, int, dict[str, Any]]:
    """Run the showdown schedule cell experiment."""
    lure_b = {k: list(v) for k, v in cell.lure_behavior.items()}
    strike_b = {k: list(v) for k, v in cell.strike_behavior.items()}
    if cell.kind == "lure_then_strike":

        def schedule(t, x, n):
            """Construct the round-by-round probing schedule."""
            return lure_b if t < cell.strike_round else strike_b

    elif cell.kind == "drift":

        def schedule(t, x, n):
            """Construct the round-by-round probing schedule."""
            w = (t / (n - 1)) ** cell.drift_power if n > 1 else 1.0
            return {
                lab: [
                    (1.0 - w) * p + w * q
                    for p, q in zip(lure_b[lab], strike_b[lab], strict=True)
                ]
                for lab in lure_b
            }

    else:
        raise ValueError(
            f"unknown schedule kind {cell.kind!r}; expected 'lure_then_strike' or 'drift'"
        )

    _t0 = perf_counter()
    result = _run_showdown_method_vs_schedule(
        cell.method_name,
        cell.game,
        schedule,
        cell.rounds,
        cell.episodes_per_round,
        cell.v_ref,
        cell.x_blueprint,
        cell.delta,
        cell.method,
        cell.eps_safe,
        cell.rho_cap,
        cell.kappa,
        cell.safety_debt_max,
        cell.rnr_p,
        cell.seed,
        cell.latch_threshold,
    )
    result["wall_time"] = perf_counter() - _t0
    return cell.method_name, cell.seed, result


def run_showdown_nonstationary(
    kind: str = "lure_then_strike",
    lure: str = "overfold",
    strike: str = "calling_station",
    strike_round: int | None = None,
    drift_power: float = 1.0,
    rounds: int = 60,
    episodes_per_round: int = 200,
    delta: float = 0.1,
    eps_safe: float = 0.0,
    method: str = "empirical_bernstein",
    rho_cap: float = 0.5,
    kappa: float = 0.2,
    safety_debt_max: float = 3.0,
    rnr_p: float = 0.5,
    game: str = "holdem",
    seeds: Sequence[int] = DEFAULT_SEEDS,
    latch_threshold: float | None = None,
    out_dir: str | Path | None = "results",
    workers: int | None = None,
) -> dict[str, Any]:
    """Run the showdown nonstationary experiment."""
    if strike_round is None:
        strike_round = rounds // 2
    blueprint = solve_blueprint(game, method="lp")
    v_ref = blueprint.value
    assert blueprint.realization is not None
    x_blueprint = tuple(blueprint.realization)
    lure_behavior = opponent_from_spec({"game": game, "type": lure}).behavior
    strike_behavior = opponent_from_spec({"game": game, "type": strike}).behavior

    if kind not in {"lure_then_strike", "drift"}:
        raise ValueError(
            f"unknown schedule kind {kind!r}; expected 'lure_then_strike' or 'drift'"
        )

    cells = [
        _ShowdownScheduleCell(
            method_name=name,
            seed=seed,
            game=game,
            kind=kind,
            lure_behavior={k: tuple(v) for k, v in lure_behavior.items()},
            strike_behavior={k: tuple(v) for k, v in strike_behavior.items()},
            strike_round=strike_round,
            drift_power=drift_power,
            rounds=rounds,
            episodes_per_round=episodes_per_round,
            v_ref=v_ref,
            x_blueprint=x_blueprint,
            delta=delta,
            method=method,
            eps_safe=eps_safe,
            rho_cap=rho_cap,
            kappa=kappa,
            safety_debt_max=safety_debt_max,
            rnr_p=rnr_p,
            latch_threshold=latch_threshold,
        )
        for name in _SHOWDOWN_METHODS
        for seed in seeds
    ]
    flat = _parallel_map(_run_showdown_schedule_cell, cells, workers)
    grouped: dict[str, list[dict[str, Any]]] = {}
    for method_name, _seed, result in flat:
        grouped.setdefault(method_name, []).append(result)

    floor = v_ref - eps_safe
    per_method: dict[str, Any] = {}
    for name in _SHOWDOWN_METHODS:
        guarantee = _SHOWDOWN_GUARANTEE[name]
        runs = grouped[name]
        violation_max = max(r["safety_violation"] for r in runs)
        if guarantee == "hard_safe":
            held = violation_max <= _SAFETY_GUARANTEE_TOL
        elif guarantee == "certified_budget":
            held = violation_max <= rho_cap + _SAFETY_GUARANTEE_TOL
        elif guarantee == "gift_funded":
            held = min(r["aggregate_min"] for r in runs) >= -_GIFT_GUARANTEE_TOL
        else:
            held = True
        gains = [r["exploitation_gain"] for r in runs]
        per_method[name] = {
            "guarantee": guarantee,
            "guarantee_held": held,
            "exploitation_gain_mean": mean(gains),
            "exploitation_gain_std": pstdev(gains) if len(gains) > 1 else 0.0,
            "exploitation_gain_ci95": _ci95_halfwidth(gains),
            "min_safety_value": min(r["min_safety_value"] for r in runs),
            "safety_violation_max": violation_max,
            "cumulative_above_floor_mean": mean(
                r["cumulative_above_floor"] for r in runs
            ),
        }

    results: dict[str, Any] = {
        "game": game,
        "monitoring": "showdown",
        "kind": kind,
        "lure": lure,
        "strike": strike,
        "strike_round": strike_round,
        "drift_power": drift_power,
        "seeds": list(seeds),
        "rounds": rounds,
        "episodes_per_round": episodes_per_round,
        "delta": delta,
        "safety_floor": floor,
        "latch_threshold": latch_threshold,
        "workers": workers,
        "method_names": list(_SHOWDOWN_METHODS),
        "methods": per_method,
    }
    if out_dir is not None:
        evaluation.save_results(
            results, Path(out_dir) / f"showdown_nonstationary_{kind}_{game}.json"
        )
    return results


def run_showdown_population(
    n: int = 50,
    population_seed: int = 2026,
    leak_range: tuple[float, float] = (0.3, 0.9),
    rounds: int = 60,
    episodes_per_round: int = 200,
    delta: float = 0.1,
    eps_safe: float = 0.0,
    method: str = "empirical_bernstein",
    rho_cap: float = 0.5,
    kappa: float = 0.2,
    safety_debt_max: float = 3.0,
    rnr_p: float = 0.5,
    game: str = "holdem",
    seeds: Sequence[int] = DEFAULT_SEEDS,
    latch_threshold: float | None = None,
    out_dir: str | Path | None = "results",
    workers: int | None = None,
) -> dict[str, Any]:
    """Run the showdown population experiment."""
    from ..opponents import holdem_population_sample

    suite = holdem_population_sample(
        n=n, seed=population_seed, game=game, leak_range=leak_range
    )
    comp = run_showdown_comparison(
        suite,
        rounds=rounds,
        episodes_per_round=episodes_per_round,
        delta=delta,
        eps_safe=eps_safe,
        method=method,
        rho_cap=rho_cap,
        kappa=kappa,
        safety_debt_max=safety_debt_max,
        rnr_p=rnr_p,
        seeds=seeds,
        out_dir=None,
        workers=workers,
        latch_threshold=latch_threshold,
    )

    opp_rows = comp["opponents"]
    floor = next(iter(opp_rows.values()))["game_value"] - rho_cap
    per_method: dict[str, Any] = {}
    for name in _SHOWDOWN_METHODS:
        gains = [opp_rows[o]["methods"][name]["exploitation_gain_mean"] for o in suite]
        min_safes = [opp_rows[o]["methods"][name]["min_safety_value"] for o in suite]
        held = [opp_rows[o]["methods"][name]["guarantee_held"] for o in suite]
        per_method[name] = {
            "guarantee": _SHOWDOWN_GUARANTEE[name],
            "gain_mean": mean(gains),
            "gain_std": pstdev(gains) if len(gains) > 1 else 0.0,
            "gain_min": min(gains),
            "gain_max": max(gains),
            "gain_ci95": _ci95_halfwidth(gains),
            "worst_min_safety": min(min_safes),
            "guarantee_held_all": all(held),
            "guarantee_held_frac": sum(held) / len(held),
        }

    oracle_gaps = [opp_rows[o]["oracle_gap"] for o in suite]

    results: dict[str, Any] = {
        "game": game,
        "monitoring": "showdown",
        "n_opponents": len(suite),
        "population_seed": population_seed,
        "leak_range": list(leak_range),
        "seeds": list(seeds),
        "rounds": rounds,
        "episodes_per_round": episodes_per_round,
        "delta": delta,
        "safety_floor": floor,
        "latch_threshold": latch_threshold,
        "method_names": list(_SHOWDOWN_METHODS),
        "oracle_gap_mean": mean(oracle_gaps),
        "oracle_gap_max": max(oracle_gaps),
        "oracle_gap_ci95": _ci95_halfwidth(oracle_gaps),
        "methods": per_method,
    }
    if out_dir is not None:
        evaluation.save_results(
            results, Path(out_dir) / f"showdown_population_{game}.json"
        )
    return results


def run_ablation(
    opponent: Opponent,
    deltas: Sequence[float] = (0.01, 0.05, 0.1, 0.2),
    eps_safes: Sequence[float] = (0.0, 0.05, 0.1, 0.2),
    methods: Sequence[str] = ("hoeffding", "empirical_bernstein"),
    rounds: int = 150,
    episodes_per_round: int = 50,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    out_dir: str | Path | None = "results",
) -> dict[str, list[dict[str, Any]]]:
    """Run the ablation experiment for the online workflow."""
    base = dict(
        rounds=rounds,
        episodes_per_round=episodes_per_round,
        seeds=seeds,
        out_dir=None,
    )

    def cell(**override: Any) -> dict[str, Any]:
        """Run one independently reproducible experiment cell."""
        params = {"delta": 0.05, "eps_safe": 0.0, "method": "hoeffding", **override}
        r = run_online_adaptation_replicated(opponent, **base, **params)
        return {
            **params,
            "exploitation_gain_mean": r["exploitation_gain_mean"],
            "exploitation_gain_std": r["exploitation_gain_std"],
            "min_safety_value": r["min_safety_value"],
            "safety_preserved": r["safety_preserved"],
        }

    ablation = {
        "delta": [cell(delta=d) for d in deltas],
        "eps_safe": [cell(eps_safe=e) for e in eps_safes],
        "method": [cell(method=m) for m in methods],
    }
    results = {"opponent": opponent.name, "ablation": ablation}
    if out_dir is not None:
        prefix = "ablation" if opponent.game == "kuhn" else f"ablation_{opponent.game}"
        evaluation.save_results(
            results, Path(out_dir) / f"{prefix}_{opponent.name}.json"
        )
    return ablation


__all__ = [
    "run_online_adaptation",
    "run_suite",
    "run_online_adaptation_replicated",
    "run_ablation",
    "DEFAULT_SEEDS",
    "KNOWN_KUHN_VALUE",
]
