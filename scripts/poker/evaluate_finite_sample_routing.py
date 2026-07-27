"""Evaluate the finite sample routing experiment. See Experiments and supplementary Certification at the Unbucketed River."""

import json
import os
from pathlib import Path
import statistics
from typing import Any

from safe_observation import native
from safe_observation.confidence import OpponentEvidenceStore
from safe_observation.opponents import Opponent, holdem_equilibrium_opponent
from safe_observation.sequence_form import compile as compile_game
from safe_observation.solvers import opponent_reach_weights, solve_blueprint
from scripts.poker.estimate_identifiable_value import _fold_action_indices
from scripts.poker.evaluate_public_routing import _build_gate_suite, _pearson, _spearman

GAME = os.environ.get("FINITE_ROUTING_GAME", "holdem_tr_b2")
GATE_JSON = Path(
    os.environ.get(
        "FINITE_ROUTING_GATE_JSON", "results/public_routing_holdem_tr_b2.json"
    )
)
EPISODE_GRID = [
    int(x)
    for x in os.environ.get("FINITE_ROUTING_GRID", "1000,10000,100000").split(",")
]
SEEDS = [
    int(x)
    for x in os.environ.get("FINITE_ROUTING_SEEDS", "2026,2027,2028,2029,2030").split(
        ","
    )
]
SIGMA_K = float(os.environ.get("FINITE_ROUTING_SIGMA_K", "4.0"))
OUT = Path(
    os.environ.get("FINITE_ROUTING_OUT", f"results/finite_sample_routing_{GAME}.json")
)


def _is_river_key(key: str) -> bool:
    """Compute is river key for the evaluate finite sample routing workflow."""
    return "/" in key


def _pf_eq_turn(
    groups, info_by_label, y_eq, omega, fold_idx
) -> dict[str, tuple[float, int]]:
    """Compute pf eq turn for the evaluate finite sample routing workflow."""
    out: dict[str, tuple[float, int]] = {}
    for key, members in groups.items():
        if _is_river_key(key):
            continue
        fi = next((fold_idx[m] for m in members if m in fold_idx), None)
        if fi is None:
            continue
        num = den = 0.0
        for m in members:
            info = info_by_label[m]
            w = omega.get(m, 0.0)
            if w <= 0.0 or fi >= len(info.children):
                continue
            num += w * y_eq[info.children[fi][1]]
            den += w * y_eq[info.parent_seq]
        pf_eq = num / den if den > 1e-12 else 0.0
        out[key] = (pf_eq, fi)
    return out


def _empirical_signal(ev_public, pf_eq_turn, episodes: int) -> float:
    """Compute empirical signal for the evaluate finite sample routing workflow."""
    counts = ev_public._public_counts()
    total = 0.0
    for key, (pf_eq, fi) in pf_eq_turn.items():
        c = counts.get(key)
        if not c:
            continue
        visits = sum(c)
        if visits == 0 or fi >= len(c):
            continue
        pf_hat = c[fi] / visits
        total += (visits / episodes) * abs(pf_hat - pf_eq)
    return total


def main() -> None:
    """Run the command-line entry point."""
    game = GAME
    print(
        f"# Q4b FINITE-SAMPLE validation of fold_pub_turn  game={game}  "
        f"grid={EPISODE_GRID}  seeds={len(SEEDS)}",
        flush=True,
    )
    gate = json.loads(GATE_JSON.read_text())
    lift_by_name = {r["opponent"]: r["obs_minus_core"] for r in gate["rows"]}
    arms_by_name = {r["opponent"]: r for r in gate["rows"]}

    sf0 = compile_game(game, 0)
    sf1 = compile_game(game, 1)
    info_by_label = {info.label: info for info in sf1.info_sets}
    bp = solve_blueprint(game, method="lp")
    assert bp.realization is not None
    x_bp = bp.realization
    agent_behavior = sf0.behavior_from_realization(x_bp)
    omega = opponent_reach_weights(x_bp, game=game)
    fold_idx = _fold_action_indices(sf1)
    actions = {info.label: [a for a, _ in info.children] for info in sf1.info_sets}
    groups = OpponentEvidenceStore.for_game(game).public_groups()
    eq = holdem_equilibrium_opponent(game).behavior
    y_eq = list(Opponent(name="eq", behavior=eq, game=game).realization())
    pf_eq_turn = _pf_eq_turn(groups, info_by_label, y_eq, omega, fold_idx)
    suite = _build_gate_suite(game, actions, eq)

    records: dict[str, dict[int, list[float]]] = {}
    for name, behavior in suite.items():
        if name not in lift_by_name:
            continue
        records[name] = {}
        for episodes in EPISODE_GRID:
            vals: list[float] = []
            for seed in SEEDS:
                ev_public = OpponentEvidenceStore.for_game(game)
                _pay, show, fold = native.simulate_showdown(
                    game, agent_behavior, behavior, episodes, seed
                )
                for label, c in show.items():
                    ev_public.record(label, c)
                for label, c in fold.items():
                    ev_public.record(label, c)
                vals.append(_empirical_signal(ev_public, pf_eq_turn, episodes))
            records[name][episodes] = vals

    eq_name = "tr_equilibrium"
    floor_stats = {
        n: (statistics.mean(records[eq_name][n]), _std(records[eq_name][n]))
        for n in EPISODE_GRID
    }

    print("\n  per-opponent empirical signal mean +/- std (by N):", flush=True)
    header = "  {:<22} {:>8}".format("opponent", "lift")
    for n in EPISODE_GRID:
        header += f"  {('N=' + str(n)):>16}"
    print(header, flush=True)
    rows_out: list[dict[str, Any]] = []
    for name in records:
        line = f"  {name:<22} {lift_by_name[name]:>+8.3f}"
        per_n: dict[int, dict[str, float]] = {}
        for n in EPISODE_GRID:
            m = statistics.mean(records[name][n])
            s = _std(records[name][n])
            per_n[n] = {"mean": m, "std": s}
            line += f"  {m:>7.4f}+-{s:<7.4f}"
        print(line, flush=True)
        rows_out.append({"opponent": name, "lift": lift_by_name[name], "per_n": per_n})

    print("\n  signal-vs-lift correlation and gate accuracy (by N):", flush=True)
    summary: dict[int, dict[str, Any]] = {}
    for n in EPISODE_GRID:
        means = [statistics.mean(records[name][n]) for name in records]
        lifts = [lift_by_name[name] for name in records]
        pear = _pearson(means, lifts)
        spear = _spearman(means, lifts)
        floor_mean, floor_std = floor_stats[n]
        tau = floor_mean + SIGMA_K * floor_std

        engaged = [name for name in records if statistics.mean(records[name][n]) > tau]
        available = sum(lifts)
        captured = sum(lift_by_name[name] for name in engaged)

        oracle_tot = sum(arms_by_name[name]["oracle"] for name in records)
        always_core = sum(arms_by_name[name]["core"] for name in records)
        always_obs = sum(arms_by_name[name]["obs"] for name in records)
        gate_val = sum(
            arms_by_name[name]["obs"] if name in engaged else arms_by_name[name]["core"]
            for name in records
        )

        true_engage = {name for name in records if lift_by_name[name] > 0.02}
        seed_correct = 0
        seed_total = 0
        for name in records:
            for v in records[name][n]:
                decided = v > tau
                seed_correct += int(decided == (name in true_engage))
                seed_total += 1
        summary[n] = {
            "pearson": pear,
            "spearman": spear,
            "tau": tau,
            "floor_mean": floor_mean,
            "floor_std": floor_std,
            "solve_rate": len(engaged) / len(records),
            "captured_frac": captured / available if abs(available) > 1e-9 else 0.0,
            "gate_value": gate_val,
            "always_core": always_core,
            "always_obs": always_obs,
            "oracle": oracle_tot,
            "gate_capture_of_max": (gate_val - always_core) / (always_obs - always_core)
            if abs(always_obs - always_core) > 1e-9
            else 0.0,
            "per_seed_accuracy": seed_correct / seed_total if seed_total else 0.0,
            "engaged": engaged,
        }
        print(
            f"    N={n:>7}  pearson={pear:+.3f}  spearman={spear:+.3f}  "
            f"tau={tau:.4f}  solve_rate={len(engaged) / len(records):.2f}  "
            f"captured={summary[n]['captured_frac']:+.3f}  "
            f"gate_capture_of_max={summary[n]['gate_capture_of_max']:+.3f}  "
            f"seed_acc={summary[n]['per_seed_accuracy']:.2f}",
            flush=True,
        )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps(
            {
                "game": game,
                "grid": EPISODE_GRID,
                "seeds": SEEDS,
                "sigma_k": SIGMA_K,
                "floor_stats": {str(n): floor_stats[n] for n in EPISODE_GRID},
                "rows": rows_out,
                "summary": {str(n): summary[n] for n in EPISODE_GRID},
            },
            indent=2,
        )
    )
    print(f"\nwrote {OUT}", flush=True)


def _std(values: list[float]) -> float:
    """Compute the sample standard deviation of the supplied values."""
    return statistics.stdev(values) if len(values) > 1 else 0.0


if __name__ == "__main__":
    main()
