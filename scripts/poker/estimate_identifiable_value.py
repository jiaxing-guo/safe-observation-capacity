"""Estimate the identifiable value experiment. See Experiments and supplementary Certification at the Unbucketed River."""

import json
import os
from pathlib import Path
from typing import Any

from safe_observation.confidence import OpponentEvidenceStore
from safe_observation.opponents import Opponent, holdem_equilibrium_opponent
from safe_observation.payoff import build as build_payoff
from safe_observation.sequence_form import compile as compile_game
from safe_observation.solvers import (
    opponent_reach_weights,
    robust_safe_response_linear,
    robust_safe_response_public,
    safety_constrained_best_response,
    safety_verifier,
    solve_blueprint,
)
from scripts.poker.run_turn_river_methods import _build_opponents, _perturb, _top_rank

GAME = os.environ.get("IDENTIFIABLE_VALUE_GAME", "holdem_tr_b4")
RHO = float(os.environ.get("IDENTIFIABLE_VALUE_RHO", "0.5"))


LEAN = os.environ.get("IDENTIFIABLE_VALUE_LEAN", "0") == "1"
OPPONENT_FILTER = [
    x.strip()
    for x in os.environ.get("IDENTIFIABLE_VALUE_OPPONENTS", "").split(",")
    if x.strip()
]
OUT = Path(
    os.environ.get("IDENTIFIABLE_VALUE_OUT", f"results/identifiable_value_{GAME}.json")
)


def _fold_action_indices(sf1) -> dict[str, int]:
    """Compute fold action indices for the estimate identifiable value workflow."""
    out: dict[str, int] = {}
    for info in sf1.info_sets:
        for idx, (action, _child) in enumerate(info.children):
            if action == "f":
                out[info.label] = idx
                break
    return out


def _diagnostic_opponents(
    game: str, actions: dict[str, list[str]], eq: dict[str, list[float]]
) -> dict[str, dict[str, Any]]:
    """Compute diagnostic opponents for the estimate identifiable value workflow."""
    revealed_call = _perturb(
        eq,
        actions,
        lambda hole, hist, acts: (
            "/" in hist and "f" in acts and "c" in acts and _top_rank(hole) >= 9
        ),
        "c",
        0.6,
    )
    return {
        "identifiable_revealed_call_strong": {
            "behavior": revealed_call,
            "component": "revealed / identifiable",
        },
    }


def _population_public_intervals(
    groups: dict[str, list[str]],
    info_by_label: dict,
    y_star: list[float],
    omega: dict[str, float],
    fold_only: bool = False,
    fold_idx: dict[str, int] | None = None,
) -> dict[str, list[tuple[float, float]]]:
    """Compute the population public intervals."""
    fold_idx = fold_idx or {}
    out: dict[str, list[tuple[float, float]]] = {}
    for key, members in groups.items():
        infos = [info_by_label[label] for label in members]
        n_actions = len(infos[0].children)
        fi = fold_idx.get(members[0]) if fold_only else None
        row: list[tuple[float, float]] = []
        for a in range(n_actions):
            if fold_only and a != fi:
                row.append((0.0, 1.0))
                continue
            num = 0.0
            den = 0.0
            for info in infos:
                w = omega.get(info.label, 0.0)
                if w <= 0.0:
                    continue
                child = info.children[a][1]
                num += w * y_star[child]
                den += w * y_star[info.parent_seq]
            p = num / den if den > 1e-15 else 0.0
            p = min(1.0, max(0.0, p))
            row.append((p, p))
        out[key] = row
    return out


def _population_event_constraints(
    sf1, y_star: list[float], omega: dict[str, float], fold_idx: dict[str, int]
) -> tuple[list[tuple[int, int, float]], list[float], list[tuple[str, int]]]:
    """Compute population event constraints for the estimate identifiable value workflow."""
    entries: list[tuple[int, int, float]] = []
    h: list[float] = []
    meta: list[tuple[str, int]] = []
    row = 0
    for info in sf1.info_sets:
        w = omega.get(info.label, 0.0)
        if w <= 0.0:
            continue
        fi = fold_idx.get(info.label)
        for a_idx, (_action, child) in enumerate(info.children):
            if fi is not None and a_idx == fi:
                continue
            value = y_star[child]
            entries.append((row, child, 1.0))
            h.append(value)
            meta.append((info.label, a_idx))
            row += 1
            entries.append((row, child, -1.0))
            h.append(-value)
            meta.append((info.label, a_idx))
            row += 1
    return entries, h, meta


def main() -> None:
    """Run the command-line entry point."""
    game = GAME
    print(
        f"# C_obs IDEAL value test (Q1, population/exact)  game={game}  rho={RHO}  lean={LEAN}",
        flush=True,
    )
    payoff = build_payoff(game)
    sf1 = compile_game(game, 1)
    info_by_label = {info.label: info for info in sf1.info_sets}
    bp = solve_blueprint(game, method="lp")
    assert bp.realization is not None
    x_bp = bp.realization
    v_ref = bp.value
    omega = opponent_reach_weights(x_bp, game=game)
    fold_idx = _fold_action_indices(sf1)
    actions = {info.label: [a for a, _ in info.children] for info in sf1.info_sets}
    groups = OpponentEvidenceStore.for_game(game).public_groups()

    eq = holdem_equilibrium_opponent(game).behavior
    opponents: dict[str, dict[str, Any]] = {
        name: {"behavior": behavior, "component": "method-table"}
        for name, behavior in _build_opponents(game, actions).items()
    }
    opponents.update(_diagnostic_opponents(game, actions, eq))
    if OPPONENT_FILTER:
        missing = sorted(set(OPPONENT_FILTER) - set(opponents))
        if missing:
            raise ValueError(f"unknown IDENTIFIABLE_VALUE_OPPONENTS entries: {missing}")
        opponents = {name: opponents[name] for name in OPPONENT_FILTER}

    print(
        f"  sequences={sf1.num_sequences} infosets={len(sf1.info_sets)} "
        f"public_states={len(groups)} v_ref={v_ref:+.5f}",
        flush=True,
    )
    print(
        f"  {'opponent':<34} {'core':>8} {'obs':>8} {'oracle':>8} "
        f"{'obs-core':>9} {'orac-obs':>9} {'core_cert':>9} {'obs_cert':>9} {'min_safe':>9}",
        flush=True,
    )

    floor = v_ref - RHO
    rows: list[dict[str, Any]] = []
    for name, case in opponents.items():
        behavior = case["behavior"]
        y_star = list(Opponent(name=name, behavior=behavior, game=game).realization())

        pub_intervals = _population_public_intervals(
            groups, info_by_label, y_star, omega
        )
        event_entries, event_h, event_meta = _population_event_constraints(
            sf1, y_star, omega, fold_idx
        )
        obs_pub_intervals = (
            _population_public_intervals(
                groups, info_by_label, y_star, omega, fold_only=True, fold_idx=fold_idx
            )
            if LEAN
            else pub_intervals
        )

        core = robust_safe_response_public(
            groups, pub_intervals, v_ref=v_ref, eps_safe=RHO, game=game, weights=omega
        )
        obs = robust_safe_response_linear(
            groups,
            obs_pub_intervals,
            event_entries,
            event_h,
            v_ref=v_ref,
            eps_safe=RHO,
            game=game,
            weights=omega,
            row_meta=event_meta,
        )
        oracle = safety_constrained_best_response(
            behavior, v_ref=v_ref, eps_safe=RHO, game=game
        )

        def realized(response, y_star=y_star) -> float:
            """Evaluate a realization plan against the selected opponent."""
            return payoff.bilinear(list(response.realization), y_star) - v_ref

        core_gain = realized(core)
        obs_gain = realized(obs)
        oracle_gain = realized(oracle)
        min_safety = min(
            safety_verifier(list(response.realization), game=game).value
            for response in (core, obs, oracle)
        )
        row = {
            "opponent": name,
            "component": case["component"],
            "rho": RHO,
            "floor": floor,
            "core": core_gain,
            "obs": obs_gain,
            "oracle": oracle_gain,
            "obs_minus_core": obs_gain - core_gain,
            "oracle_minus_obs": oracle_gain - obs_gain,
            "core_cert_exploit": core.robust_value - v_ref,
            "obs_cert_exploit": obs.robust_value - v_ref,
            "n_event_rows": len(event_h),
            "min_safety": min_safety,
            "safety_margin": min_safety - floor,
        }
        rows.append(row)
        print(
            f"  {name:<34} {core_gain:>+8.3f} {obs_gain:>+8.3f} {oracle_gain:>+8.3f} "
            f"{row['obs_minus_core']:>+9.3f} {row['oracle_minus_obs']:>+9.3f} "
            f"{row['core_cert_exploit']:>+9.3f} {row['obs_cert_exploit']:>+9.3f} "
            f"{min_safety:>+9.3f}",
            flush=True,
        )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps(
            {
                "game": game,
                "rho": RHO,
                "lean": LEAN,
                "v_ref": v_ref,
                "n_public_states": len(groups),
                "rows": rows,
            },
            indent=2,
        )
    )
    print(f"\nwrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
