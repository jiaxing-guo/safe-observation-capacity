"""Evaluate the identification scope experiment. See Experiments and supplementary Certification at the Unbucketed River."""

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
from scripts.poker.estimate_identifiable_value import (
    _fold_action_indices,
    _population_event_constraints,
    _population_public_intervals,
)
from scripts.poker.run_turn_river_methods import _build_opponents, _perturb, _top_rank

GAME = os.environ.get("IDENTIFICATION_SCOPE_GAME", "holdem_tr_b2")
RHO = float(os.environ.get("IDENTIFICATION_SCOPE_RHO", "0.5"))
KFRACS = [
    float(x)
    for x in os.environ.get(
        "IDENTIFICATION_SCOPE_KFRACS", "0.01,0.02,0.05,0.1,0.25,0.5,1.0"
    ).split(",")
]
OPPONENT_FILTER = [
    x.strip()
    for x in os.environ.get(
        "IDENTIFICATION_SCOPE_OPPONENTS", "tr_turn_overfold,tr_river_overfold_strong"
    ).split(",")
    if x.strip()
]
OUT = Path(
    os.environ.get(
        "IDENTIFICATION_SCOPE_OUT", f"results/identification_scope_{GAME}.json"
    )
)


def _pin_importance(
    sf1,
    y_star: list[float],
    omega: dict[str, float],
    fold_idx: dict[str, int],
    pub_intervals: dict[str, list[tuple[float, float]]],
    info_by_label: dict,
    groups: dict[str, list[str]],
) -> dict[tuple[str, int], float]:
    """Compute pin importance for the evaluate identification scope workflow."""

    label_to_key: dict[str, str] = {}
    for key, members in groups.items():
        for label in members:
            label_to_key[label] = key

    content: dict[tuple[str, int], float] = {}
    for info in sf1.info_sets:
        w = omega.get(info.label, 0.0)
        if w <= 0.0:
            continue
        fi = fold_idx.get(info.label)
        key = label_to_key.get(info.label)
        bounds = pub_intervals.get(key, []) if key is not None else []
        parent_val = y_star[info.parent_seq]
        for a_idx, (_action, child) in enumerate(info.children):
            if fi is not None and a_idx == fi:
                continue
            p_pub = bounds[a_idx][0] if a_idx < len(bounds) else 0.0
            expected = p_pub * parent_val
            content[(info.label, a_idx)] = w * abs(y_star[child] - expected)
    return content


def _topk_event_constraints(
    sf1,
    y_star: list[float],
    omega: dict[str, float],
    fold_idx: dict[str, int],
    keep: set[tuple[str, int]],
) -> tuple[list[tuple[int, int, float]], list[float], list[tuple[str, int]]]:
    """Compute topk event constraints for the evaluate identification scope workflow."""
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
            if (info.label, a_idx) not in keep:
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
        f"# C_obs SAD test (Q3, top-k event pins by identifiability content)  "
        f"game={game}  rho={RHO}",
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

    base = _build_opponents(game, actions)
    eq = holdem_equilibrium_opponent(game).behavior

    revealed_call = _perturb(
        eq,
        actions,
        lambda hole, hist, acts: (
            "/" in hist and "f" in acts and "c" in acts and _top_rank(hole) >= 9
        ),
        "c",
        0.6,
    )
    opponents = dict(base)
    opponents["identifiable_revealed_call_strong"] = revealed_call
    names = [n for n in OPPONENT_FILTER if n in opponents] or list(opponents)

    floor = v_ref - RHO
    results: list[dict[str, Any]] = []
    for name in names:
        behavior = opponents[name]
        y_star = list(Opponent(name=name, behavior=behavior, game=game).realization())
        pub_intervals = _population_public_intervals(
            groups, info_by_label, y_star, omega
        )

        core = robust_safe_response_public(
            groups, pub_intervals, v_ref=v_ref, eps_safe=RHO, game=game, weights=omega
        )

        obs_pub = _population_public_intervals(
            groups, info_by_label, y_star, omega, fold_only=True, fold_idx=fold_idx
        )
        full_entries, full_h, full_meta = _population_event_constraints(
            sf1, y_star, omega, fold_idx
        )
        obs_full = robust_safe_response_linear(
            groups,
            obs_pub,
            full_entries,
            full_h,
            v_ref=v_ref,
            eps_safe=RHO,
            game=game,
            weights=omega,
            row_meta=full_meta,
        )

        def realized(response, y_star=y_star) -> float:
            """Evaluate a realization plan against the selected opponent."""
            return payoff.bilinear(list(response.realization), y_star) - v_ref

        core_gain = realized(core)
        obs_full_gain = realized(obs_full)
        full_gap = obs_full_gain - core_gain

        content = _pin_importance(
            sf1, y_star, omega, fold_idx, pub_intervals, info_by_label, groups
        )
        ranked = sorted(content.items(), key=lambda kv: kv[1], reverse=True)
        n_pins = len(ranked)

        print(
            f"\n== {name} ==  core={core_gain:+.3f}  obs_full={obs_full_gain:+.3f}  "
            f"full_gap={full_gap:+.3f}  n_pins={n_pins}",
            flush=True,
        )
        print(
            f"  {'kfrac':>7} {'kpins':>6} {'obs_k':>8} {'recovered':>10} {'min_safe':>9}",
            flush=True,
        )
        sweep: list[dict[str, Any]] = []
        for kfrac in KFRACS:
            k = max(1, int(round(kfrac * n_pins)))
            keep = {label_action for label_action, _ in ranked[:k]}
            entries, h, meta = _topk_event_constraints(
                sf1, y_star, omega, fold_idx, keep
            )
            obs_k = robust_safe_response_linear(
                groups,
                obs_pub,
                entries,
                h,
                v_ref=v_ref,
                eps_safe=RHO,
                game=game,
                weights=omega,
                row_meta=meta,
            )
            obs_k_gain = realized(obs_k)
            recovered = (
                (obs_k_gain - core_gain) / full_gap if abs(full_gap) > 1e-9 else 0.0
            )
            min_safety = min(
                safety_verifier(list(r.realization), game=game).value
                for r in (core, obs_k)
            )
            sweep.append(
                {
                    "kfrac": kfrac,
                    "k": k,
                    "obs_k": obs_k_gain,
                    "recovered_frac": recovered,
                    "min_safety": min_safety,
                }
            )
            print(
                f"  {kfrac:>7.3f} {k:>6} {obs_k_gain:>+8.3f} {recovered:>10.3f} "
                f"{min_safety:>+9.3f}",
                flush=True,
            )

        oracle = safety_constrained_best_response(
            behavior, v_ref=v_ref, eps_safe=RHO, game=game
        )
        results.append(
            {
                "opponent": name,
                "rho": RHO,
                "floor": floor,
                "core": core_gain,
                "obs_full": obs_full_gain,
                "oracle": realized(oracle),
                "full_gap": full_gap,
                "n_pins": n_pins,
                "sweep": sweep,
            }
        )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps(
            {
                "game": game,
                "rho": RHO,
                "v_ref": v_ref,
                "kfracs": KFRACS,
                "rows": results,
            },
            indent=2,
        )
    )
    print(f"\nwrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
