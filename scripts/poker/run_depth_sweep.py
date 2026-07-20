""

import json
import os
from pathlib import Path
import random
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
from scripts.poker.evaluate_public_routing import _pearson, _spearman
from scripts.poker.evaluate_reveal_routing import _fold_pub_signal

TYPES = int(os.environ.get("DSW_TYPES", "4"))
DEPTHS = [int(x) for x in os.environ.get("DSW_DEPTHS", "1,2,3,4,5").split(",")]
RHO = float(os.environ.get("DSW_RHO", "0.5"))
WEIGHT = float(os.environ.get("DSW_WEIGHT", "0.6"))
LEAK_ROUND = os.environ.get("DSW_LEAK_ROUND", "last")
SUBPOP = int(os.environ.get("DSW_SUBPOP", "40"))
BASE_SEED = int(os.environ.get("DSW_SEED", "2026"))
OUT = Path(os.environ.get("DEPTH_SWEEP_OUT", "results/depth_sweep_cchain.json"))


def _round_of(hist: str) -> int:
    ""
    return hist.count("/")


def _is_fold_infoset(label: str, actions: dict[str, list[str]]) -> bool:
    return "f" in actions.get(label, [])


def _canonical_leak(
    game: str,
    sf1,
    actions: dict[str, list[str]],
    eq: dict[str, list[float]],
    depth: int,
    weight: float,
    leak_round: str,
    high_types_only: bool = True,
) -> dict[str, list[float]]:
    ""
    out = {label: list(dist) for label, dist in eq.items()}
    high = {str(t) for t in range(max(1, TYPES // 2), TYPES)}
    for info in sf1.info_sets:
        typ, hist = info.label.split("|", 1)
        if not _is_fold_infoset(info.label, actions):
            continue
        rnd = _round_of(hist)
        target_round = {
            "first": 0,
            "last": depth - 1,
        }.get(leak_round, rnd)
        if leak_round != "all" and rnd != target_round:
            continue
        if high_types_only and typ not in high:
            continue
        fi = actions[info.label].index("f")
        d = out[info.label]
        out[info.label] = [
            (1.0 - weight) * p + (weight if k == fi else 0.0) for k, p in enumerate(d)
        ]
    return out


def _random_chain_opponent(
    rng: random.Random,
    sf1,
    actions: dict[str, list[str]],
    eq: dict[str, list[float]],
    depth: int,
) -> dict[str, list[float]]:
    ""
    out = {label: list(dist) for label, dist in eq.items()}
    leak_round = rng.randrange(depth)
    types_hit = {str(t) for t in range(TYPES) if rng.random() < 0.5}
    weight = rng.uniform(0.1, 0.8)
    for info in sf1.info_sets:
        typ, hist = info.label.split("|", 1)
        if not _is_fold_infoset(info.label, actions):
            continue
        if _round_of(hist) != leak_round or typ not in types_hit:
            continue
        fi = actions[info.label].index("f")
        d = out[info.label]
        out[info.label] = [
            (1.0 - weight) * p + (weight if k == fi else 0.0) for k, p in enumerate(d)
        ]
    return out


def _solve_gap(game, sf1, info_by, payoff, v_ref, omega, fold_idx, groups, behavior):
    ""
    y = list(Opponent(name="x", behavior=behavior, game=game).realization())
    pub = _population_public_intervals(groups, info_by, y, omega)
    obspub = _population_public_intervals(
        groups, info_by, y, omega, fold_only=True, fold_idx=fold_idx
    )
    ee, eh, em = _population_event_constraints(sf1, y, omega, fold_idx)
    core = robust_safe_response_public(
        groups, pub, v_ref=v_ref, eps_safe=RHO, game=game, weights=omega
    )
    obs = robust_safe_response_linear(
        groups,
        obspub,
        ee,
        eh,
        v_ref=v_ref,
        eps_safe=RHO,
        game=game,
        weights=omega,
        row_meta=em,
    )
    oracle = safety_constrained_best_response(
        behavior, v_ref=v_ref, eps_safe=RHO, game=game
    )
    g_core = payoff.bilinear(list(core.realization), y) - v_ref
    g_obs = payoff.bilinear(list(obs.realization), y) - v_ref
    g_oracle = payoff.bilinear(list(oracle.realization), y) - v_ref
    min_safe = min(
        safety_verifier(list(r.realization), game=game).value
        for r in (core, obs, oracle)
    )
    return g_core, g_obs, g_oracle, min_safe, y


def main() -> None:
    print(
        f"depth sweep  K={TYPES}  depths={DEPTHS}  rho={RHO}  "
        f"leak_round={LEAK_ROUND}  subpop={SUBPOP}",
        flush=True,
    )
    print(
        f"  {'depth':>5} {'pub_states':>10} {'core':>8} {'obs':>8} {'oracle':>8} "
        f"{'Delta':>8} {'min_safe':>9} {'floor':>8} {'pearson':>8} {'spearman':>9}",
        flush=True,
    )
    rows: list[dict[str, Any]] = []
    for depth in DEPTHS:
        game = f"cchain_d{depth}_k{TYPES}"
        sf1 = compile_game(game, 1)
        info_by = {i.label: i for i in sf1.info_sets}
        payoff = build_payoff(game)
        bp = solve_blueprint(game, method="lp")
        v_ref = bp.value
        omega = opponent_reach_weights(bp.realization, game=game)
        fold_idx = _fold_action_indices(sf1)
        groups = OpponentEvidenceStore.for_game(game).public_groups()
        eq = holdem_equilibrium_opponent(game).behavior
        y_eq = list(Opponent(name="eq", behavior=eq, game=game).realization())
        actions = {i.label: [a for a, _ in i.children] for i in sf1.info_sets}
        floor = v_ref - RHO

        leak = _canonical_leak(game, sf1, actions, eq, depth, WEIGHT, LEAK_ROUND)
        g_core, g_obs, g_oracle, min_safe, _ = _solve_gap(
            game, sf1, info_by, payoff, v_ref, omega, fold_idx, groups, leak
        )
        delta = g_obs - g_core

        d_es: list[float] = []
        deltas: list[float] = []
        for s in range(SUBPOP):
            rng = random.Random(BASE_SEED + 1000 * depth + s)
            beh = _random_chain_opponent(rng, sf1, actions, eq, depth)
            gc, go, _go2, _ms, y = _solve_gap(
                game, sf1, info_by, payoff, v_ref, omega, fold_idx, groups, beh
            )
            d_e = _fold_pub_signal(
                sf1, y, y_eq, omega, fold_idx, groups, info_by, restrict="turn"
            )
            d_es.append(d_e)
            deltas.append(go - gc)
        pear = _pearson(d_es, deltas)
        spear = _spearman(d_es, deltas)

        row = {
            "depth": depth,
            "game": game,
            "n_public_states": len(groups),
            "v_ref": v_ref,
            "floor": floor,
            "core": g_core,
            "obs": g_obs,
            "oracle": g_oracle,
            "delta": delta,
            "min_safety": min_safe,
            "pearson_De_Delta": pear,
            "spearman_De_Delta": spear,
            "subpop": SUBPOP,
            "subpop_delta_mean": sum(deltas) / len(deltas) if deltas else 0.0,
            "subpop_delta_max": max(deltas) if deltas else 0.0,
        }
        rows.append(row)
        print(
            f"  {depth:>5} {len(groups):>10} {g_core:>+8.3f} {g_obs:>+8.3f} "
            f"{g_oracle:>+8.3f} {delta:>+8.3f} {min_safe:>+9.3f} {floor:>+8.3f} "
            f"{pear:>+8.3f} {spear:>+9.3f}",
            flush=True,
        )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps(
            {
                "types": TYPES,
                "depths": DEPTHS,
                "rho": RHO,
                "weight": WEIGHT,
                "leak_round": LEAK_ROUND,
                "subpop": SUBPOP,
                "rows": rows,
            },
            indent=2,
        )
    )
    print(f"\nwrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
