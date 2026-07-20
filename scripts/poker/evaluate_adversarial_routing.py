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
    safety_verifier,
    solve_blueprint,
)
from scripts.poker.estimate_identifiable_value import (
    _fold_action_indices,
    _population_event_constraints,
    _population_public_intervals,
)
from scripts.poker.evaluate_reveal_routing import _fold_pub_signal
from scripts.poker.run_opponent_population import _random_opponent
from scripts.poker.run_turn_river_methods import _perturb, _top_rank

GAME = os.environ.get("ADV_GAME", "holdem_tr_b2")
RHO = float(os.environ.get("ADV_RHO", "0.5"))
N_RANDOM = int(os.environ.get("ADV_N_RANDOM", "200"))
BASE_SEED = int(os.environ.get("ADV_SEED", "2026"))


EPS_FRAC = float(os.environ.get("ADV_EPS_FRAC", "0.15"))
OUT = Path(
    os.environ.get(
        "ADVERSARIAL_ROUTING_OUT", f"results/adversarial_routing_{GAME}.json"
    )
)


def _structured_suite(game, actions, eq):
    ""

    def river_pick(hole, hist, acts):
        return "/" in hist and "f" in acts

    def turn_pick(hole, hist, acts):
        return "/" not in hist and "f" in acts

    suite: dict[str, dict[str, float]] = {"tr_equilibrium": eq}
    for w in (0.3, 0.6, 0.9):
        suite[f"turn_overfold_all_w{int(w * 100)}"] = _perturb(
            eq, actions, turn_pick, "f", w
        )
        suite[f"river_overfold_all_w{int(w * 100)}"] = _perturb(
            eq, actions, river_pick, "f", w
        )

    suite["turn_overfold_strong"] = _perturb(
        eq,
        actions,
        turn_pick,
        "f",
        lambda hole, hist: 0.6 if _top_rank(hole) >= 9 else 0.0,
    )
    suite["river_overfold_strong"] = _perturb(
        eq,
        actions,
        river_pick,
        "f",
        lambda hole, hist: 0.6 if _top_rank(hole) >= 9 else 0.0,
    )
    suite["revealed_call_strong"] = _perturb(
        eq,
        actions,
        lambda hole, hist, acts: (
            "/" in hist and "f" in acts and "c" in acts and _top_rank(hole) >= 9
        ),
        "c",
        0.6,
    )
    return suite


def main() -> None:
    print(
        f"# TA2 adversarial gate envelope  game={GAME}  rho={RHO}  "
        f"n_random={N_RANDOM}  eps_frac={EPS_FRAC}",
        flush=True,
    )
    sf1 = compile_game(GAME, 1)
    info_by = {i.label: i for i in sf1.info_sets}
    payoff = build_payoff(GAME)
    bp = solve_blueprint(GAME, method="lp")
    v_ref = bp.value
    omega = opponent_reach_weights(bp.realization, game=GAME)
    fold_idx = _fold_action_indices(sf1)
    groups = OpponentEvidenceStore.for_game(GAME).public_groups()
    actions = {i.label: [a for a, _ in i.children] for i in sf1.info_sets}
    eq = holdem_equilibrium_opponent(GAME).behavior
    y_eq = list(Opponent(name="eq", behavior=eq, game=GAME).realization())
    floor = v_ref - RHO

    def evaluate(name, behavior, kind):
        y = list(Opponent(name=name, behavior=behavior, game=GAME).realization())
        pub = _population_public_intervals(groups, info_by, y, omega)
        obspub = _population_public_intervals(
            groups, info_by, y, omega, fold_only=True, fold_idx=fold_idx
        )
        ee, eh, em = _population_event_constraints(sf1, y, omega, fold_idx)
        core = robust_safe_response_public(
            groups, pub, v_ref=v_ref, eps_safe=RHO, game=GAME, weights=omega
        )
        obs = robust_safe_response_linear(
            groups,
            obspub,
            ee,
            eh,
            v_ref=v_ref,
            eps_safe=RHO,
            game=GAME,
            weights=omega,
            row_meta=em,
        )
        d_e = _fold_pub_signal(
            sf1, y, y_eq, omega, fold_idx, groups, info_by, restrict="turn"
        )
        g_core = payoff.bilinear(list(core.realization), y) - v_ref
        g_obs = payoff.bilinear(list(obs.realization), y) - v_ref
        min_safe = min(
            safety_verifier(list(r.realization), game=GAME).value for r in (core, obs)
        )
        return {
            "name": name,
            "kind": kind,
            "delta": g_obs - g_core,
            "D_e": d_e,
            "core": g_core,
            "obs": g_obs,
            "min_safety": min_safe,
            "safe": min_safe >= floor - 1e-7,
        }

    rows: list[dict[str, Any]] = []
    structured = _structured_suite(GAME, actions, eq)
    for name, beh in structured.items():
        rows.append(evaluate(name, beh, "structured"))
        print(
            f"  [structured] {name:<26} Delta={rows[-1]['delta']:+.3f} D_e={rows[-1]['D_e']:.4f}",
            flush=True,
        )

    for i in range(N_RANDOM):
        rng = random.Random(BASE_SEED + i)
        beh, prof = _random_opponent(rng, eq, actions)
        r = evaluate(f"rand{i}", beh, "random")
        r["profile"] = prof
        rows.append(r)
        if (i + 1) % 25 == 0:
            print(f"  [random] {i + 1}/{N_RANDOM}", flush=True)

    deltas = [r["delta"] for r in rows]
    d_es = [r["D_e"] for r in rows]
    dmax = max(deltas) if deltas else 0.0
    emax = max(d_es) if d_es else 0.0
    eps_d = EPS_FRAC * dmax
    eps_e = EPS_FRAC * emax

    fn_cands = [r for r in rows if r["D_e"] <= eps_e]
    fp_cands = [r for r in rows if r["delta"] <= eps_d]
    worst_fn = max(fn_cands, key=lambda r: r["delta"]) if fn_cands else None
    worst_fp = max(fp_cands, key=lambda r: r["D_e"]) if fp_cands else None
    true_pos = max(rows, key=lambda r: r["delta"])
    all_safe = all(r["safe"] for r in rows)

    print("\n  === envelope ===", flush=True)
    print(
        f"  max Delta (true positive): {true_pos['name']} Delta={true_pos['delta']:+.3f} D_e={true_pos['D_e']:.4f}",
        flush=True,
    )
    if worst_fn:
        print(
            f"  worst FN (max Delta | D_e<={eps_e:.4f}): {worst_fn['name']} Delta={worst_fn['delta']:+.3f} D_e={worst_fn['D_e']:.4f} "
            f"-> gate misses {worst_fn['delta'] / dmax * 100:.0f}% of max gap",
            flush=True,
        )
    if worst_fp:
        print(
            f"  worst FP (max D_e | Delta<={eps_d:.4f}): {worst_fp['name']} Delta={worst_fp['delta']:+.3f} D_e={worst_fp['D_e']:.4f}",
            flush=True,
        )
    print(f"  all opponents floor-safe: {all_safe}", flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps(
            {
                "game": GAME,
                "rho": RHO,
                "n_random": N_RANDOM,
                "eps_frac": EPS_FRAC,
                "max_delta": dmax,
                "max_D_e": emax,
                "eps_delta": eps_d,
                "eps_De": eps_e,
                "all_safe": all_safe,
                "true_positive": true_pos,
                "worst_false_negative": worst_fn,
                "worst_false_positive": worst_fp,
                "rows": rows,
            },
            indent=2,
        )
    )
    print(f"\nwrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
