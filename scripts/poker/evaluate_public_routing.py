"""Evaluate the public routing experiment. See Experiments and supplementary Certification at the Unbucketed River."""

from collections.abc import Callable
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
from scripts.poker.run_turn_river_methods import _perturb, _top_rank

GAME = os.environ.get("ROUTING_GAME", "holdem_tr_b2")
RHO = float(os.environ.get("ROUTING_RHO", "0.5"))
OUT = Path(os.environ.get("PUBLIC_ROUTING_OUT", f"results/public_routing_{GAME}.json"))


def _river_fold_pick(actions: dict[str, list[str]]) -> Callable:
    """Compute river fold pick for the evaluate public routing workflow."""

    def pick(hole, hist, acts):
        """Compute pick for the evaluate public routing workflow."""
        return "/" in hist and "f" in acts

    return pick


def _turn_fold_pick(actions: dict[str, list[str]]) -> Callable:
    """Compute turn fold pick for the evaluate public routing workflow."""

    def pick(hole, hist, acts):
        """Compute pick for the evaluate public routing workflow."""
        return "/" not in hist and "f" in acts

    return pick


def _build_gate_suite(
    game: str, actions: dict[str, list[str]], eq: dict[str, list[float]]
) -> dict[str, dict[str, float]]:
    """Build gate suite for the evaluate public routing workflow."""
    river_pick = _river_fold_pick(actions)
    turn_pick = _turn_fold_pick(actions)
    suite: dict[str, dict[str, float]] = {"tr_equilibrium": eq}

    for w in (0.05, 0.2, 0.35, 0.5, 0.65, 0.8):
        suite[f"river_overfold_w{int(w * 100):02d}"] = _perturb(
            eq, actions, river_pick, "f", w
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
    for w in (0.3, 0.5, 0.7):
        suite[f"turn_overfold_w{int(w * 100):02d}"] = _perturb(
            eq, actions, turn_pick, "f", w
        )
    return suite


def _diagnostic_D(
    sf1,
    y_star: list[float],
    y_eq: list[float],
    omega: dict[str, float],
    fold_idx: dict[str, int],
) -> dict[str, float]:
    """Compute diagnostic d for the evaluate public routing workflow."""
    revealed = 0.0
    turn = 0.0
    river = 0.0
    fold = 0.0
    for info in sf1.info_sets:
        w = omega.get(info.label, 0.0)
        if w <= 0.0:
            continue
        fi = fold_idx.get(info.label)
        hist = info.label.split("|", 1)[1] if "|" in info.label else ""
        is_river = "/" in hist
        for a_idx, (_action, child) in enumerate(info.children):
            dev = w * abs(y_star[child] - y_eq[child])
            if fi is not None and a_idx == fi:
                fold += dev
                continue
            revealed += dev
            if is_river:
                river += dev
            else:
                turn += dev
    return {
        "revealed": revealed,
        "revealed_turn": turn,
        "revealed_river": river,
        "fold": fold,
    }


def _pearson(xs: list[float], ys: list[float]) -> float:
    """Compute the Pearson correlation between two score vectors."""
    n = len(xs)
    if n < 2:
        return 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True))
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    if sxx <= 1e-15 or syy <= 1e-15:
        return 0.0
    return sxy / (sxx**0.5 * syy**0.5)


def _spearman(xs: list[float], ys: list[float]) -> float:
    """Compute Spearman rank correlation between two score vectors."""

    def ranks(v: list[float]) -> list[float]:
        """Assign average ranks while preserving tied values."""
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r

    return _pearson(ranks(xs), ranks(ys))


def main() -> None:
    """Run the command-line entry point."""
    game = GAME
    print(
        f"# C_obs GATE test (Q4, cheap diagnostic vs obs-core)  game={game}  rho={RHO}",
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
    y_eq = list(Opponent(name="eq", behavior=eq, game=game).realization())
    suite = _build_gate_suite(game, actions, eq)

    print(
        f"  {'opponent':<24} {'D_rev':>9} {'D_river':>9} {'D_fold':>9} "
        f"{'core':>8} {'obs':>8} {'oracle':>8} {'obs-core':>9} {'min_safe':>9}",
        flush=True,
    )
    rows: list[dict[str, Any]] = []
    for name, behavior in suite.items():
        y_star = list(Opponent(name=name, behavior=behavior, game=game).realization())
        pub_intervals = _population_public_intervals(
            groups, info_by_label, y_star, omega
        )
        obs_pub = _population_public_intervals(
            groups, info_by_label, y_star, omega, fold_only=True, fold_idx=fold_idx
        )
        d = _diagnostic_D(sf1, y_star, y_eq, omega, fold_idx)

        core = robust_safe_response_public(
            groups, pub_intervals, v_ref=v_ref, eps_safe=RHO, game=game, weights=omega
        )
        entries, h, meta = _population_event_constraints(sf1, y_star, omega, fold_idx)
        obs = robust_safe_response_linear(
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
            safety_verifier(list(r.realization), game=game).value
            for r in (core, obs, oracle)
        )
        rows.append(
            {
                "opponent": name,
                "D": d["revealed"],
                "D_turn": d["revealed_turn"],
                "D_river": d["revealed_river"],
                "D_fold": d["fold"],
                "core": core_gain,
                "obs": obs_gain,
                "oracle": oracle_gain,
                "obs_minus_core": obs_gain - core_gain,
                "min_safety": min_safety,
            }
        )
        print(
            f"  {name:<24} {d['revealed']:>9.4f} {d['revealed_river']:>9.4f} "
            f"{d['fold']:>9.4f} {core_gain:>+8.3f} {obs_gain:>+8.3f} "
            f"{oracle_gain:>+8.3f} {obs_gain - core_gain:>+9.3f} {min_safety:>+9.3f}",
            flush=True,
        )

    ds = [r["D"] for r in rows]
    lifts = [r["obs_minus_core"] for r in rows]
    pear = _pearson(ds, lifts)
    spear = _spearman(ds, lifts)

    available = sum(lifts)
    candidate_taus = sorted({0.0, *ds})
    gate_curve: list[dict[str, Any]] = []
    for tau in candidate_taus:
        engaged = [r for r in rows if r["D"] > tau]
        captured = sum(r["obs_minus_core"] for r in engaged)
        gate_curve.append(
            {
                "tau": tau,
                "solve_rate": len(engaged) / len(rows),
                "n_engaged": len(engaged),
                "captured": captured,
                "captured_frac": captured / available if abs(available) > 1e-9 else 0.0,
            }
        )

    nonzero_ds = sorted(d for d in ds if d > 1e-9)
    tau_star = nonzero_ds[0] * 0.5 if nonzero_ds else 0.0
    skipped = [r for r in rows if r["D"] <= tau_star and r["obs_minus_core"] > 0.02]

    print(
        f"\n  pearson(D, obs-core)={pear:+.3f}  spearman={spear:+.3f}  "
        f"available_lift={available:+.3f}",
        flush=True,
    )
    print("  gate tradeoff (tau, solve_rate, captured_frac):", flush=True)
    for pt in gate_curve:
        print(
            f"    tau={pt['tau']:>8.4f}  solve_rate={pt['solve_rate']:.2f}  "
            f"captured_frac={pt['captured_frac']:+.3f}  (n={pt['n_engaged']})",
            flush=True,
        )
    if skipped:
        print(
            f"  WARNING false negatives at tau={tau_star:.4f} (D small, lift>0.02): "
            + ", ".join(r["opponent"] for r in skipped),
            flush=True,
        )
    else:
        print(f"  no false negatives at tau={tau_star:.4f}", flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps(
            {
                "game": game,
                "rho": RHO,
                "v_ref": v_ref,
                "pearson": pear,
                "spearman": spear,
                "available_lift": available,
                "rows": rows,
                "gate_curve": gate_curve,
            },
            indent=2,
        )
    )
    print(f"\nwrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
