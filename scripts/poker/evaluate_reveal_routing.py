""

import json
import os
from pathlib import Path
from typing import Any

from safe_observation.confidence import OpponentEvidenceStore
from safe_observation.opponents import Opponent, holdem_equilibrium_opponent
from safe_observation.sequence_form import compile as compile_game
from safe_observation.solvers import opponent_reach_weights, solve_blueprint
from scripts.poker.estimate_identifiable_value import _fold_action_indices
from scripts.poker.evaluate_public_routing import _build_gate_suite, _pearson, _spearman

GAME = os.environ.get("ROUTING_GAME", "holdem_tr_b2")
RHO = float(os.environ.get("ROUTING_RHO", "0.5"))
GATE_JSON = Path(
    os.environ.get("PUBLIC_ROUTING_INPUT", f"results/public_routing_{GAME}.json")
)
OUT = Path(os.environ.get("REVEAL_ROUTING_OUT", f"results/reveal_routing_{GAME}.json"))


def _is_river(label: str) -> bool:
    return "/" in (label.split("|", 1)[1] if "|" in label else "")


def _behav_signal(
    sf1, behavior, eq, omega, fold_idx, restrict: str | None = None
) -> float:
    ""
    total = 0.0
    for info in sf1.info_sets:
        w = omega.get(info.label, 0.0)
        if w <= 0.0:
            continue
        b = behavior.get(info.label)
        e = eq.get(info.label)
        if b is None or e is None:
            continue
        fi = fold_idx.get(info.label)
        river = _is_river(info.label)
        if restrict == "turn" and river:
            continue
        if restrict == "river" and not river:
            continue
        s = 0.0
        for a in range(len(b)):
            if restrict == "fold" and a != fi:
                continue
            if restrict == "nonfold" and a == fi:
                continue
            s += abs(b[a] - e[a])
        total += w * 0.5 * s
    return total


def _fold_pub_signal(
    sf1,
    y_star,
    y_eq,
    omega,
    fold_idx,
    groups,
    info_by_label,
    restrict: str | None = None,
) -> float:
    ""
    total = 0.0
    for _key, members in groups.items():
        if restrict == "turn" and _is_river(members[0]):
            continue
        if restrict == "river" and not _is_river(members[0]):
            continue
        fi = next((fold_idx[m] for m in members if m in fold_idx), None)
        if fi is None:
            continue
        num_star = den_star = num_eq = den_eq = 0.0
        for m in members:
            info = info_by_label[m]
            w = omega.get(m, 0.0)
            if w <= 0.0 or fi >= len(info.children):
                continue
            fold_child = info.children[fi][1]
            num_star += w * y_star[fold_child]
            den_star += w * y_star[info.parent_seq]
            num_eq += w * y_eq[fold_child]
            den_eq += w * y_eq[info.parent_seq]
        pf_star = num_star / den_star if den_star > 1e-12 else 0.0
        pf_eq = num_eq / den_eq if den_eq > 1e-12 else 0.0
        total += den_eq * abs(pf_star - pf_eq)
    return total


def _gate_curve(signal: list[float], lift: list[float]) -> list[dict[str, Any]]:
    available = sum(lift)
    curve: list[dict[str, Any]] = []
    for tau in sorted({0.0, *signal}):
        engaged = [i for i, s in enumerate(signal) if s > tau]
        captured = sum(lift[i] for i in engaged)
        curve.append(
            {
                "tau": tau,
                "solve_rate": len(engaged) / len(signal),
                "captured_frac": captured / available if abs(available) > 1e-9 else 0.0,
            }
        )
    return curve


def _knee(curve: list[dict[str, Any]], target: float = 0.9) -> dict[str, Any] | None:
    ""
    feasible = [pt for pt in curve if pt["captured_frac"] >= target]
    return min(feasible, key=lambda pt: pt["solve_rate"]) if feasible else None


def main() -> None:
    game = GAME
    print(
        f"# C_obs GATE-2 test (Q4b, fold/value-aware signal panel)  game={game}  rho={RHO}",
        flush=True,
    )
    if not GATE_JSON.exists():
        raise SystemExit(
            f"missing {GATE_JSON}; run python -m scripts.poker.evaluate_public_routing first"
        )
    gate = json.loads(GATE_JSON.read_text())
    lift_by_name = {r["opponent"]: r["obs_minus_core"] for r in gate["rows"]}
    core_by_name = {r["opponent"]: r["core"] for r in gate["rows"]}

    sf1 = compile_game(game, 1)
    info_by_label = {info.label: info for info in sf1.info_sets}
    bp = solve_blueprint(game, method="lp")
    assert bp.realization is not None
    omega = opponent_reach_weights(bp.realization, game=game)
    fold_idx = _fold_action_indices(sf1)
    actions = {info.label: [a for a, _ in info.children] for info in sf1.info_sets}
    groups = OpponentEvidenceStore.for_game(game).public_groups()
    eq = holdem_equilibrium_opponent(game).behavior
    y_eq = list(Opponent(name="eq", behavior=eq, game=game).realization())
    suite = _build_gate_suite(game, actions, eq)

    signal_names = [
        "rev_real",
        "behav_tv",
        "behav_fold",
        "behav_turn",
        "behav_river",
        "fold_pub",
        "fold_pub_turn",
        "fold_pub_river",
        "neg_core",
    ]
    rows: list[dict[str, Any]] = []
    for name, behavior in suite.items():
        if name not in lift_by_name:
            continue
        y_star = list(Opponent(name=name, behavior=behavior, game=game).realization())
        sig = {
            "rev_real": _behav_signal(sf1, behavior, eq, omega, fold_idx, "nonfold"),
            "behav_tv": _behav_signal(sf1, behavior, eq, omega, fold_idx, None),
            "behav_fold": _behav_signal(sf1, behavior, eq, omega, fold_idx, "fold"),
            "behav_turn": _behav_signal(sf1, behavior, eq, omega, fold_idx, "turn"),
            "behav_river": _behav_signal(sf1, behavior, eq, omega, fold_idx, "river"),
            "fold_pub": _fold_pub_signal(
                sf1, y_star, y_eq, omega, fold_idx, groups, info_by_label
            ),
            "fold_pub_turn": _fold_pub_signal(
                sf1, y_star, y_eq, omega, fold_idx, groups, info_by_label, "turn"
            ),
            "fold_pub_river": _fold_pub_signal(
                sf1, y_star, y_eq, omega, fold_idx, groups, info_by_label, "river"
            ),
            "neg_core": -core_by_name[name],
        }
        rows.append({"opponent": name, "lift": lift_by_name[name], "signals": sig})

    lifts = [r["lift"] for r in rows]

    panel: dict[str, dict[str, Any]] = {}
    for sname in signal_names:
        xs = [r["signals"][sname] for r in rows]
        curve = _gate_curve(xs, lifts)
        panel[sname] = {
            "pearson": _pearson(xs, lifts),
            "spearman": _spearman(xs, lifts),
            "knee90": _knee(curve, 0.9),
            "gate_curve": curve,
        }

    print(f"\n  {len(rows)} opponents, available_lift={sum(lifts):+.3f}", flush=True)
    print(
        f"  {'signal':<16} {'pearson':>9} {'spearman':>9} {'knee90_solve_rate':>18}",
        flush=True,
    )
    ranked = sorted(panel.items(), key=lambda kv: kv[1]["spearman"], reverse=True)
    for sname, info in ranked:
        knee = info["knee90"]
        knee_str = f"{knee['solve_rate']:.2f}" if knee else "n/a"
        print(
            f"  {sname:<16} {info['pearson']:>+9.3f} {info['spearman']:>+9.3f} "
            f"{knee_str:>18}",
            flush=True,
        )

    best = ranked[0][0]
    print(f"\n  best by spearman: {best}", flush=True)
    print(f"  {'opponent':<24} {'lift':>8} {best:>12}", flush=True)
    for r in sorted(rows, key=lambda r: r["signals"][best], reverse=True):
        print(
            f"  {r['opponent']:<24} {r['lift']:>+8.3f} {r['signals'][best]:>12.4f}",
            flush=True,
        )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps(
            {
                "game": game,
                "rho": RHO,
                "available_lift": sum(lifts),
                "best_signal": best,
                "panel": {
                    k: {kk: vv for kk, vv in v.items() if kk != "gate_curve"}
                    | {"gate_curve": v["gate_curve"]}
                    for k, v in panel.items()
                },
                "rows": rows,
            },
            indent=2,
        )
    )
    print(f"\nwrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
