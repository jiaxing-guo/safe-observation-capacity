""

from __future__ import annotations

import time

from safe_observation.opponents import Opponent, holdem_equilibrium_opponent
from safe_observation.sequence_form import compile as compile_game
from safe_observation.solvers import (
    opponent_reach_weights,
    robust_safe_response_probe,
    safety_verifier,
    solve_blueprint,
)
from scripts.poker.estimate_identifiable_value import _fold_action_indices
from scripts.poker.evaluate_public_routing import _build_gate_suite

GAME = "holdem_tr_b2"
LEAK = "turn_overfold_w70"
RHO_GRID = [0.0, 0.05, 0.1, 0.2]
TOPK = 3
BETA = 1e6


def _kappa_curve(target_labels, v_ref, triv_iv, cont_beh):
    ""
    curves: dict[str, list[float]] = {lbl: [] for lbl in target_labels}
    for rho in RHO_GRID:
        for lbl in target_labels:
            pr = robust_safe_response_probe(
                triv_iv,
                cont_beh,
                {lbl: 1.0},
                v_ref=v_ref,
                eps_safe=rho,
                beta=BETA,
                rho=0.0,
                game=GAME,
            )
            x = list(pr.realization)
            sv = safety_verifier(x, game=GAME).value
            if sv < v_ref - rho - 1e-6:
                curves[lbl].append(0.0)
                continue
            om = opponent_reach_weights(x, game=GAME)
            curves[lbl].append(float(om.get(lbl, 0.0)))
    return curves


def _summarize(curves):
    ""
    rows = []
    for lbl, k in curves.items():
        k0 = k[0]
        mu = (k[1] - k0) / RHO_GRID[1] if RHO_GRID[1] > 0 else float("nan")

        slopes = [
            (k[i + 1] - k[i]) / (RHO_GRID[i + 1] - RHO_GRID[i])
            for i in range(len(k) - 1)
        ]
        concave = all(slopes[i] >= slopes[i + 1] - 1e-4 for i in range(len(slopes) - 1))
        rows.append((lbl, k0, mu, max(k), concave))
    return rows


def main() -> None:
    sf1 = compile_game(GAME, 1)
    fold_idx = _fold_action_indices(sf1)
    triv_iv = {i.label: [(0.0, 1.0)] * len(i.children) for i in sf1.info_sets}
    cont_beh = {}
    for info in sf1.info_sets:
        n = len(info.children)
        fi = fold_idx.get(info.label)
        if fi is None:
            cont_beh[info.label] = [1.0 / n] * n
        else:
            nf = [k for k in range(n) if k != fi]
            cont_beh[info.label] = [
                (1.0 / len(nf)) if k in nf else 0.0 for k in range(n)
            ]

    eqb = holdem_equilibrium_opponent(GAME).behavior
    actions = {i.label: [a for a, _ in i.children] for i in sf1.info_sets}
    suite = _build_gate_suite(GAME, actions, eqb)
    y_eq = list(Opponent(name="eq", behavior=eqb, game=GAME).realization())
    y_star = list(Opponent(name=LEAK, behavior=suite[LEAK], game=GAME).realization())

    def is_river(label: str) -> bool:
        return "/" in (label.split("|", 1)[1] if "|" in label else "")

    carriers = []
    for info in sf1.info_sets:
        if not is_river(info.label):
            continue
        fi = fold_idx.get(info.label)
        dev = sum(
            abs(y_star[child] - y_eq[child])
            for a, (_ac, child) in enumerate(info.children)
            if not (fi is not None and a == fi)
        )
        if dev > 1e-6:
            carriers.append((info.label, dev))
    carriers.sort(key=lambda t: t[1], reverse=True)
    target_labels = [t[0] for t in carriers[:TOPK]]
    print(
        f"# targets: {len(target_labels)} value-carrying river infosets of leak {LEAK}\n"
    )

    blueprints = {}
    t0 = time.time()
    bp_lp = solve_blueprint(GAME, method="lp")
    blueprints["LP (exact)"] = bp_lp.value
    print(f"LP blueprint   v_ref={bp_lp.value:+.5f}  ({time.time() - t0:.1f}s)")
    for it in [100_000, 500_000]:
        t0 = time.time()
        bp = solve_blueprint(GAME, method="mccfr", iterations=it, seed=2026)
        blueprints[f"MCCFR {it // 1000}k"] = bp.value
        gap = abs(bp.value - bp_lp.value)
        print(
            f"MCCFR {it // 1000}k    v_ref={bp.value:+.5f}  gap-to-LP={gap:.4f}  ({time.time() - t0:.1f}s)"
        )

    print(
        f"\n{'blueprint':<14}{'target':<26}{'kappa_0':>9}{'mu_I':>9}{'kappa_max':>10}{'concave':>9}"
    )
    for name, v_ref in blueprints.items():
        curves = _kappa_curve(target_labels, v_ref, triv_iv, cont_beh)
        for lbl, k0, mu, kmax, concave in _summarize(curves):
            short = lbl.split("|", 1)[-1][:24]
            print(
                f"{name:<14}{short:<26}{k0:>9.4f}{mu:>9.3f}{kmax:>10.4f}"
                f"{'YES' if concave else 'NO':>9}"
            )
        print()


if __name__ == "__main__":
    main()
