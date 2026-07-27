"""Validate capacity tradeoff. See supplementary Additional Experiments."""

from __future__ import annotations

import json
import sys

from safe_observation.opponents import Opponent
from safe_observation.sequence_form import compile as compile_game
from safe_observation.solvers import (
    opponent_reach_weights,
    robust_safe_response_probe,
    safety_verifier,
    solve_blueprint,
)

GAME = sys.argv[1] if len(sys.argv) > 1 else "kuhn"
RHO_GRID = [0.0, 0.01, 0.02, 0.04, 0.08, 0.12, 0.2, 0.3, 0.45, 0.6, 0.8, 1.0]


def _targets(sf1):
    """Compute targets for the validate capacity tradeoff workflow."""
    out = []
    for info in sf1.info_sets:
        acts = [a for a, _ in info.children]
        if ("p" in acts or "f" in acts) and ("b" in acts or "c" in acts):
            out.append(info.label)
    return out


def _kappa(sf1, target, rho, v_ref):
    """Solve safe observation capacity at the supplied safety budget."""
    triv_iv = {i.label: [(0.0, 1.0)] * len(i.children) for i in sf1.info_sets}
    cont_beh = {}
    for info in sf1.info_sets:
        acts = [a for a, _ in info.children]
        if info.label == target and ("b" in acts or "c" in acts):
            ci = acts.index("b" if "b" in acts else "c")
            row = [0.0] * len(acts)
            row[ci] = 1.0
            cont_beh[info.label] = row
        else:
            cont_beh[info.label] = [1.0 / len(acts)] * len(acts)
    probe = robust_safe_response_probe(
        triv_iv,
        cont_beh,
        {target: 1.0},
        v_ref=v_ref,
        eps_safe=rho,
        beta=1e6,
        rho=0.0,
        game=GAME,
    )
    xpr = list(probe.realization)
    w = opponent_reach_weights(xpr, game=GAME).get(target, 0.0)
    safe = safety_verifier(xpr, game=GAME).value
    return w, safe


def main() -> None:
    """Run the command-line entry point."""
    sf1 = compile_game(GAME, 1)
    bp = solve_blueprint(GAME, method="lp")
    v_ref = bp.value
    cands = _targets(sf1)
    print(f"# V1b: safety-information tradeoff law on {GAME}  (v_ref={v_ref:.4f})")
    print(f"# targets facing a bet: {cands}\n")

    results = {}
    for target in cands:
        curve = []
        for rho in RHO_GRID:
            w, safe = _kappa(sf1, target, rho, v_ref)
            curve.append({"rho": rho, "kappa": w, "safety": safe})
        results[target] = curve
        ceil = max(c["kappa"] for c in curve)

        slopes = []
        for i in range(1, len(curve)):
            dk = curve[i]["kappa"] - curve[i - 1]["kappa"]
            dr = curve[i]["rho"] - curve[i - 1]["rho"]
            if dr > 0:
                slopes.append(dk / dr)

        k0 = curve[0]["kappa"]
        kind = "WALL" if k0 < 1e-6 else "free"

        sat = next(
            (c["rho"] for c in curve if c["kappa"] >= 0.98 * ceil and ceil > 0), None
        )

        init_slope = slopes[0] if slopes else 0.0
        L_imp = ceil / init_slope if init_slope > 1e-9 else float("inf")
        print(
            f"== {target}  [{kind}]  ceiling(reach)={ceil:.4f}  saturates@rho~{sat}  "
            f"init_slope={init_slope:.3f}  implied L_I={L_imp:.3f}"
        )
        print(f"   {'rho':>6}{'kappa':>10}{'kappa/ceil':>12}{'safety':>10}")
        for c in curve:
            frac = c["kappa"] / ceil if ceil > 0 else 0.0
            print(
                f"   {c['rho']:>6.2f}{c['kappa']:>10.5f}{frac:>12.3f}{c['safety']:>10.4f}"
            )
        print()

    out = {"game": GAME, "v_ref": v_ref, "rho_grid": RHO_GRID, "curves": results}
    with open(f"results/spike_tradeoff_law_{GAME}.json", "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"# wrote results/spike_tradeoff_law_{GAME}.json")
    _ = Opponent


if __name__ == "__main__":
    main()
