""

from __future__ import annotations

import json
import math
from pathlib import Path
import sys
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
from scripts.poker.evaluate_fano_packing import _continue_behavior, _select_targets

GAME = sys.argv[1] if len(sys.argv) > 1 else "holdem_tr_b2"
RHOS = [
    float(x) for x in (sys.argv[2].split(",") if len(sys.argv) > 2 else ["0.1", "0.5"])
]
MMAX = int(sys.argv[3]) if len(sys.argv) > 3 else 8
ITERS = int(sys.argv[4]) if len(sys.argv) > 4 else 14
BETA = 1e6


def _reach_vector(triv_iv, cont_beh, targets, weights, v_ref, rho):
    ""
    pr = robust_safe_response_probe(
        triv_iv,
        cont_beh,
        weights,
        v_ref=v_ref,
        eps_safe=rho,
        beta=BETA,
        rho=0.0,
        game=GAME,
    )
    x = list(pr.realization)
    sv = safety_verifier(x, game=GAME).value
    assert sv >= v_ref - rho - 1e-6, f"probe unsafe: {sv} < {v_ref - rho}"
    om = opponent_reach_weights(x, game=GAME)
    f = [t["pi"] * float(om.get(t["label"], 0.0)) for t in targets]
    return x, f


def _rstar(triv_iv, cont_beh, targets, lam, v_ref, rho, iters):
    ""
    M = len(targets)
    fscale = max(lam) if max(lam) > 1e-12 else 1.0
    eta = math.sqrt(8.0 * math.log(M) / iters) if M > 1 else 0.0
    w = [1.0 / M] * M
    acc = [0.0] * M
    ub = float("inf")
    trace = []
    for t in range(iters):
        weights = {targets[j]["label"]: w[j] * targets[j]["pi"] for j in range(M)}
        _x, f = _reach_vector(triv_iv, cont_beh, targets, weights, v_ref, rho)
        g_w = sum(w[j] * f[j] for j in range(M))
        ub = min(ub, g_w)
        for j in range(M):
            acc[j] += f[j]
        lb = min(a / (t + 1) for a in acc)
        trace.append({"iter": t, "g_w": g_w, "ub": ub, "lb": lb, "min_f": min(f)})

        m = [math.exp(-eta * min(1.0, f[j] / fscale)) for j in range(M)]
        w = [w[j] * m[j] for j in range(M)]
        z = sum(w)
        w = [wi / z for wi in w]
    xbar_reach = [a / iters for a in acc]
    return {
        "rho": rho,
        "M": M,
        "rstar_lb": min(xbar_reach),
        "rstar_ub": ub,
        "xbar_reach": xbar_reach,
        "lambda_j": lam,
        "min_lambda": min(lam),
        "mean_lambda": sum(lam) / M,
        "trace": trace,
    }


def main() -> None:
    t0 = time.time()
    sf = compile_game(GAME, 1)
    fold_idx = _fold_action_indices(sf)
    v_ref = solve_blueprint(GAME, method="lp").value
    eq = holdem_equilibrium_opponent(GAME).behavior
    yr_base = list(Opponent(name="eq", behavior=eq, game=GAME).realization())
    triv_iv = {i.label: [(0.0, 1.0)] * len(i.children) for i in sf.info_sets}
    cont_beh = _continue_behavior(sf, fold_idx)
    targets = _select_targets(sf, eq, yr_base, MMAX)
    M = len(targets)
    print(
        f"# r* balanced-cover probe on {GAME}  (v_ref={v_ref:.4f}), M={M} targets, iters={ITERS}\n"
    )
    if M < 2:
        print("FAIL: <2 targets.")
        return

    out_rows = []
    for rho in RHOS:
        lam = []
        for t in targets:
            _x, f = _reach_vector(
                triv_iv, cont_beh, targets, {t["label"]: t["pi"]}, v_ref, rho
            )
            lam.append(max(f))
        res = _rstar(triv_iv, cont_beh, targets, lam, v_ref, rho, ITERS)
        out_rows.append(res)

        rs_lb, rs_ub = res["rstar_lb"], res["rstar_ub"]
        print(f"rho={rho}")
        print(f"  lambda_j (solo): {[round(v, 5) for v in lam]}")
        print(
            f"  min lambda_j = {res['min_lambda']:.5f}   mean lambda_j = {res['mean_lambda']:.5f}"
        )
        print(f"  r* in [{rs_lb:.5f}, {rs_ub:.5f}]   (gap {rs_ub - rs_lb:.5f})")
        print(
            f"  balanced plan per-target reach: {[round(v, 5) for v in res['xbar_reach']]}"
        )
        denom = res["min_lambda"] if res["min_lambda"] > 1e-12 else float("nan")
        print(
            f"  r*/min_lambda = {rs_lb / denom:.3f}  (1 => no competition; 0 => severe)"
        )
        print(
            f"  M*r* (balanced total) = {M * rs_lb:.5f}  vs Lambda_act-equivalent sum"
        )
        print(
            f"  routing: certify-all-M needs N >= ~1/(r* eps^2);  r*_lb={rs_lb:.5f}\n"
        )

    verdict_rho = out_rows[0]
    rs = verdict_rho["rstar_lb"]
    ratio = rs / verdict_rho["min_lambda"] if verdict_rho["min_lambda"] > 1e-12 else 0.0
    print("=== r* VERDICT ===")
    if rs > 1e-4 and ratio > 0.25:
        print(
            f"CLEAN: r*={rs:.5f} > 0 and r*/min_lambda={ratio:.2f} -- a single floor-safe plan covers "
            f"ALL {M} targets at a positive balanced rate; the routing-optimality corollary COMPLETES "
            f"(certify-all-M cost ~1/(r* eps^2), competition factor {ratio:.2f})."
        )
    elif rs > 1e-4:
        print(
            f"PARTIAL: r*={rs:.5f} > 0 but r*/min_lambda={ratio:.2f} small -- balanced coverage is "
            f"feasible but expensive (severe competition); routing corollary holds with a large constant."
        )
    else:
        print(
            f"COLLAPSE: r*~0 -- no single safe plan reaches all {M} targets; certify-all-M is "
            f"sequential. Fall back to the portfolio-sum bound + partial-monitoring reframe."
        )

    Path("results").mkdir(exist_ok=True)
    with open(f"results/probe_fano_rstar_{GAME}.json", "w") as fh:
        json.dump(
            {
                "game": GAME,
                "rhos": RHOS,
                "M": M,
                "iters": ITERS,
                "rows": out_rows,
                "targets": [
                    {"hist": t["hist"], "label": t["label"], "pi": t["pi"]}
                    for t in targets
                ],
                "elapsed_s": time.time() - t0,
            },
            fh,
            indent=2,
        )
    print(f"\nwrote results/probe_fano_rstar_{GAME}.json  ({time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main()
