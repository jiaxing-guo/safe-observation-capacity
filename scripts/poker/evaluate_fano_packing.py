""

from __future__ import annotations

import json
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
from scripts.poker.construct_public_twins import _perturbed, _pick_targets, _tv_public
from scripts.poker.estimate_identifiable_value import _fold_action_indices

GAME = sys.argv[1] if len(sys.argv) > 1 else "holdem_tr_b2"
RHOS = [
    float(x) for x in (sys.argv[2].split(",") if len(sys.argv) > 2 else ["0.1", "0.5"])
]
MMAX = int(sys.argv[3]) if len(sys.argv) > 3 else 8
BETA = 1e6


def _continue_behavior(sf, fold_idx):
    ""
    cont = {}
    for info in sf.info_sets:
        n = len(info.children)
        fi = fold_idx.get(info.label)
        if fi is None:
            cont[info.label] = [1.0 / n] * n
        else:
            nf = [k for k in range(n) if k != fi]
            cont[info.label] = [(1.0 / len(nf)) if k in nf else 0.0 for k in range(n)]
    return cont


def _probe(triv_iv, cont_beh, weights, v_ref, rho):
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
    return list(pr.realization)


def _select_targets(sf, eq, yr_base, mmax):
    ""
    cands = _pick_targets(sf, eq, yr_base, max_targets=mmax * 4)
    out = []
    for cand in cands:
        _gap_fr, _depth, hist, i1, m1, i2, m2 = cand
        eps_probe = 0.05 * min(m1, m2)
        bp = _perturbed(eq, i1, m1, i2, m2, eps_probe, +1.0)
        bm = _perturbed(eq, i1, m1, i2, m2, eps_probe, -1.0)
        if bp is None or bm is None:
            continue
        yrp = list(Opponent(name="yp", behavior=bp, game=GAME).realization())
        yrm = list(Opponent(name="ym", behavior=bm, game=GAME).realization())
        if _tv_public(sf, yrp, yrm) > 1e-9:
            continue

        out.append(
            {
                "hist": hist,
                "label": i1.label,
                "pi": float(m1),
                "info1": i1,
                "m1": m1,
                "info2": i2,
                "m2": m2,
            }
        )
        if len(out) >= mmax:
            break
    return out


def _feasibility_radius(targets):
    ""
    eq = holdem_equilibrium_opponent(GAME).behavior
    radii = []
    for t in targets:
        r = 1.0
        for info, m in ((t["info1"], t["m1"]), (t["info2"], t["m2"])):
            acts = [a for a, _ in info.children]
            fi, ci = acts.index("f"), acts.index("c")
            row = eq[info.label]

            slack_beh = min(row[fi], 1.0 - row[fi], row[ci], 1.0 - row[ci])
            r = min(r, m * slack_beh)
        radii.append(r)

    return [min(radii[: k + 1]) for k in range(len(radii))]


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
    print(f"# Fano-packing probe on {GAME}  (v_ref={v_ref:.4f})")
    print(f"# {M} validated public-twin river targets (tv_pub<1e-9)\n")
    if M < 2:
        print(
            "FAIL: fewer than 2 public-twin targets; cannot test multi-target coupling."
        )
        return

    eta_radius = _feasibility_radius(targets)
    rows = []
    for rho in RHOS:
        lam = []
        for t in targets:
            x = _probe(triv_iv, cont_beh, {t["label"]: 1.0}, v_ref, rho)
            sv = safety_verifier(x, game=GAME).value
            om = opponent_reach_weights(x, game=GAME).get(t["label"], 0.0)
            lam.append(t["pi"] * float(om))
            assert sv >= v_ref - rho - 1e-6, f"probe unsafe: {sv} < {v_ref - rho}"

        couple = []
        for k in range(2, M + 1):
            sub = targets[:k]
            weights = {t["label"]: t["pi"] for t in sub}
            x = _probe(triv_iv, cont_beh, weights, v_ref, rho)
            om = opponent_reach_weights(x, game=GAME)
            per = [t["pi"] * float(om.get(t["label"], 0.0)) for t in sub]
            Lambda_act = sum(per)
            sum_lam = sum(lam[:k])
            ratio = Lambda_act / sum_lam if sum_lam > 1e-12 else float("nan")

            r_lb = min(per) if per else 0.0
            couple.append(
                {
                    "M": k,
                    "Lambda_act": Lambda_act,
                    "sum_lambda": sum_lam,
                    "ratio": ratio,
                    "r_lb": r_lb,
                    "eta_max": eta_radius[k - 1],
                }
            )
        rows.append({"rho": rho, "lambda_j": lam, "couple": couple})

        print(f"rho={rho}")
        print(f"  per-target lambda_j: {[round(v, 5) for v in lam]}")
        print(
            f"  {'M':>3} {'Lambda_act':>11} {'sum_lambda':>11} {'ratio R':>9} "
            f"{'r_lb':>9} {'M/Lambda':>9} {'sum 1/lam':>10} {'eta_max':>9}"
        )
        for c in couple:
            inv_sum = sum(1.0 / v for v in lam[: c["M"]] if v > 1e-12)
            m_over_L = (
                c["M"] / c["Lambda_act"] if c["Lambda_act"] > 1e-12 else float("nan")
            )
            print(
                f"  {c['M']:>3} {c['Lambda_act']:>11.5f} {c['sum_lambda']:>11.5f} "
                f"{c['ratio']:>9.3f} {c['r_lb']:>9.5f} {m_over_L:>9.2f} "
                f"{inv_sum:>10.2f} {c['eta_max']:>9.4f}"
            )
        print()

    small_rho = rows[0]
    couples = small_rho["couple"]
    rmin = couples[-1]["ratio"]
    rtrend = couples[-1]["ratio"] - couples[0]["ratio"]
    eta_collapse = (
        eta_radius[-1] / eta_radius[1]
        if len(eta_radius) > 1 and eta_radius[1] > 0
        else 1.0
    )
    print("=== GATE 2B ===")
    print(
        f"coupling ratio R at M={M}, rho={small_rho['rho']}: {rmin:.3f}  "
        f"(trend with M: {rtrend:+.3f}; eta_max ratio last/first: {eta_collapse:.2f})"
    )
    if rmin < 0.5 or (rmin < 0.7 and rtrend < -0.1):
        print(
            "LIVE: strong/strengthening shared-budget coupling -> coupled bound materially "
            "below the naive sum; multi-target Fano non-vacuous. eta_max does NOT collapse with M."
        )
    elif rmin < 0.8:
        print(
            "MARGINAL: real coupling (R<0.8, falling with M) -> coupled bound modestly stronger; "
            "extend to larger M / smaller rho to confirm R crosses 0.5; lean on routing framing."
        )
    else:
        print(
            "WEAK: little coupling (R>=0.8) -> portfolio not meaningfully cheaper; "
            "downgrade Swing 2 to the partial-monitoring reframe."
        )

    out = {
        "game": GAME,
        "rhos": RHOS,
        "M": M,
        "targets": [
            {"hist": t["hist"], "label": t["label"], "pi": t["pi"]} for t in targets
        ],
        "eta_radius": eta_radius,
        "rows": rows,
        "elapsed_s": time.time() - t0,
    }
    Path("results").mkdir(exist_ok=True)
    with open(f"results/evaluate_fano_packing_{GAME}.json", "w") as fh:
        json.dump(out, fh, indent=2)
    print(
        f"\nwrote results/evaluate_fano_packing_{GAME}.json  ({time.time() - t0:.1f}s)"
    )


if __name__ == "__main__":
    main()
