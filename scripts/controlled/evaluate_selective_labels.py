""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from scipy.optimize import linprog

P0, P1 = 0.8, 0.2
Q0 = 0.9
QG, QB = 0.95, 0.45
Q1_TRUE = 0.5 * QG + 0.5 * QB
A_G, A_B = 1.0, 0.2
R1 = 0.5 * A_G + 0.5 * A_B
Q1_OBS = (0.5 * A_G * QG + 0.5 * A_B * QB) / R1
C = 0.65
L1 = R1 * Q1_OBS
U1 = R1 * Q1_OBS + (1.0 - R1)
U0 = P0 * (Q0 - C)
SLOPE = P1 * (C - L1)
RHO_FACE = [0.004, 0.008, 0.012, 0.016, 0.02, 0.026]
RHO_SAT = [0.05, 0.1]
EPS = 0.05
DELTA = 0.1
BASE_SEED = 2026
N_REPL = 12


def _capacity_lp(rho: float):
    ""

    res = linprog(
        c=[-1.0], A_ub=[[SLOPE]], b_ub=[rho], bounds=[(0.0, 1.0)], method="highs"
    )
    alpha = float(res.x[0])

    mu = 0.0
    if res.ineqlin is not None and len(res.ineqlin.marginals) > 0:
        mu = float(-res.ineqlin.marginals[0])
    return alpha, mu


def _simulate(rho: float, seed: int):
    ""
    rng = np.random.default_rng(seed)
    n = 200_000
    z = (rng.random(n) < P1).astype(int)
    sub_good = rng.random(n) < 0.5
    y1 = np.where(
        z == 0,
        (rng.random(n) < Q0),
        np.where(sub_good, rng.random(n) < QG, rng.random(n) < QB),
    )

    a_hist = np.where(
        z == 0, 1, np.where(sub_good, rng.random(n) < A_G, rng.random(n) < A_B)
    )
    s1 = z == 1
    passive = y1[s1 & (a_hist == 1)].mean()

    alpha, _ = _capacity_lp(rho)
    act = s1 & (rng.random(n) < alpha)
    active = y1[act].mean() if act.sum() > 0 else float("nan")
    return float(passive), float(active), int(act.sum())


def main() -> None:
    print("# selective-labels faithfulness probe (Swing 1 gate)")
    print(
        f"# q1_true={Q1_TRUE:.4f}  q1_obs(passive)={Q1_OBS:.4f}  partial-ID P=[{L1:.3f},{U1:.3f}]"
    )
    print(
        f"# U0={U0:.3f}  floor slope={SLOPE:.4f}  predicted dual mu=1/(c-L1)={1 / (C - L1):.4f}\n"
    )

    print(
        f"{'rho':>7} {'kappa1':>8} {'dual mu':>9} {'lambda1':>9} "
        f"{'N_cert':>11} {'N_cert*rho':>11}"
    )
    n_labels = math.log(2.0 / DELTA) / (2.0 * EPS**2)
    face_prod = []
    rows = []
    for rho in RHO_FACE + RHO_SAT:
        alpha, mu = _capacity_lp(rho)
        kappa1 = alpha
        lam1 = P1 * kappa1
        n_cert = n_labels / lam1 if lam1 > 1e-12 else float("inf")
        prod = n_cert * rho
        rows.append(
            {
                "rho": rho,
                "kappa1": kappa1,
                "dual": mu,
                "lambda1": lam1,
                "N_cert": n_cert,
                "N_cert_rho": prod,
            }
        )
        if rho <= SLOPE + 1e-9:
            face_prod.append(prod)
        print(
            f"{rho:>7.3f} {kappa1:>8.4f} {mu:>9.3f} {lam1:>9.5f} "
            f"{n_cert:>11.1f} {prod:>11.2f}"
        )

    pred_prod = (C - L1) * n_labels
    cv = float(np.std(face_prod) / np.mean(face_prod)) if face_prod else float("nan")
    print(
        f"\nfirst-face N_cert*rho: mean={np.mean(face_prod):.2f}  cv={cv:.2e}  "
        f"predicted={pred_prod:.2f}"
    )

    print(
        f"conditional capacity dual kappa1'(0) = {1 / SLOPE:.3f}  "
        f"(= 1/(p1(c-L1)) = {1 / (P1 * (C - L1)):.3f})"
    )

    print("\nMonte-Carlo (rho=0.026, first-face max):")
    pas, act = [], []
    for k in range(N_REPL):
        p, a, _ = _simulate(0.026, BASE_SEED + k)
        pas.append(p)
        act.append(a)
    print(
        f"  passive estimate: {np.mean(pas):.4f} +/- {np.std(pas):.4f}  "
        f"(biased estimand q1_obs={Q1_OBS:.4f}; true q1={Q1_TRUE:.4f})"
    )
    print(
        f"  active  estimate: {np.mean(act):.4f} +/- {np.std(act):.4f}  "
        f"(recovers true q1={Q1_TRUE:.4f})"
    )

    pas_ok = abs(np.mean(pas) - Q1_OBS) < 0.01 and abs(np.mean(pas) - Q1_TRUE) > 0.1
    act_ok = abs(np.mean(act) - Q1_TRUE) < 0.01
    cons_ok = cv < 1e-6

    print("\n=== GATE 1B ===")
    print(
        f"  (a) passive MNAR inconsistency: {'PASS' if pas_ok else 'FAIL'} "
        f"(passive->{np.mean(pas):.4f} != true {Q1_TRUE:.2f})"
    )
    print(
        f"  (b) safe active recovery:       {'PASS' if act_ok else 'FAIL'} "
        f"(active->{np.mean(act):.4f} = true {Q1_TRUE:.2f})"
    )
    print(
        f"  (c) conservation N_cert*rho:    {'PASS' if cons_ok else 'FAIL'} "
        f"(cv={cv:.1e} on first face)"
    )
    verdict = (
        "FAITHFUL (all three hold)"
        if (pas_ok and act_ok and cons_ok)
        else "NEEDS REVIEW"
    )
    print(f"  VERDICT: {verdict}")

    out = {
        "toy": {
            "q1_true": Q1_TRUE,
            "q1_obs": Q1_OBS,
            "L1": L1,
            "U1": U1,
            "U0": U0,
            "slope": SLOPE,
            "c": C,
            "p1": P1,
        },
        "rows": rows,
        "first_face_prod_mean": float(np.mean(face_prod)),
        "first_face_prod_cv": cv,
        "predicted_prod": pred_prod,
        "passive_mean": float(np.mean(pas)),
        "active_mean": float(np.mean(act)),
        "gate": {
            "passive_inconsistency": bool(pas_ok),
            "active_recovery": bool(act_ok),
            "conservation": bool(cons_ok),
        },
    }
    Path("results").mkdir(exist_ok=True)
    with open("results/probe_selective_labels.json", "w") as fh:
        json.dump(out, fh, indent=2)
    print("\nwrote results/probe_selective_labels.json")


if __name__ == "__main__":
    main()
