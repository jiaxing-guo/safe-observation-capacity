"""Evaluate the decoupled observation experiment. See supplementary Additional Experiments."""

from __future__ import annotations

import json

import numpy as np
from scipy.optimize import linprog

MU0 = 0.0
SIGMA = 1.0
DELTA = 0.1
RHOS = [0.0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.45, 0.6, 0.8, 1.0]
L_I = 0.5


def _max_coord_under_floor(rewards, idx, rho):
    """Compute max coord under floor."""
    n = len(rewards)
    c = np.zeros(n)
    c[idx] = -1.0
    A_ub = (-np.asarray(rewards)).reshape(1, n)
    b_ub = np.array([-(MU0 - rho)])
    A_eq = np.ones((1, n))
    b_eq = np.array([1.0])
    res = linprog(
        c,
        A_ub=A_ub,
        b_ub=b_ub,
        A_eq=A_eq,
        b_eq=b_eq,
        bounds=[(0.0, 1.0)] * n,
        method="highs",
    )
    if not res.success:
        return 0.0, 0.0
    val = float(res.x[idx])
    try:
        dual = -float(res.ineqlin.marginals[0])
    except Exception:
        dual = 0.0
    return val, dual


def run_case(L_rev: float):
    """Run the case experiment for the evaluate decoupled observation workflow."""
    rewards = [MU0, MU0 - L_I, MU0 - L_rev]
    idx_reach, idx_rev = 1, 2
    I_R = 1.0 / SIGMA**2
    rows = []
    for rho in RHOS:
        kap, mu_reach = _max_coord_under_floor(rewards, idx_reach, rho)
        rev_cap, mu_info = _max_coord_under_floor(rewards, idx_rev, rho)

        n_cert = (SIGMA**2 / (DELTA**2 * rev_cap)) if rev_cap > 1e-12 else float("inf")
        rows.append(
            {
                "rho": rho,
                "kappa_reach": kap,
                "mu_reach": mu_reach,
                "reveal_cap": rev_cap,
                "mu_info": mu_info,
                "n_cert": n_cert,
                "n_rho": (n_cert * rho if np.isfinite(n_cert) else float("nan")),
            }
        )

    lin = [
        r
        for r in rows
        if 0 < r["rho"] < min(L_I, L_rev if L_rev > 0 else 1e9)
        and np.isfinite(r["n_cert"])
    ]
    mu_reach = (
        np.mean([r["mu_reach"] for r in rows if 0 < r["rho"] < L_I]) if L_I > 0 else 0.0
    )
    mu_info = (
        np.mean([r["mu_info"] for r in lin])
        if lin
        else (1.0 / L_rev if L_rev > 0 else 0.0)
    )
    nrho = [r["n_rho"] for r in lin]
    return rows, mu_reach, mu_info, nrho, I_R


def main() -> None:
    """Run the command-line entry point."""
    cases = {
        "COUPLED (L_rev = L_I)": L_I,
        "DECOUPLED (L_rev = 0, free reveal)": 0.0,
        "REVEAL-DEARER (L_rev = 2 L_I)": 2 * L_I,
        "REVEAL-CHEAPER (L_rev = L_I/2)": L_I / 2,
    }
    out = {"L_I": L_I, "sigma": SIGMA, "delta": DELTA, "rhos": RHOS, "cases": {}}
    print(
        f"decoupled observation probe.  reach cost L_I = {L_I}, mu_reach = 1/L_I = {1 / L_I:.3f}\n"
    )
    for name, L_rev in cases.items():
        rows, mu_reach, mu_info, nrho, I_R = run_case(L_rev)
        out["cases"][name] = {"L_rev": L_rev, "rows": rows}
        coupled = abs(mu_info - mu_reach) < 1e-6
        if nrho:
            nrho_str = f"{np.mean(nrho):.1f} +/- {np.std(nrho):.1f}"
            conserved = np.std(nrho) < 1e-6 * max(1.0, np.mean(nrho))
        else:
            nrho_str, conserved = "n/a (reveal free => N const => N*rho -> 0)", False
        print(f"## {name}:  L_rev = {L_rev}")
        print(
            f"   mu_reach (1/L_I) = {mu_reach:.3f};  mu_info (1/L_rev) = {mu_info:.3f}"
        )
        print(f"   coupling mu_info == mu_reach?  {'YES' if coupled else 'NO'}")
        print(
            f"   N_cert*rho (reveal-priced) = {nrho_str}"
            f"   conserved? {'YES' if conserved else 'NO'}"
        )

        if L_rev == 0.0:
            caps = [r["reveal_cap"] for r in rows]
            print(
                f"   reveal capacity vs rho = {[round(c, 2) for c in caps[:5]]}..."
                f"  (==1 for all rho: information is FREE, no safety throttle)"
            )
        print()

    with open("results/decoupled_observation.json", "w") as fh:
        json.dump(out, fh, indent=2)
    print("wrote results/decoupled_observation.json")
    print(
        "# VERDICT: coupling (mu_info==mu_reach, N*rho conserved) holds IFF L_rev == L_I, i.e. iff"
    )
    print(
        "#   the reveal channel is co-active with the floor exactly as the reach is. The price of"
    )
    print(
        "#   safety is governed by the REVEAL dual 1/L_rev. Poker: showdown forces reveal=reach."
    )


if __name__ == "__main__":
    main()
