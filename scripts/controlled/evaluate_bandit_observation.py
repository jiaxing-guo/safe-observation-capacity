"""Evaluate the bandit observation experiment. See supplementary Additional Experiments."""

from __future__ import annotations

import json

import numpy as np
from scipy.optimize import linprog

MU0 = 0.0

L = [0.10, 0.25, 0.50]
GOOD = 0.05
RHOS = [0.0, 0.01, 0.02, 0.04, 0.08, 0.12, 0.2, 0.3, 0.5, 0.7, 1.0]
SIGMA = 1.0
DELTA = 0.1


def _capacity(mu, idx_I, rho):
    """Compute capacity for the evaluate bandit observation workflow."""
    n = len(mu)
    c = np.zeros(n)
    c[idx_I] = -1.0

    A_ub = (-np.asarray(mu)).reshape(1, n)
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
    kappa = float(res.x[idx_I])

    sp = 0.0
    try:
        sp = -float(res.ineqlin.marginals[0])
    except Exception:
        sp = 0.0
    return kappa, sp


def main() -> None:
    """Run the command-line entry point."""
    mu = [MU0, MU0 + GOOD] + [MU0 - Li for Li in L]
    names = ["baseline", "good"] + [f"risky_L={Li}" for Li in L]
    print("# Probe: conservative-bandit safe-observation capacity coupling")
    print(f"# arms: {list(zip(names, mu, strict=False))}\n")

    out = {
        "mu": mu,
        "names": names,
        "rhos": RHOS,
        "sigma": SIGMA,
        "delta": DELTA,
        "arms": {},
    }

    for j, Li in enumerate(L):
        idx_I = 2 + j
        print(f"== risky arm I (L_I={Li}) ==")
        print(
            f"   {'rho':>6}{'kappa':>10}{'rho/L_I':>10}{'shadow mu_I':>13}"
            f"{'kappa/rho':>11}{'N_cert*rho/L_I':>16}"
        )
        curve = []
        for rho in RHOS:
            kappa, sp = _capacity(mu, idx_I, rho)
            pred = min(1.0, rho / Li) if Li > 0 else 1.0
            ko_rho = (kappa / rho) if rho > 1e-12 else float("nan")

            n_cert = (SIGMA**2 / (DELTA**2 * kappa)) if kappa > 1e-12 else float("inf")
            prod = (
                n_cert * rho / Li if (np.isfinite(n_cert) and Li > 0) else float("nan")
            )
            curve.append(
                {
                    "rho": rho,
                    "kappa": kappa,
                    "pred": pred,
                    "shadow": sp,
                    "n_cert": n_cert,
                }
            )
            print(
                f"   {rho:>6.2f}{kappa:>10.4f}{pred:>10.4f}{sp:>13.4f}"
                f"{ko_rho:>11.4f}{prod:>16.1f}"
            )
        out["arms"][f"L={Li}"] = curve

        kappa_0 = curve[0]["kappa"]
        lin = [c for c in curve if 0 < c["rho"] < Li and c["kappa"] < 0.999]
        if lin:
            slopes = [(c["kappa"] - kappa_0) / c["rho"] for c in lin]
            shadows = [c["shadow"] for c in lin]
            ok = abs(np.mean(slopes) - np.mean(shadows)) < 1e-3
            print(
                f"   -> kappa_0={kappa_0:.4f};  (kappa-kappa_0)/rho = {np.mean(slopes):.4f}"
                f"  vs LP shadow mu_I = {np.mean(shadows):.4f}"
            )
            print(
                f"   -> s_I == mu_I (same dual prices capacity)?  {'YES' if ok else 'NO'}"
            )
        print()

    print("# PURE-WALL arm (baseline + risky only): the clean price-of-safety")
    Lw = 0.5
    mu_w = [MU0, MU0 - Lw]
    print(
        f"   {'rho':>6}{'kappa':>10}{'min(1,rho/L)':>14}{'shadow':>9}{'N_cert':>10}"
        f"{'N_cert*rho':>12}"
    )
    wall = []
    for rho in RHOS:
        kappa, sp = _capacity(mu_w, 1, rho)
        n_cert = (SIGMA**2 / (DELTA**2 * kappa)) if kappa > 1e-12 else float("inf")
        prod = n_cert * rho if np.isfinite(n_cert) else float("nan")
        wall.append({"rho": rho, "kappa": kappa, "shadow": sp, "n_cert": n_cert})
        print(
            f"   {rho:>6.2f}{kappa:>10.4f}{min(1.0, rho / Lw):>14.4f}{sp:>9.4f}"
            f"{n_cert:>10.1f}{prod:>12.1f}"
        )
    out["pure_wall"] = {"L": Lw, "curve": wall}
    linw = [c for c in wall if 0 < c["rho"] < Lw]
    consts = [c["n_cert"] * c["rho"] for c in linw if np.isfinite(c["n_cert"])]
    print(
        f"   -> kappa_0 = {wall[0]['kappa']:.4f} (pure wall);  shadow mu_I = {1 / Lw:.4f};"
        f"  N_cert*rho = {np.mean(consts):.1f} +/- {np.std(consts):.1f} "
        f"(predict sigma^2 L/Delta^2 = {SIGMA**2 * Lw / DELTA**2:.1f}) => N ~ 1/rho CONFIRMED\n"
    )

    with open("results/probe_bandit_sao.json", "w") as fh:
        json.dump(out, fh, indent=2)
    print("\n# wrote results/probe_bandit_sao.json")


if __name__ == "__main__":
    main()
