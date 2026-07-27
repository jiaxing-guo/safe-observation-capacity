"""Evaluate the MDP observation experiment. See supplementary Additional Experiments."""

from __future__ import annotations

import json

import numpy as np
from scipy.optimize import linprog

GAMMA = 0.9
DEPTHS = [
    0,
    1,
    2,
    4,
]
RHOS = [0.0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.45, 0.6, 0.8, 1.0]
SIGMA = 1.0
DELTA = 0.1


def build_mdp(depth: int, gamma: float = GAMMA) -> dict:
    """Build MDP for the evaluate MDP observation workflow."""
    safe = 1
    gate0 = 2
    target = 2 + depth
    n_states = 3 + depth
    trans: dict[tuple[int, int], tuple[dict[int, float], float]] = {}
    trans[(0, 0)] = ({safe: 1.0}, 0.0)
    trans[(0, 1)] = (
        {gate0: 1.0},
        0.0,
    )
    trans[(safe, 0)] = ({safe: 1.0}, 1.0)
    if depth == 0:
        trans[(0, 1)] = ({target: 1.0}, 0.0)
    else:
        for k in range(depth):
            s = gate0 + k
            nxt = (gate0 + k + 1) if k < depth - 1 else target
            trans[(s, 0)] = ({nxt: 1.0}, 0.0)
    trans[(target, 0)] = ({target: 1.0}, 0.0)
    return {
        "n_states": n_states,
        "trans": trans,
        "nu": {0: 1.0},
        "gamma": gamma,
        "target": target,
    }


def _lp_matrices(mdp: dict):
    """Compute linear program matrices for the evaluate MDP observation workflow."""
    trans, nu, gamma = mdp["trans"], mdp["nu"], mdp["gamma"]
    n_states, target = mdp["n_states"], mdp["target"]
    keys = sorted(trans.keys())
    n = len(keys)
    idx = {k: i for i, k in enumerate(keys)}

    A_flow = np.zeros((n_states, n))
    b_flow = np.zeros(n_states)
    for sp in range(n_states):
        for (s, a), (dist, _r) in trans.items():
            if s == sp:
                A_flow[sp, idx[(s, a)]] += 1.0
            A_flow[sp, idx[(s, a)]] -= gamma * dist.get(sp, 0.0)
        b_flow[sp] = (1.0 - gamma) * nu.get(sp, 0.0)
    r_vec = np.array([trans[k][1] for k in keys])
    occT = np.array([1.0 if k[0] == target else 0.0 for k in keys])
    return keys, A_flow, b_flow, r_vec, occT


def _reward_max(mdp: dict) -> float:
    """Compute reward max for the evaluate MDP observation workflow."""
    keys, A_flow, b_flow, r_vec, _ = _lp_matrices(mdp)
    res = linprog(
        -r_vec,
        A_eq=A_flow,
        b_eq=b_flow,
        bounds=[(0.0, None)] * len(keys),
        method="highs",
    )
    return float(-res.fun)


def _capacity(mdp: dict, v_ref: float, rho: float):
    """Compute capacity for the evaluate MDP observation workflow."""
    keys, A_flow, b_flow, r_vec, occT = _lp_matrices(mdp)

    A_ub = (-r_vec).reshape(1, -1)
    b_ub = np.array([-(v_ref - rho)])
    res = linprog(
        -occT,
        A_ub=A_ub,
        b_ub=b_ub,
        A_eq=A_flow,
        b_eq=b_flow,
        bounds=[(0.0, None)] * len(keys),
        method="highs",
    )
    if not res.success:
        return 0.0, 0.0
    kappa = float(occT @ res.x)
    sp = 0.0
    try:
        sp = -float(res.ineqlin.marginals[0])
    except Exception:
        sp = 0.0
    return kappa, sp


def main() -> None:
    """Run the command-line entry point."""
    print("# Probe: safe-RL occupancy-measure capacity coupling (3rd convex body)\n")
    out = {"gamma": GAMMA, "rhos": RHOS, "sigma": SIGMA, "delta": DELTA, "depths": {}}

    for depth in DEPTHS:
        mdp = build_mdp(depth)
        v_ref = _reward_max(mdp)
        omega_max, _ = _capacity(mdp, v_ref, rho=1e6)
        print(
            f"== target depth={depth}  (v_ref={v_ref:.4f}, omega_max={omega_max:.4f}) =="
        )
        print(
            f"   {'rho':>6}{'kappa':>10}{'shadow mu_I':>13}{'N_cert':>10}{'N_cert*rho':>12}"
        )
        curve = []
        for rho in RHOS:
            kappa, sp = _capacity(mdp, v_ref, rho)
            n_cert = (SIGMA**2 / (DELTA**2 * kappa)) if kappa > 1e-12 else float("inf")
            prod = n_cert * rho if np.isfinite(n_cert) else float("nan")
            curve.append({"rho": rho, "kappa": kappa, "shadow": sp, "n_cert": n_cert})
            print(
                f"   {rho:>6.2f}{kappa:>10.4f}{sp:>13.4f}{n_cert:>10.1f}{prod:>12.1f}"
            )
        out["depths"][str(depth)] = {
            "v_ref": v_ref,
            "omega_max": omega_max,
            "curve": curve,
        }

        kappa_0 = curve[0]["kappa"]
        lin = [c for c in curve if 0 < c["rho"] and c["kappa"] < 0.999 * omega_max]
        if lin:
            slopes = [(c["kappa"] - kappa_0) / c["rho"] for c in lin]
            shadows = [c["shadow"] for c in lin]
            consts = [c["n_cert"] * c["rho"] for c in lin if np.isfinite(c["n_cert"])]
            ok = abs(np.mean(slopes) - np.mean(shadows)) < 1e-3
            print(
                f"   -> kappa_0={kappa_0:.4f};  (kappa-kappa_0)/rho = {np.mean(slopes):.4f}"
                f"  vs LP shadow mu_I = {np.mean(shadows):.4f}"
            )
            print(
                f"   -> s_I == mu_I (same dual prices capacity on occupancy polytope)?  "
                f"{'YES' if ok else 'NO'}"
            )
            print(
                f"   -> N_cert*rho = {np.mean(consts):.1f} +/- {np.std(consts):.1f}"
                f"  => N ~ 1/rho (price of safety)"
            )
        print()

    with open("results/probe_rl_sao.json", "w") as fh:
        json.dump(out, fh, indent=2)
    print("# wrote results/probe_rl_sao.json")


if __name__ == "__main__":
    main()
