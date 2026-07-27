"""Run the selective labels frontier experiment. See supplementary Additional Experiments."""

from __future__ import annotations

import json
import math
from pathlib import Path
import statistics
import sys
import time

import numpy as np

from scripts.controlled.selective_labels import (
    build_partial_id,
    make_lending_instance,
    robust_optimal_value,
    safe_capacity,
    simulate_logs,
)

M = int(sys.argv[1]) if len(sys.argv) > 1 else 50
C = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
TARGET_Q = float(sys.argv[3]) if len(sys.argv) > 3 else 0.25
EPS = 0.05
DELTA = 0.1
LOG_N = 200_000
LOG_SEED = 2026
RHOS = [0.0, 0.005, 0.01, 0.02, 0.04, 0.06, 0.1, 0.2, 0.4]


def _pick_target(inst, logs) -> int:
    """Compute pick target for the run selective labels frontier workflow."""
    L, U = build_partial_id(logs, delta=DELTA)
    idx = int(round(TARGET_Q * (inst.m - 1)))
    for off in range(inst.m):
        for j in (idx - off, idx + off):
            if 0 <= j < inst.m and L[j] < inst.c - 1e-3 and U[j] - L[j] > 1e-3:
                return j
    return idx


def _frontier(inst, j: int, label: str):
    """Compute frontier for the run selective labels frontier workflow."""
    U0 = robust_optimal_value(inst)
    K = np.log(2.0 / DELTA) / (2.0 * EPS**2)
    kappa0, mu0 = safe_capacity(inst, j, 0.0, U0=U0)

    rho1 = (1.0 - kappa0) / mu0 if mu0 > 1e-12 else float("inf")

    if math.isfinite(rho1):
        face = [f * rho1 for f in (0.15, 0.3, 0.45, 0.6, 0.75, 0.9)]
        past = [g * rho1 for g in (1.5, 3.0, 6.0, 12.0)]
        rhos = [0.0] + face + past
    else:
        rhos = RHOS
    rows = []
    face_prod = []
    for rho in rhos:
        kappa, mu = safe_capacity(inst, j, rho, U0=U0)
        lam = inst.p[j] * kappa
        n_cert = K / lam if lam > 1e-12 else float("inf")
        prod = n_cert * rho
        rows.append(
            {
                "rho": rho,
                "kappa": kappa,
                "mu": mu,
                "lambda": lam,
                "N_cert": n_cert,
                "N_cert_rho": prod,
            }
        )
        if 0 < rho < rho1 - 1e-9:
            face_prod.append(prod)
    cv = (
        (statistics.pstdev(face_prod) / statistics.mean(face_prod))
        if len(face_prod) > 1
        else 0.0
    )
    print(
        f"\n[{label}]  target j={j}  kappa0={kappa0:.4f}  origin dual mu={mu0:.3f}  "
        f"first-face rho1={rho1:.4f}"
    )
    print(
        f"  {'rho':>8} {'kappa':>9} {'mu':>9} {'lambda':>9} {'N_cert':>12} {'N_cert*rho':>11}"
    )
    for r in rows:
        print(
            f"  {r['rho']:>8.4f} {r['kappa']:>9.4f} {r['mu']:>9.3f} {r['lambda']:>9.5f} "
            f"{r['N_cert']:>12.1f} {r['N_cert_rho']:>11.2f}"
        )
    coupling = face_prod and inst.p[j] * mu0 * statistics.mean(face_prod)
    print(
        f"  first-face N_cert*rho: mean={statistics.mean(face_prod):.2f}  cv={cv:.2e}  "
        f"(const => conservation law)"
    )
    print(
        f"  coupling check: p_j*mu*N_cert*rho = {coupling:.2f}  (= K = {K:.2f}; "
        f"floor dual mu prices the information rate)"
    )
    return {
        "label": label,
        "target": j,
        "kappa0": kappa0,
        "mu0": mu0,
        "rho1": rho1,
        "first_face_prod_mean": statistics.mean(face_prod) if face_prod else None,
        "first_face_prod_cv": cv,
        "rows": rows,
    }


def main() -> None:
    """Run the command-line entry point."""
    t0 = time.time()
    inst = make_lending_instance(m=M, c=C, seed=LOG_SEED, monotone=True)
    logs = simulate_logs(inst, n=LOG_N, seed=LOG_SEED)
    L, U = build_partial_id(logs, delta=DELTA)

    mono = make_lending_instance(m=M, c=C, seed=LOG_SEED, monotone=True)
    mono.L, mono.U = L, U
    box = make_lending_instance(m=M, c=C, seed=LOG_SEED, monotone=False)
    box.L, box.U = L, U
    j = _pick_target(inst, logs)

    print(f"# Safe observation frontier on lending selective labels (m={M}, c={C})")
    print(
        f"# target stratum j={j}: score={j / (M - 1):.3f}, q_true={inst.q_true[j]:.3f}, "
        f"partial-ID=[{L[j]:.3f},{U[j]:.3f}], incumbent pi0={inst.pi0[j]:.0f}"
    )

    fr_mono = _frontier(mono, j, "monotone-P")
    fr_box = _frontier(box, j, "box-P")

    out = {
        "m": M,
        "c": C,
        "eps": EPS,
        "delta": DELTA,
        "target": j,
        "score": j / (M - 1),
        "q_true": float(inst.q_true[j]),
        "partial_id": [float(L[j]), float(U[j])],
        "monotone": fr_mono,
        "box": fr_box,
        "elapsed_s": time.time() - t0,
    }
    Path("results").mkdir(exist_ok=True)
    with open("results/sl_frontier.json", "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nwrote results/sl_frontier.json  ({time.time() - t0:.1f}s)")
    print(
        f"\n4TH CONSERVATION POINT (monotone first-face N_cert*rho): "
        f"{fr_mono['first_face_prod_mean']:.2f}  (cv={fr_mono['first_face_prod_cv']:.1e})"
    )


if __name__ == "__main__":
    main()
