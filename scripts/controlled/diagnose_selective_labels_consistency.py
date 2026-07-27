"""Diagnose selective labels consistency. See supplementary Additional Experiments."""

from __future__ import annotations

import math

import numpy as np

from scripts.controlled import run_selective_labels_deployment as D

C = D.C
DELTA = D.DELTA
N = 100_000


def consistent_model(m=30):
    """Compute consistent model for the diagnose selective labels consistency workflow."""
    s = np.linspace(0.0, 1.0, m)
    p = np.full(m, 1.0 / m)
    gem = (s >= 0.40) & (s <= 0.66)
    high = s > 0.66
    qG = np.where(gem, 0.80, np.clip(0.20 + 0.75 * s, 0.0, 1.0))
    qB = np.where(
        gem, 0.30, np.clip(np.where(gem, 0.80, 0.20 + 0.75 * s) - 0.40, 0.0, 1.0)
    )

    qG = np.maximum.accumulate(qG)
    qB = np.maximum.accumulate(qB)
    beta = np.full(m, 0.5)
    q_true = beta * qG + (1.0 - beta) * qB
    selG = np.where(gem, 0.15, np.where(high, 0.9, 0.15))
    selB = np.where(gem, 0.60, np.where(high, 0.9, 0.05))
    return {
        "m": m,
        "s": s,
        "p": p,
        "qG": qG,
        "qB": qB,
        "beta": beta,
        "q_true": q_true,
        "censored": gem,
        "selG": selG,
        "selB": selB,
    }


def combined_confidence(model, rho, N, seed):
    """Construct confidence constraints for combined."""
    L, U = D._passive_confidence(model, N, seed)
    rng = np.random.default_rng(seed + 777)
    m = model["m"]
    targets = list(np.where(model["censored"])[0])
    pi_probe = D.reach_probe(D._G["inst_P"], targets, rho, U0=D._G["W_star"])
    z = rng.choice(m, size=N, p=model["p"])
    for i in targets:
        ni = int((z == i).sum())
        k = int(round(ni * float(np.clip(pi_probe[i], 0.0, 1.0))))
        if k >= 5:
            good = rng.random(k) < model["beta"][i]
            q = np.where(good, model["qG"][i], model["qB"][i])
            y = (rng.random(k) < q).astype(int)
            qhat = y.mean()
            hw = math.sqrt(math.log(2.0 / DELTA) / (2.0 * k))
            lo = max(float(D._G["L_P"][i]), qhat - hw)
            hi = min(float(D._G["U_P"][i]), qhat + hw)
            if lo <= hi:
                L[i], U[i] = lo, hi
    return L, U


def main():
    """Run the command-line entry point."""
    D.M = 30
    D._build_model = consistent_model
    D._init()
    model = D._G["model"]
    q, gem = model["q_true"], model["censored"]
    print("CONSISTENT model (monotone DGP). q_true now == true mean everywhere.")
    print(
        f"  gem mass={model['p'][gem].sum():.3f}  gem q_true={q[gem].mean():.3f}  "
        f"margin={q[gem].mean() - C:+.3f}"
    )
    print(f"  max recoverable gem value = {(model['p'][gem] * (q[gem] - C)).sum():.4f}")
    print(f"  monotone check: {bool(np.all(np.diff(q) >= -1e-12))}")

    for rho in (0.05, 0.10):
        print(f"\n--- rho={rho}, N={N}, 10 seeds ---")
        print(f"{'method':12} {'real':>8} {'cov':>5} {'gap_vs_pass':>12}")
        agg = {
            meth: {"real": [], "cov": [], "gap": []}
            for meth in ("passive", "active", "combined", "oracle")
        }
        for sd in range(2026, 2036):
            Lp, Up = D._passive_confidence(model, N, sd)
            pip, _ = D._certify(model, Lp, Up, rho)
            rp = D._realized(model, pip)
            agg["passive"]["real"].append(rp)
            agg["passive"]["cov"].append(int(D._coverage(model, Lp, Up)))
            agg["passive"]["gap"].append(0.0)

            La, Ua, _ = D._active_confidence(model, rho, N, sd)
            pia, _ = D._certify(model, La, Ua, rho)
            agg["active"]["real"].append(D._realized(model, pia))
            agg["active"]["cov"].append(int(D._coverage(model, La, Ua)))
            agg["active"]["gap"].append(D._realized(model, pia) - rp)

            Lc, Uc = combined_confidence(model, rho, N, sd)
            pic, _ = D._certify(model, Lc, Uc, rho)
            agg["combined"]["real"].append(D._realized(model, pic))
            agg["combined"]["cov"].append(int(D._coverage(model, Lc, Uc)))
            agg["combined"]["gap"].append(D._realized(model, pic) - rp)

        qo = model["q_true"]
        pio, _ = D._certify(model, qo.copy(), qo.copy(), rho)
        ro = D._realized(model, pio)
        for meth in ("passive", "active", "combined"):
            a = agg[meth]
            print(
                f"{meth:12} {np.mean(a['real']):8.4f} {np.mean(a['cov']):5.2f} "
                f"{np.mean(a['gap']):+12.4f}"
            )
        print(
            f"{'oracle':12} {ro:8.4f} {1.0:5.2f} {ro - np.mean(agg['passive']['real']):+12.4f}"
        )


if __name__ == "__main__":
    main()
