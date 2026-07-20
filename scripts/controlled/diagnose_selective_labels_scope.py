""

from __future__ import annotations

import math

import numpy as np

from scripts.controlled import run_selective_labels_deployment as D

C = D.C
DELTA = D.DELTA
N = 100_000


def consistent_model(m=30):
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


def active_broad(model, rho, N, seed, target_set):
    ""
    rng = np.random.default_rng(seed + 777)
    m = model["m"]
    pi_probe = D.reach_probe(D._G["inst_P"], target_set, rho, U0=D._G["W_star"])
    L = np.array(D._G["L_P"])
    U = np.array(D._G["U_P"])
    z = rng.choice(m, size=N, p=model["p"])
    for i in range(m):
        ni = int((z == i).sum())
        k = int(round(ni * float(np.clip(pi_probe[i], 0.0, 1.0))))
        if k >= 5:
            good = rng.random(k) < model["beta"][i]
            q = np.where(good, model["qG"][i], model["qB"][i])
            y = (rng.random(k) < q).astype(int)
            qhat = y.mean()
            hw = math.sqrt(math.log(2.0 / DELTA) / (2.0 * k))
            lo = max(L[i], qhat - hw)
            hi = min(U[i], qhat + hw)
            if lo <= hi:
                L[i], U[i] = lo, hi
    return L, U, pi_probe


def main():
    D.M = 30
    D._build_model = consistent_model
    D._init()
    model = D._G["model"]
    gem = model["censored"]
    s = model["s"]
    gems = list(np.where(gem)[0])
    high = list(np.where(s > 0.66)[0])
    allprof = list(np.where(model["q_true"] > C)[0])

    UP = np.array(D._G["U_P"])
    LP = np.array(D._G["L_P"])
    up_gt_c = list(np.where(UP > C)[0])
    ambiguous = list(np.where((UP > C) & (LP < C))[0])
    allstrata = list(range(model["m"]))
    print(
        f"deployable target sizes: U_P>c={len(up_gt_c)}  ambiguous={len(ambiguous)}  "
        f"all={len(allstrata)}  (oracle gems+high={len(allprof)})"
    )

    variants = {
        "gems only (now)  ": gems,
        "U_P>c (deploy)   ": up_gt_c,
        "ambiguous(deploy)": ambiguous,
        "all strata(deploy)": allstrata,
        "gems+high(oracle)": gems + high,
    }

    qo = model["q_true"]
    for rho in (0.05, 0.10):
        pio, certo = D._certify(model, qo.copy(), qo.copy(), rho)
        ro = D._realized(model, pio)
        print(
            f"\n--- rho={rho}, N={N}, 10 seeds (oracle cert={certo:.4f} real={ro:.4f}) ---"
        )
        print(
            f"{'probe targets':16} {'cert':>8} {'real':>8} {'cov':>5} {'gap_real':>9} {'reach':>7}"
        )

        pcert, pr, pc = [], [], []
        for sd in range(2026, 2036):
            Lp, Up = D._passive_confidence(model, N, sd)
            pip, cep = D._certify(model, Lp, Up, rho)
            pcert.append(cep)
            pr.append(D._realized(model, pip))
            pc.append(int(D._coverage(model, Lp, Up)))
        rp = np.mean(pr)
        print(
            f"{'passive':16} {np.mean(pcert):8.4f} {rp:8.4f} {np.mean(pc):5.2f} {0.0:+9.4f}"
        )
        for name, ts in variants.items():
            ce, rr, cc, reach = [], [], [], []
            for sd in range(2026, 2036):
                L, U, pp = active_broad(model, rho, N, sd, ts)
                pi, cer = D._certify(model, L, U, rho)
                ce.append(cer)
                rr.append(D._realized(model, pi))
                cc.append(int(D._coverage(model, L, U)))
                reach.append(float(np.sum(model["p"] * pp)))
            print(
                f"{name:16} {np.mean(ce):8.4f} {np.mean(rr):8.4f} {np.mean(cc):5.2f} "
                f"{np.mean(rr) - rp:+9.4f} {np.mean(reach):7.3f}"
            )


if __name__ == "__main__":
    main()
