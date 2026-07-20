""

from __future__ import annotations

import math

import numpy as np

from scripts.controlled import run_selective_labels_deployment as D

C = D.C
DELTA = D.DELTA
N = 100_000
SEED = 2026


def combined_confidence(model, rho, N=N, seed=SEED):
    ""
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


def make_model():
    return D._build_model_orig()


def main():

    D._build_model_orig = D._build_model
    D.M = 30
    D._init()
    model = D._G["model"]
    rho = 0.05

    def band(pi):
        p, q, gem, s = model["p"], model["q_true"], model["censored"], model["s"]
        val = p * pi * (q - C)
        return (
            float(val[q < C].sum()),
            float(val[gem].sum()),
            float(val[(s > 0.66) & ~gem].sum()),
        )

    print(f"baseline model, rho={rho}, N={N}")
    print(
        f"{'method':10} {'cert':>7} {'real':>7} {'cov':>4} | "
        f"{'low':>7} {'gem':>7} {'high':>7}"
    )

    Lp, Up = D._passive_confidence(model, N, SEED)
    pip, cp = D._certify(model, Lp, Up, rho)
    print(
        f"{'passive':10} {cp:7.4f} {D._realized(model, pip):7.4f} "
        f"{int(D._coverage(model, Lp, Up)):>4} | "
        + " ".join(f"{x:7.4f}" for x in band(pip))
    )

    La, Ua, _ = D._active_confidence(model, rho, N, SEED)
    pia, ca = D._certify(model, La, Ua, rho)
    print(
        f"{'active(now)':10} {ca:7.4f} {D._realized(model, pia):7.4f} "
        f"{int(D._coverage(model, La, Ua)):>4} | "
        + " ".join(f"{x:7.4f}" for x in band(pia))
    )

    Lc, Uc = combined_confidence(model, rho, N, SEED)
    pic, cc = D._certify(model, Lc, Uc, rho)
    print(
        f"{'combined':10} {cc:7.4f} {D._realized(model, pic):7.4f} "
        f"{int(D._coverage(model, Lc, Uc)):>4} | "
        + " ".join(f"{x:7.4f}" for x in band(pic))
    )

    q = model["q_true"]
    pio, co = D._certify(model, q.copy(), q.copy(), rho)
    print(
        f"{'oracle':10} {co:7.4f} {D._realized(model, pio):7.4f}    1 | "
        + " ".join(f"{x:7.4f}" for x in band(pio))
    )

    print("\nGAPS vs passive (realized):")
    print(f"  active(now): {D._realized(model, pia) - D._realized(model, pip):+.4f}")
    print(f"  combined   : {D._realized(model, pic) - D._realized(model, pip):+.4f}")

    print("\nmulti-seed combined gap (10 seeds):")
    ga, gc = [], []
    for sd in range(2026, 2036):
        Lp, Up = D._passive_confidence(model, N, sd)
        pip, _ = D._certify(model, Lp, Up, rho)
        rp = D._realized(model, pip)
        La, Ua, _ = D._active_confidence(model, rho, N, sd)
        pia, _ = D._certify(model, La, Ua, rho)
        Lc, Uc = combined_confidence(model, rho, N, sd)
        pic, _ = D._certify(model, Lc, Uc, rho)
        cov_c = D._coverage(model, Lc, Uc)
        ga.append(D._realized(model, pia) - rp)
        gc.append((D._realized(model, pic) - rp, int(cov_c)))
    print(f"  active(now) mean gap = {np.mean(ga):+.4f}")
    print(
        f"  combined    mean gap = {np.mean([g for g, _ in gc]):+.4f}  "
        f"cov={np.mean([c for _, c in gc]):.2f}"
    )


if __name__ == "__main__":
    main()
