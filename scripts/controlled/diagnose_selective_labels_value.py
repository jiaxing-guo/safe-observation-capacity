""

from __future__ import annotations

import numpy as np

from scripts.controlled import run_selective_labels_deployment as D

C = D.C
N = 100_000
SEED = 2026


def make_model(gem_lo, gem_hi, qG_gem, qB_gem, selG_gem, selB_gem, m=30):
    ""
    s = np.linspace(0.0, 1.0, m)
    p = np.full(m, 1.0 / m)
    gem = (s >= gem_lo) & (s <= gem_hi)
    high = s > gem_hi
    qG = np.where(gem, qG_gem, np.clip(0.20 + 0.75 * s, 0.0, 1.0))
    qB = np.where(
        gem, qB_gem, np.clip(np.where(gem, qG_gem, 0.20 + 0.75 * s) - 0.40, 0.0, 1.0)
    )
    beta = np.full(m, 0.5)
    q_true = beta * qG + (1.0 - beta) * qB
    q_true = np.maximum.accumulate(q_true)
    selG = np.where(gem, selG_gem, np.where(high, 0.9, 0.15))
    selB = np.where(gem, selB_gem, np.where(high, 0.9, 0.05))
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


def run_once(model, rho, N=N, seed=SEED):
    ""
    D.M = model["m"]
    D._build_model = lambda: model
    D._init()
    out = {}

    q = model["q_true"]
    pi_o, cert_o = D._certify(model, q.copy(), q.copy(), rho)
    out["oracle"] = (pi_o, cert_o, D._realized(model, pi_o), True)

    Lp, Up = D._passive_confidence(model, N, seed)
    pi_p, cert_p = D._certify(model, Lp, Up, rho)
    out["passive"] = (
        pi_p,
        cert_p,
        D._realized(model, pi_p),
        D._coverage(model, Lp, Up),
    )

    La, Ua, _ = D._active_confidence(model, rho, N, seed)
    pi_a, cert_a = D._certify(model, La, Ua, rho)
    out["active"] = (pi_a, cert_a, D._realized(model, pi_a), D._coverage(model, La, Ua))
    return out


def band_decomp(model, pi):
    ""
    p, q, gem = model["p"], model["q_true"], model["censored"]
    s = model["s"]
    val = p * pi * (q - C)
    low = q < C
    high = (s > 0.66) & ~gem
    return float(val[low].sum()), float(val[gem].sum()), float(val[high].sum())


def main():
    base = dict(
        gem_lo=0.40, gem_hi=0.66, qG_gem=0.80, qB_gem=0.30, selG_gem=0.15, selB_gem=0.60
    )
    rho = 0.05

    print("=" * 78)
    print("PART 1 — where the value lives (baseline model, rho=0.05, N=1e5)")
    print("=" * 78)
    model = make_model(**base)
    q, gem = model["q_true"], model["censored"]
    print(
        f"gem strata: {int(gem.sum())}/{model['m']}  gem mass={model['p'][gem].sum():.3f}"
    )
    print(
        f"gem q_true (mean)={q[gem].mean():.3f}  margin (q-c)={q[gem].mean() - C:+.3f}"
    )
    print(
        f"max recoverable gem value = mass*margin = {(model['p'][gem] * (q[gem] - C)).sum():.4f}"
    )
    print()
    res = run_once(model, rho)
    print(
        f"{'method':8} {'cert':>7} {'real':>7} {'cov':>4} | "
        f"{'low':>7} {'gem':>7} {'high':>7}"
    )
    for meth in ("passive", "active", "oracle"):
        pi, cert, real, cov = res[meth]
        lo, ge, hi = band_decomp(model, pi)
        print(
            f"{meth:8} {cert:7.4f} {real:7.4f} {int(cov):>4} | "
            f"{lo:7.4f} {ge:7.4f} {hi:7.4f}"
        )
    pa, ac = res["passive"], res["active"]
    print(f"\nactive-passive realized gap = {ac[2] - pa[2]:+.4f}")
    print(
        f"  (gem-band: passive {band_decomp(model, pa[0])[1]:.4f} -> "
        f"active {band_decomp(model, ac[0])[1]:.4f})"
    )

    print("\n" + "=" * 78)
    print("PART 2 — scaling the GEM MARGIN (qG_gem up => q_true_gem up)")
    print("=" * 78)
    print(
        f"{'qG_gem':>7} {'q_gem':>6} {'margin':>7} {'maxrec':>7} | "
        f"{'pass_r':>7} {'act_r':>7} {'gap':>7} {'act_cov':>7}"
    )
    for qG in (0.70, 0.80, 0.90, 1.00):
        m2 = make_model(**{**base, "qG_gem": qG})
        qg = m2["q_true"][m2["censored"]].mean()
        maxrec = (m2["p"][m2["censored"]] * (m2["q_true"][m2["censored"]] - C)).sum()
        r = run_once(m2, rho)
        print(
            f"{qG:7.2f} {qg:6.3f} {qg - C:+7.3f} {maxrec:7.4f} | "
            f"{r['passive'][2]:7.4f} {r['active'][2]:7.4f} "
            f"{r['active'][2] - r['passive'][2]:+7.4f} {int(r['active'][3]):>7}"
        )

    print("\n" + "=" * 78)
    print("PART 3 — scaling the GEM MASS (band width)")
    print("=" * 78)
    print(
        f"{'band':>12} {'ngem':>5} {'mass':>6} {'maxrec':>7} | "
        f"{'pass_r':>7} {'act_r':>7} {'gap':>7} {'act_cov':>7}"
    )
    for lo, hi in ((0.45, 0.60), (0.40, 0.66), (0.35, 0.72), (0.30, 0.78)):
        m3 = make_model(**{**base, "gem_lo": lo, "gem_hi": hi})
        ng = int(m3["censored"].sum())
        mass = m3["p"][m3["censored"]].sum()
        maxrec = (m3["p"][m3["censored"]] * (m3["q_true"][m3["censored"]] - C)).sum()
        r = run_once(m3, rho)
        print(
            f"[{lo:.2f},{hi:.2f}] {ng:>5} {mass:6.3f} {maxrec:7.4f} | "
            f"{r['passive'][2]:7.4f} {r['active'][2]:7.4f} "
            f"{r['active'][2] - r['passive'][2]:+7.4f} {int(r['active'][3]):>7}"
        )

    print("\n" + "=" * 78)
    print("PART 4 — scaling the BUDGET rho (baseline model)")
    print("=" * 78)
    print(
        f"{'rho':>6} | {'pass_r':>7} {'act_r':>7} {'gap':>7} "
        f"{'orc_r':>7} {'act_cov':>7} {'pass_cov':>8}"
    )
    for rr in (0.01, 0.05, 0.10, 0.20):
        r = run_once(make_model(**base), rr)
        print(
            f"{rr:6.2f} | {r['passive'][2]:7.4f} {r['active'][2]:7.4f} "
            f"{r['active'][2] - r['passive'][2]:+7.4f} {r['oracle'][2]:7.4f} "
            f"{int(r['active'][3]):>7} {int(r['passive'][3]):>8}"
        )


if __name__ == "__main__":
    main()
