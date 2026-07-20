""

from __future__ import annotations

import numpy as np

RNG = np.random.default_rng(2026)
SIGMA = 0.7
THETA0 = 0.3
N_MC = 4_000_000


def _loglik(theta, b, z, q_fn, sigma):
    q = q_fn(theta)

    log_phi = -0.5 * ((z - theta) / sigma) ** 2 - np.log(sigma * np.sqrt(2 * np.pi))
    return b * (np.log(q) + log_phi) + (1 - b) * np.log1p(-q)


def fisher_mc(q_fn, sigma, theta0, n=N_MC, h=1e-4):
    ""
    q = q_fn(theta0)
    b = (RNG.random(n) < q).astype(float)
    z = np.where(
        b > 0, theta0 + sigma * RNG.standard_normal(n), RNG.standard_normal(n) * 3.0
    )

    lp_plus = _loglik(theta0 + h, b, z, q_fn, sigma)
    lp_minus = _loglik(theta0 - h, b, z, q_fn, sigma)
    score = (lp_plus - lp_minus) / (2 * h)
    return float(np.mean(score**2))


def main() -> None:
    I_R = 1.0 / SIGMA**2
    print(f"# Gaussian reveal: I_R = 1/sigma^2 = {I_R:.4f}\n")
    print(
        f"{'case':<28}{'q':>8}{'q_prime':>9}{'I_step MC':>12}{'formula':>12}{'q*I_R':>10}{'match':>7}"
    )

    for qc in [0.05, 0.2, 0.5]:

        def q_fn(t, qc=qc):
            return qc + 0.0 * t

        Imc = fisher_mc(q_fn, SIGMA, THETA0)
        formula = 0.0 + qc * I_R
        ok = abs(Imc - formula) / formula < 0.03
        print(
            f"{'preserving (q const)':<28}{qc:>8.3f}{0.0:>9.3f}{Imc:>12.4f}"
            f"{formula:>12.4f}{qc * I_R:>10.4f}{'YES' if ok else 'NO':>7}"
        )

    print()
    for qc, slope in [(0.2, 0.3), (0.2, 0.6), (0.5, 0.5)]:

        def q_fn(t, qc=qc, slope=slope):
            return qc + slope * t

        q_at = q_fn(THETA0)
        Imc = fisher_mc(q_fn, SIGMA, THETA0)
        rate_term = slope**2 / (q_at * (1 - q_at))
        formula = rate_term + q_at * I_R
        ok = abs(Imc - formula) / formula < 0.03
        print(
            f"{'changing (q + slope*theta)':<28}{q_at:>8.3f}{slope:>9.3f}{Imc:>12.4f}"
            f"{formula:>12.4f}{q_at * I_R:>10.4f}{'YES' if ok else 'NO':>7}"
        )
        print(
            f"    -> extra RATE-channel term q'^2/(q(1-q)) = {rate_term:.4f}"
            f"  (uncapped, public; absent when q'=0)"
        )

    print("\n# Conclusion:")
    print(
        "#  (L1) rate-preserving => I_step = q * I_R exactly: information per step = (reveal rate)"
    )
    print(
        "#  x (per-reveal Fisher). The reveal rate q<=lambda_rho=kappa_rho*pi is bought with safety"
    )
    print(
        "#  budget at the dual rate mu_I (kappa_rho=kappa_0+mu_I*rho). Hence the SAME mu_I converts"
    )
    print(
        "#  budget -> reach -> reveals -> Fisher information: one exchange rate, not two theorems."
    )
    print(
        "#  Rate-CHANGING directions add an uncapped public channel => learned free => never priced;"
    )
    print("#  only the censored (rate-preserving) fiber pays the price of safety.")


if __name__ == "__main__":
    main()
