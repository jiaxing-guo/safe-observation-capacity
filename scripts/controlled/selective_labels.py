"""Research utility for selective labels. See supplementary Additional Experiments."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy.optimize import linprog


@dataclass
class SLInstance:
    """Represent selective-label instance for the selective labels workflow."""

    p: np.ndarray
    L: np.ndarray
    U: np.ndarray
    c: float
    pi0: np.ndarray
    monotone: bool = True
    q_true: np.ndarray | None = None

    @property
    def m(self) -> int:
        """Return the number of latent categories in the instance."""
        return int(len(self.p))


def _monotone_ub(m: int) -> tuple[np.ndarray, np.ndarray] | None:
    """Compute monotone ub for the selective labels workflow."""
    if m < 2:
        return None
    rows = []
    for i in range(m - 1):
        r = np.zeros(m)
        r[i] = 1.0
        r[i + 1] = -1.0
        rows.append(r)
    return np.array(rows), np.zeros(m - 1)


def robust_utility(inst: SLInstance, pi: np.ndarray) -> tuple[float, np.ndarray]:
    """Compute robust utility for the selective labels workflow."""
    w = inst.p * np.asarray(pi, dtype=float)
    bounds = list(zip(inst.L, inst.U, strict=True))
    A_ub, b_ub = (None, None)
    if inst.monotone:
        mu = _monotone_ub(inst.m)
        if mu is not None:
            A_ub, b_ub = mu
    res = linprog(c=w, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
    if not res.success:
        raise RuntimeError(f"inner robust-utility LP failed: {res.message}")
    q = np.asarray(res.x, dtype=float)
    W = float(w @ q - inst.c * w.sum())
    return W, q


def incumbent_utility(inst: SLInstance) -> float:
    """Compute incumbent utility for the selective labels workflow."""
    return robust_utility(inst, inst.pi0)[0]


def robust_optimal_value(inst: SLInstance) -> float:
    """Compute the robust optimal value."""
    return robust_response(inst, 0.0, U0=0.0)[1]


def _capacity_program(inst: SLInstance, j: int, rho: float, U0: float):
    """Compute capacity program for the selective labels workflow."""
    m = inst.m
    n_e = (m - 1) if inst.monotone else 0
    nvar = 3 * m + n_e
    iA, iB, iE = m, 2 * m, 3 * m

    c_obj = np.zeros(nvar)
    c_obj[j] = -1.0

    A_eq = np.zeros((m, nvar))
    for k in range(m):
        A_eq[k, k] = -inst.p[k]
        A_eq[k, iA + k] = -1.0
        A_eq[k, iB + k] = 1.0
        if inst.monotone:
            if k <= m - 2:
                A_eq[k, iE + k] += -1.0
            if k >= 1:
                A_eq[k, iE + (k - 1)] += 1.0
    b_eq = np.zeros(m)

    A_ub = np.zeros((1, nvar))
    A_ub[0, 0:m] = inst.c * inst.p
    A_ub[0, iA : iA + m] = inst.U
    A_ub[0, iB : iB + m] = -inst.L
    b_ub = np.array([rho - U0])

    bounds = [(0.0, 1.0)] * m + [(0.0, None)] * (2 * m + n_e)
    return c_obj, A_ub, b_ub, A_eq, b_eq, bounds


def safe_capacity(
    inst: SLInstance, j: int, rho: float, U0: float | None = None
) -> tuple[float, float]:
    """Compute safe capacity for the selective labels workflow."""
    if U0 is None:
        U0 = incumbent_utility(inst)
    c_obj, A_ub, b_ub, A_eq, b_eq, bounds = _capacity_program(inst, j, rho, U0)
    res = linprog(
        c=c_obj,
        A_ub=A_ub,
        b_ub=b_ub,
        A_eq=A_eq,
        b_eq=b_eq,
        bounds=bounds,
        method="highs",
    )
    if not res.success:
        raise RuntimeError(f"safe-capacity LP failed (j={j}, rho={rho}): {res.message}")
    kappa = float(-res.fun)

    mu = 0.0
    if getattr(res, "ineqlin", None) is not None and len(res.ineqlin.marginals) > 0:
        mu = float(-res.ineqlin.marginals[0])
    return kappa, mu


def reach_probe(
    inst: SLInstance, targets, rho: float, U0: float | None = None
) -> np.ndarray:
    """Compute reach probe for the selective labels workflow."""
    if U0 is None:
        U0 = robust_optimal_value(inst)
    c_obj, A_ub, b_ub, A_eq, b_eq, bounds = _capacity_program(inst, 0, rho, U0)
    c_obj = np.zeros_like(c_obj)
    for i in targets:
        c_obj[i] = -inst.p[i]
    res = linprog(
        c=c_obj,
        A_ub=A_ub,
        b_ub=b_ub,
        A_eq=A_eq,
        b_eq=b_eq,
        bounds=bounds,
        method="highs",
    )
    if not res.success:
        raise RuntimeError(f"reach-probe LP failed (rho={rho}): {res.message}")
    return np.asarray(res.x[: inst.m], dtype=float)


def robust_response(
    inst: SLInstance, rho: float, U0: float | None = None
) -> tuple[np.ndarray, float]:
    """Compute robust response for the selective labels workflow."""
    if U0 is None:
        U0 = incumbent_utility(inst)
    m = inst.m
    n_e = (m - 1) if inst.monotone else 0
    nvar = 3 * m + n_e
    iA, iB = m, 2 * m

    c_obj = np.zeros(nvar)
    c_obj[0:m] = inst.c * inst.p
    c_obj[iA : iA + m] = inst.U
    c_obj[iB : iB + m] = -inst.L

    _, _, _, A_eq, b_eq, bounds = _capacity_program(inst, 0, rho, U0)
    res = linprog(c=c_obj, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
    if not res.success:
        raise RuntimeError(f"robust-response LP failed: {res.message}")
    pi = np.asarray(res.x[0:m], dtype=float)
    return pi, float(-res.fun)


def certified_safe_response(
    inst_P: SLInstance,
    C_L: np.ndarray,
    C_U: np.ndarray,
    rho: float,
    U0: float | None = None,
) -> tuple[np.ndarray, float]:
    """Compute certified safe response for the selective labels workflow."""
    if U0 is None:
        U0 = robust_optimal_value(inst_P)
    m = inst_P.m
    p, c = inst_P.p, inst_P.c
    n_e = (m - 1) if inst_P.monotone else 0
    i_pi, i_v, i_aC, i_bC, i_aP, i_bP, i_eP = (
        0,
        m,
        m + 1,
        2 * m + 1,
        3 * m + 1,
        4 * m + 1,
        5 * m + 1,
    )
    nvar = 5 * m + 1 + n_e + (m if False else 0)
    nvar = 5 * m + 1 + n_e

    c_obj = np.zeros(nvar)
    c_obj[i_v] = -1.0

    A_eq = np.zeros((2 * m, nvar))
    b_eq = np.zeros(2 * m)
    for k in range(m):
        A_eq[k, i_pi + k] = -p[k]
        A_eq[k, i_aC + k] = -1.0
        A_eq[k, i_bC + k] = 1.0
        A_eq[m + k, i_pi + k] = -p[k]
        A_eq[m + k, i_aP + k] = -1.0
        A_eq[m + k, i_bP + k] = 1.0
        if inst_P.monotone:
            if k <= m - 2:
                A_eq[m + k, i_eP + k] += -1.0
            if k >= 1:
                A_eq[m + k, i_eP + (k - 1)] += 1.0

    A_ub = np.zeros((2, nvar))
    A_ub[0, i_v] = 1.0
    A_ub[0, i_bC : i_bC + m] = -np.asarray(C_L)
    A_ub[0, i_aC : i_aC + m] = np.asarray(C_U)
    A_ub[0, i_pi : i_pi + m] = c * p
    A_ub[1, i_bP : i_bP + m] = -inst_P.L
    A_ub[1, i_aP : i_aP + m] = inst_P.U
    A_ub[1, i_pi : i_pi + m] = c * p
    b_ub = np.array([0.0, rho - U0])

    bounds = [(0.0, 1.0)] * m + [(None, None)] + [(0.0, None)] * (4 * m + n_e)
    res = linprog(
        c=c_obj,
        A_ub=A_ub,
        b_ub=b_ub,
        A_eq=A_eq,
        b_eq=b_eq,
        bounds=bounds,
        method="highs",
    )
    if not res.success:
        raise RuntimeError(f"certified-safe-response LP failed: {res.message}")
    pi = np.asarray(res.x[i_pi : i_pi + m], dtype=float)
    return pi, float(res.x[i_v])


def make_lending_instance(
    m: int = 50,
    c: float = 0.5,
    seed: int = 2026,
    censored_frac: float = 0.5,
    monotone: bool = True,
) -> SLInstance:
    """Create lending instance for the selective labels workflow."""
    rng = np.random.default_rng(seed)
    p = np.full(m, 1.0 / m)
    score = np.linspace(0.0, 1.0, m)
    q_true = 0.15 + 0.8 * score
    q_true = np.clip(q_true + 0.02 * rng.standard_normal(m), 0.02, 0.98)
    q_true = np.maximum.accumulate(q_true)
    return SLInstance(
        p=p,
        L=np.zeros(m),
        U=np.ones(m),
        c=c,
        pi0=(q_true >= c).astype(float),
        monotone=monotone,
        q_true=q_true,
    )


def simulate_logs(inst: SLInstance, n: int, seed: int) -> dict:
    """Simulate logs for the selective labels workflow."""
    rng = np.random.default_rng(seed)
    z = rng.choice(inst.m, size=n, p=inst.p)

    a = (inst.q_true[z] + 0.10 * rng.standard_normal(n) >= inst.c).astype(int)
    y = (rng.random(n) < inst.q_true[z]).astype(int)
    r = np.zeros(inst.m)
    qobs = np.full(inst.m, np.nan)
    for i in range(inst.m):
        zi = z == i
        ni = int(zi.sum())
        if ni == 0:
            continue
        ai = a[zi]
        r[i] = ai.mean()
        if ai.sum() > 0:
            qobs[i] = y[zi][ai == 1].mean()
    return {"r": r, "qobs": qobs, "n": n}


def build_partial_id(logs: dict, delta: float = 0.1) -> tuple[np.ndarray, np.ndarray]:
    """Build partial identifier for the selective labels workflow."""
    r, qobs, n = logs["r"], logs["qobs"], logs["n"]
    m = len(r)
    L = np.zeros(m)
    U = np.ones(m)
    halfwidth = math.sqrt(math.log(2.0 / delta) / (2.0 * max(n, 1)))
    for i in range(m):
        if r[i] <= 0 or np.isnan(qobs[i]):
            continue
        base = r[i] * qobs[i]
        L[i] = max(0.0, base - halfwidth)
        U[i] = min(1.0, base + (1.0 - r[i]) + halfwidth)
    return L, U


def _regression_1b() -> bool:
    """Compute regression 1b for the selective labels workflow."""
    inst = SLInstance(
        p=np.array([0.8, 0.2]),
        L=np.array([0.9, 0.52]),
        U=np.array([0.9, 0.92]),
        c=0.65,
        pi0=np.array([1.0, 0.0]),
        monotone=False,
    )
    U0 = incumbent_utility(inst)
    slope = 0.2 * (0.65 - 0.52)
    ok = abs(U0 - 0.20) < 1e-9
    print(f"  U0={U0:.6f} (expect 0.20)  slope={slope:.4f}")
    print(f"  {'rho':>7} {'kappa1':>9} {'predicted':>10} {'mu':>9} {'pred mu':>9}")
    for rho in [0.004, 0.012, 0.02, 0.026, 0.05]:
        kappa, mu = safe_capacity(inst, 1, rho, U0=U0)
        pred = min(1.0, rho / slope)
        pred_mu = (1.0 / slope) if rho < slope else 0.0
        ok &= abs(kappa - pred) < 1e-6

        if rho < slope - 1e-3:
            ok &= abs(mu - 1.0 / slope) < 1e-3
        print(f"  {rho:>7.3f} {kappa:>9.5f} {pred:>10.5f} {mu:>9.3f} {pred_mu:>9.3f}")
    return bool(ok)


def _demo_monotone_coupling() -> bool:
    """Compute demo monotone coupling for the selective labels workflow."""
    p = np.full(3, 1.0 / 3.0)
    L = np.array([0.6, 0.3, 0.5])
    U = np.array([0.9, 0.9, 0.9])
    pi = np.array([1.0, 1.0, 1.0])
    box = SLInstance(p=p, L=L, U=U, c=0.5, pi0=np.zeros(3), monotone=False)
    mon = SLInstance(p=p, L=L, U=U, c=0.5, pi0=np.zeros(3), monotone=True)
    Wb, qb = robust_utility(box, pi)
    Wm, qm = robust_utility(mon, pi)
    print(f"  box  worst-case q = {np.round(qb, 3)}  W={Wb:.5f}")
    print(
        f"  mono worst-case q = {np.round(qm, 3)}  W={Wm:.5f}  (running-max of L => coupled)"
    )
    ok = (Wm > Wb + 1e-9) and np.allclose(qm, np.maximum.accumulate(L), atol=1e-6)
    return bool(ok)


def main() -> None:
    """Run the command-line entry point."""
    print("selective-labels core self-test\n")
    print("(A) regression vs 1B closed form (box mode):")
    a_ok = _regression_1b()
    print(f"    => {'PASS' if a_ok else 'FAIL'}\n")
    print("(B) monotone-coupling demonstration:")
    b_ok = _demo_monotone_coupling()
    print(f"    => {'PASS' if b_ok else 'FAIL'}\n")
    print(f"CORE CHECK: {'PASS' if (a_ok and b_ok) else 'FAIL'}")


if __name__ == "__main__":
    main()
