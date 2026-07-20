""

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import linprog

GAME_TAG = os.environ.get("SAB_TAG", "single_action")
W1 = float(os.environ.get("SAB_W1", "0.5"))
BETA = float(os.environ.get("SAB_BETA", "0.3"))
PHI = float(os.environ.get("SAB_PHI", "0.0"))
RHO = float(os.environ.get("SAB_RHO", "1.0"))
N_DELTA = int(os.environ.get("SAB_NDELTA", "40"))
OUT = Path(
    os.environ.get(
        "SINGLE_ACTION_BOUND_OUT", f"results/single_action_bound_{GAME_TAG}.json"
    )
)


M = np.array(
    [float(x) for x in os.environ.get("SAB_M", "1,-1,-1,1").split(",")]
).reshape(2, 2)


def _payoff_columns() -> np.ndarray:
    ""

    return np.array(
        [[M[a, 0], PHI, M[a, 1], PHI] for a in range(M.shape[0])], dtype=float
    )


def _y_star(w1: float, beta: float, delta: float) -> np.ndarray:
    w2 = 1.0 - w1
    return np.array(
        [w1 * (1.0 - beta - delta), w1 * (beta + delta), w2 * (1.0 - beta), w2 * beta]
    )


def _y_vertices_full(w1: float) -> list[np.ndarray]:
    ""
    w2 = 1.0 - w1
    verts = []
    for c1 in (0.0, w1):
        for c2 in (0.0, w2):
            verts.append(np.array([c1, w1 - c1, c2, w2 - c2]))
    return verts


def _cpub_vertices(w1: float, beta: float, delta: float) -> list[np.ndarray]:
    ""
    w2 = 1.0 - w1
    y = _y_star(w1, beta, delta)
    c_star = y[0] + y[2]
    lo = max(0.0, c_star - w2)
    hi = min(w1, c_star)
    verts = []
    for y1c in (lo, hi):
        y2c = c_star - y1c
        verts.append(np.array([y1c, w1 - y1c, y2c, w2 - y2c]))
    return verts


def _maximin(P_cols: np.ndarray, safety_cols: np.ndarray, floor: float) -> np.ndarray:
    ""
    m, nc = P_cols.shape
    ns = safety_cols.shape[1]

    c = np.zeros(m + 1)
    c[-1] = -1.0

    A_ub = []
    b_ub = []
    for j in range(nc):
        rowj = [-P_cols[a, j] for a in range(m)] + [1.0]
        A_ub.append(rowj)
        b_ub.append(0.0)

    for u in range(ns):
        rowu = [-safety_cols[a, u] for a in range(m)] + [0.0]
        A_ub.append(rowu)
        b_ub.append(-floor)
    A_eq = [[1.0] * m + [0.0]]
    b_eq = [1.0]
    bounds = [(0.0, 1.0)] * m + [(None, None)]
    res = linprog(
        c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs"
    )
    if not res.success:
        raise RuntimeError(f"maximin LP failed: {res.message}")
    return np.array(res.x[:m])


def _best_response(g: np.ndarray, safety_cols: np.ndarray, floor: float) -> np.ndarray:
    ""
    m = g.shape[0]
    ns = safety_cols.shape[1]
    c = -g
    A_ub = [[-safety_cols[a, u] for a in range(m)] for u in range(ns)]
    b_ub = [-floor] * ns
    A_eq = [[1.0] * m]
    b_eq = [1.0]
    bounds = [(0.0, 1.0)] * m
    res = linprog(
        c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs"
    )
    if not res.success:
        raise RuntimeError(f"BR LP failed: {res.message}")
    return np.array(res.x)


def _v_ref(A: np.ndarray, w1: float, beta: float) -> float:
    ""
    safety_cols = np.column_stack([A @ u for u in _y_vertices_full(w1)])
    x_sec = _maximin(safety_cols, safety_cols, floor=-1e9)
    g_eq = A @ _y_star(w1, beta, 0.0)
    return float(x_sec @ g_eq)


def main() -> None:
    w1 = W1
    A = _payoff_columns()
    safety_cols = np.column_stack([A @ u for u in _y_vertices_full(w1)])
    v_ref = _v_ref(A, w1, BETA)
    floor = v_ref - RHO
    delta_max = min(0.95 - BETA, 1.0 - BETA)
    deltas = np.linspace(delta_max / N_DELTA, delta_max, N_DELTA)

    print(
        f"# single-censored-action bound  w1={w1}  beta={BETA}  rho={RHO}  "
        f"v_ref={v_ref:+.4f}  floor={floor:+.4f}  delta_max={delta_max:.3f}",
        flush=True,
    )
    print(f"  {'delta':>7} {'D=w1*delta':>10} {'Delta':>9} {'ratio':>8}", flush=True)

    rows: list[dict[str, Any]] = []
    for delta in deltas:
        y_star = _y_star(w1, BETA, delta)
        g_star = A @ y_star

        cpub_cols = np.column_stack([A @ v for v in _cpub_vertices(w1, BETA, delta)])
        x_core = _maximin(cpub_cols, safety_cols, floor)
        realized_core = float(x_core @ g_star) - v_ref

        x_obs = _best_response(g_star, safety_cols, floor)
        realized_obs = float(x_obs @ g_star) - v_ref

        delta_gap = realized_obs - realized_core
        d_censored = w1 * delta
        ratio = delta_gap / d_censored if d_censored > 1e-12 else 0.0
        rows.append(
            {
                "delta": float(delta),
                "D_censored": float(d_censored),
                "Delta_gap": float(delta_gap),
                "ratio": float(ratio),
                "realized_core": realized_core,
                "realized_obs": realized_obs,
            }
        )
        print(
            f"  {delta:>7.3f} {d_censored:>10.4f} {delta_gap:>+9.4f} {ratio:>8.4f}",
            flush=True,
        )

    ratios = [r["ratio"] for r in rows if r["D_censored"] > 1e-9]
    c1 = min(ratios) if ratios else 0.0
    c2 = max(ratios) if ratios else 0.0

    sandwich_ok = all(
        c1 * r["D_censored"] - 1e-9 <= r["Delta_gap"] <= c2 * r["D_censored"] + 1e-9
        for r in rows
    )
    summary = {
        "c1_lower_slope": c1,
        "c2_upper_slope": c2,
        "positive_lower_constant": c1 > 1e-6,
        "sandwich_holds": bool(sandwich_ok),
        "n_delta": len(rows),
    }
    print(
        f"\n  c1 (lower slope) = {c1:.4f}   c2 (upper slope) = {c2:.4f}   "
        f"positive_c1={summary['positive_lower_constant']}   "
        f"sandwich_holds={summary['sandwich_holds']}",
        flush=True,
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps(
            {
                "tag": GAME_TAG,
                "w1": w1,
                "beta": BETA,
                "phi": PHI,
                "rho": RHO,
                "M": M.tolist(),
                "v_ref": v_ref,
                "floor": floor,
                "summary": summary,
                "rows": rows,
            },
            indent=2,
        )
    )
    print(f"\nwrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
