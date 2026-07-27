"""Evaluate the reveal necessity experiment. See supplementary Additional Experiments."""

import json
import os
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import linprog

RHO = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5


M = {"H": np.array([+1.0, -1.0, 0.0]), "L": np.array([-1.0, +1.0, 0.0])}
TWINS = {
    "y_A": {("A", "H"): 0.5, ("B", "L"): 0.5},
    "y_B": {("B", "H"): 0.5, ("A", "L"): 0.5},
}


def _public_obs(y: dict) -> np.ndarray:
    """Compute public obs for the evaluate reveal necessity workflow."""
    p = {"A": 0.0, "B": 0.0}
    for (a, _t), m in y.items():
        p[a] += m
    return np.array([p["A"], p["B"]])


def _reveal_obs(y: dict) -> np.ndarray:
    """Compute reveal obs for the evaluate reveal necessity workflow."""
    keys = [("A", "H"), ("A", "L"), ("B", "H"), ("B", "L")]
    return np.array([y.get(k, 0.0) for k in keys])


def _l1_gap(u: np.ndarray, v: np.ndarray) -> float:
    """Compute l1 gap for the evaluate reveal necessity workflow."""
    return float(np.abs(u - v).sum() / 2.0)


def _robust_public() -> float:
    """Compute robust public for the evaluate reveal necessity workflow."""
    nx = 9
    c = np.zeros(nx)
    c[6] = -1.0
    A_ub: list[np.ndarray] = []
    b_ub: list[float] = []

    def dA(k: float) -> np.ndarray:
        """Compute d a for the evaluate reveal necessity workflow."""
        r = np.zeros(nx)
        r[0] += k
        r[1] -= k
        return r

    def dB(k: float) -> np.ndarray:
        """Compute d b for the evaluate reveal necessity workflow."""
        r = np.zeros(nx)
        r[3] += k
        r[4] -= k
        return r

    r = np.zeros(nx)
    r[6] = 1.0
    r += dA(-0.5) + dB(0.5)
    A_ub.append(r)
    b_ub.append(0.0)
    r = np.zeros(nx)
    r[6] = 1.0
    r += dB(-0.5) + dA(0.5)
    A_ub.append(r)
    b_ub.append(0.0)
    for f in (dA, dB):
        r = np.zeros(nx)
        r[7] = 1.0
        r += f(-1.0)
        A_ub.append(r)
        b_ub.append(0.0)
    for f in (dA, dB):
        r = np.zeros(nx)
        r[8] = 1.0
        r += f(1.0)
        A_ub.append(r)
        b_ub.append(0.0)
    r = np.zeros(nx)
    r[7] = -0.5
    r[8] = -0.5
    A_ub.append(r)
    b_ub.append(RHO)
    A_eq = [[1, 1, 1, 0, 0, 0, 0, 0, 0], [0, 0, 0, 1, 1, 1, 0, 0, 0]]
    res = linprog(
        c,
        A_ub=np.array(A_ub),
        b_ub=np.array(b_ub),
        A_eq=A_eq,
        b_eq=[1.0, 1.0],
        bounds=[(0, 1)] * 6 + [(-1, 1)] * 3,
        method="highs",
    )
    return float(res.x[6])


def _oracle_public(twin: str) -> float:
    """Compute oracle public for the evaluate reveal necessity workflow."""
    nx = 8
    c = np.zeros(nx)
    if twin == "y_A":
        c[0], c[1], c[3], c[4] = -0.5, 0.5, 0.5, -0.5
    else:
        c[0], c[1], c[3], c[4] = 0.5, -0.5, -0.5, 0.5
    A_ub: list[np.ndarray] = []
    b_ub: list[float] = []

    def dA(k: float) -> np.ndarray:
        """Compute d a for the evaluate reveal necessity workflow."""
        r = np.zeros(nx)
        r[0] += k
        r[1] -= k
        return r

    def dB(k: float) -> np.ndarray:
        """Compute d b for the evaluate reveal necessity workflow."""
        r = np.zeros(nx)
        r[3] += k
        r[4] -= k
        return r

    for f in (dA, dB):
        r = np.zeros(nx)
        r[6] = 1.0
        r += f(-1.0)
        A_ub.append(r)
        b_ub.append(0.0)
    for f in (dA, dB):
        r = np.zeros(nx)
        r[7] = 1.0
        r += f(1.0)
        A_ub.append(r)
        b_ub.append(0.0)
    r = np.zeros(nx)
    r[6] = -0.5
    r[7] = -0.5
    A_ub.append(r)
    b_ub.append(RHO)
    A_eq = [[1, 1, 1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 1, 1, 0, 0]]
    res = linprog(
        c,
        A_ub=np.array(A_ub),
        b_ub=np.array(b_ub),
        A_eq=A_eq,
        b_eq=[1.0, 1.0],
        bounds=[(0, 1)] * 6 + [(-1, 1)] * 2,
        method="highs",
    )
    return float(-res.fun)


def main() -> None:
    """Run the command-line entry point."""
    nrc = {
        "obs_gap": _l1_gap(_public_obs(TWINS["y_A"]), _public_obs(TWINS["y_B"])),
        "oracle": min(_oracle_public("y_A"), _oracle_public("y_B")),
        "robust": _robust_public(),
    }
    nrc["value_gap"] = nrc["oracle"] - nrc["robust"]

    rc = {
        "obs_gap": _l1_gap(_reveal_obs(TWINS["y_A"]), _reveal_obs(TWINS["y_B"])),
        "oracle": 1.0,
        "robust": 1.0,
        "value_gap": 0.0,
    }

    out = {"rho": RHO, "non_reveal_controllable": nrc, "reveal_controllable": rc}
    print(f"# reveal-controllability necessity  rho={RHO}")
    for name, d in (("non-reveal-controllable", nrc), ("reveal-controllable", rc)):
        print(
            f"  {name:<26} obs_gap={d['obs_gap']:+.3f}  "
            f"oracle={d['oracle']:+.3f}  robust={d['robust']:+.3f}  "
            f"value_gap={d['value_gap']:+.3f}"
        )
    path = Path(
        os.environ.get("RNC_OUT", "results/reveal_controllability_counterexample.json")
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
