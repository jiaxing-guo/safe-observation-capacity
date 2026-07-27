"""Run the selective labels deployment experiment. See supplementary Additional Experiments."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import math
from pathlib import Path
import sys
import time

import numpy as np

from scripts.controlled.selective_labels import (
    SLInstance,
    certified_safe_response,
    reach_probe,
    robust_optimal_value,
    robust_utility,
)

M = int(sys.argv[1]) if len(sys.argv) > 1 else 30
NPROC = int(sys.argv[2]) if len(sys.argv) > 2 else 8
C = 0.5
EPS = 0.05
DELTA = 0.1
RHOS = [0.0, 0.01, 0.05]
NGRID = [1_000, 10_000, 100_000]
SEEDS = list(range(2026, 2036))
LOG_N = 200_000
LOG_SEED = 2026
CACHE = Path("results/sl_deploy.cells.jsonl")
OUT = Path("results/sl_deploy.json")


_G: dict = {}


def _build_model():
    """Build model for the run selective labels deployment workflow."""
    m = M
    s = np.linspace(0.0, 1.0, m)
    p = np.full(m, 1.0 / m)
    gem = (s >= 0.40) & (s <= 0.66)
    high = s > 0.66
    qG = np.where(gem, 0.80, np.clip(0.20 + 0.75 * s, 0.0, 1.0))
    qB = np.where(gem, 0.30, np.clip(qG - 0.40, 0.0, 1.0))

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


def _historical_logs(model, n, seed):
    """Compute historical logs for the run selective labels deployment workflow."""
    rng = np.random.default_rng(seed)
    m = model["m"]
    z = rng.choice(m, size=n, p=model["p"])
    good = rng.random(n) < model["beta"][z]
    q = np.where(good, model["qG"][z], model["qB"][z])
    y = (rng.random(n) < q).astype(int)
    sel = np.where(good, model["selG"][z], model["selB"][z])
    a = (rng.random(n) < sel).astype(int)
    r = np.zeros(m)
    qobs = np.full(m, np.nan)
    nobs = np.zeros(m, dtype=int)
    for i in range(m):
        zi = z == i
        ni = int(zi.sum())
        if ni == 0:
            continue
        ai = a[zi]
        r[i] = ai.mean()
        nobs[i] = int(ai.sum())
        if ai.sum() > 0:
            qobs[i] = y[zi][ai == 1].mean()
    return {"r": r, "qobs": qobs, "nobs": nobs}


def _partial_id(logs):
    """Compute partial identifier for the run selective labels deployment workflow."""
    m = M
    r, qobs, nobs = logs["r"], logs["qobs"], logs["nobs"]
    L = np.zeros(m)
    U = np.ones(m)
    for i in range(m):
        if r[i] <= 0 or np.isnan(qobs[i]):
            continue
        hw = math.sqrt(math.log(2.0 / DELTA) / (2.0 * max(nobs[i], 1)))
        base = r[i] * qobs[i]
        L[i] = max(0.0, base - hw)
        U[i] = min(1.0, base + (1.0 - r[i]) + hw)
    return L, U


def _init():
    """Initialize process-local state for parallel experiment workers."""
    model = _build_model()
    logs = _historical_logs(model, LOG_N, LOG_SEED)
    L_P, U_P = _partial_id(logs)
    inst_P = SLInstance(
        p=model["p"],
        L=L_P,
        U=U_P,
        c=C,
        pi0=np.zeros(M),
        monotone=True,
        q_true=model["q_true"],
    )
    W_star = robust_optimal_value(inst_P)
    _G.update(
        model=model,
        L_P=L_P,
        U_P=U_P,
        inst_P=inst_P,
        W_star=W_star,
        qobs_hist=logs["qobs"],
        r_hist=logs["r"],
    )


def _passive_confidence(model, N, seed):
    """Construct confidence constraints for passive."""
    logs = _historical_logs(model, N, seed)
    m = model["m"]
    L = np.array(_G["L_P"])
    U = np.array(_G["U_P"])
    for i in range(m):
        nobs = logs["nobs"][i]
        if nobs >= 5 and not np.isnan(logs["qobs"][i]):
            hw = math.sqrt(math.log(2.0 / DELTA) / (2.0 * nobs))

            lo = max(L[i], logs["qobs"][i] - hw)
            hi = min(U[i], logs["qobs"][i] + hw)
            if lo <= hi:
                L[i], U[i] = lo, hi
    return L, U


def _active_confidence(model, rho, N, seed):
    """Construct confidence constraints for active."""
    rng = np.random.default_rng(seed + 777)
    inst_P = _G["inst_P"]
    m = model["m"]

    targets = list(np.where(_G["U_P"] > C)[0])
    pi_probe = reach_probe(inst_P, targets, rho, U0=_G["W_star"])
    L = np.array(_G["L_P"])
    U = np.array(_G["U_P"])
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


def _certify(model, C_L, C_U, rho):
    """Compute certify for the run selective labels deployment workflow."""
    pi, cert = certified_safe_response(_G["inst_P"], C_L, C_U, rho, U0=_G["W_star"])
    return pi, cert


def _realized(model, pi):
    """Evaluate a realization plan against the selected opponent."""
    return float(np.sum(model["p"] * pi * (model["q_true"] - C)))


def _coverage(model, C_L, C_U):
    """Compute coverage for the run selective labels deployment workflow."""
    q = model["q_true"]
    return bool(np.all((q >= C_L - 1e-9) & (q <= C_U + 1e-9)))


def _floor_violation(model, pi, rho):
    """Compute floor violation for the run selective labels deployment workflow."""
    inst_P = _G["inst_P"]
    W = robust_utility(inst_P, pi)[0]
    return bool(W < _G["W_star"] - rho - 1e-6)


def _cell(args):
    """Run one independently reproducible experiment cell."""
    method, rho, N, seed = args
    model = _G["model"]
    if method == "oracle":
        q = model["q_true"]
        pi, cert = _certify(model, q.copy(), q.copy(), rho)
        cov, viol = True, _floor_violation(model, pi, rho)
        real = _realized(model, pi)
    elif method == "passive":
        L, U = _passive_confidence(model, N, seed)
        pi, cert = _certify(model, L, U, rho)
        cov, viol = _coverage(model, L, U), _floor_violation(model, pi, rho)
        real = _realized(model, pi)
    else:
        L, U, _ = _active_confidence(model, rho, N, seed)
        pi, cert = _certify(model, L, U, rho)
        cov, viol = _coverage(model, L, U), _floor_violation(model, pi, rho)
        real = _realized(model, pi)
    return {
        "method": method,
        "rho": rho,
        "N": N,
        "seed": seed,
        "certified": cert,
        "realized": real,
        "coverage": int(cov),
        "violation": int(viol),
    }


def main() -> None:
    """Run the command-line entry point."""
    t0 = time.time()
    _init()
    print(f"# Safe active de-censoring deployment ladder (lending, m={M})")
    print(f"# W*={_G['W_star']:.4f}  rhos={RHOS}  N={NGRID}  seeds={len(SEEDS)}\n")

    cells = [("oracle", r, 0, SEEDS[0]) for r in RHOS]
    cells += [
        (meth, r, n, s)
        for meth in ("passive", "active")
        for r in RHOS
        for n in NGRID
        for s in SEEDS
    ]
    done = {}
    if CACHE.exists():
        for line in CACHE.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                done[(row["method"], row["rho"], row["N"], row["seed"])] = row
    todo = [c for c in cells if c not in done]
    print(f"cells: {len(cells)} total, {len(done)} cached, {len(todo)} to run\n")

    CACHE.parent.mkdir(exist_ok=True)
    rows = list(done.values())
    if todo:
        with (
            ProcessPoolExecutor(max_workers=NPROC, initializer=_init) as ex,
            CACHE.open("a") as fh,
        ):
            futs = {ex.submit(_cell, c): c for c in todo}
            n_done = 0
            for fut in as_completed(futs):
                row = fut.result()
                fh.write(json.dumps(row) + "\n")
                fh.flush()
                rows.append(row)
                n_done += 1
                if n_done % 20 == 0:
                    print(f"  {n_done}/{len(todo)}")

    agg = {}
    for r in rows:
        key = f"{r['method']}|{r['rho']}|{r['N']}"
        agg.setdefault(key, []).append(r)
    summary = {}
    for key, rs in agg.items():
        summary[key] = {
            "certified": float(np.mean([x["certified"] for x in rs])),
            "realized": float(np.mean([x["realized"] for x in rs])),
            "coverage": float(np.mean([x["coverage"] for x in rs])),
            "violations": int(sum(x["violation"] for x in rs)),
            "n_cells": len(rs),
        }
    total_viol = sum(v["violations"] for v in summary.values())

    print(
        f"\n{'method':>8} {'rho':>6} {'N':>7} {'cert':>8} {'real':>8} {'cover':>7} {'viol':>5}"
    )
    for meth in ("passive", "active"):
        for rho in RHOS:
            for n in NGRID:
                k = f"{meth}|{rho}|{n}"
                if k in summary:
                    s = summary[k]
                    print(
                        f"{meth:>8} {rho:>6.2f} {n:>7} {s['certified']:>8.4f} "
                        f"{s['realized']:>8.4f} {s['coverage']:>7.2f} {s['violations']:>5}"
                    )
    for rho in RHOS:
        k = f"oracle|{rho}|0"
        if k in summary:
            s = summary[k]
            print(
                f"{'oracle':>8} {rho:>6.2f} {'-':>7} {s['certified']:>8.4f} "
                f"{s['realized']:>8.4f} {s['coverage']:>7.2f} {s['violations']:>5}"
            )
    print(f"\nTOTAL FLOOR VIOLATIONS across all cells: {total_viol}  (expect 0)")

    out = {
        "m": M,
        "c": C,
        "eps": EPS,
        "delta": DELTA,
        "W_star": _G["W_star"],
        "rhos": RHOS,
        "n_grid": NGRID,
        "seeds": SEEDS,
        "total_violations": total_viol,
        "summary": summary,
        "elapsed_s": time.time() - t0,
    }
    with OUT.open("w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nwrote {OUT}  ({time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main()
