"""Run the public twin sweep experiment. See Experiments and supplementary Certification at the Unbucketed River."""

from __future__ import annotations

from concurrent.futures import TimeoutError as FuturesTimeoutError, as_completed
import json
import multiprocessing as mp
import os
import random
import sys
import time

os.environ.setdefault("SAFE_OBSERVATION_HIGHS_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

from pebble import ProcessPool

from safe_observation.opponents import Opponent, holdem_equilibrium_opponent
from safe_observation.sequence_form import compile as compile_game
from safe_observation.solvers import (
    safety_constrained_best_response,
    solve_blueprint,
)
from scripts.poker.construct_public_twins import _perturbed, _pick_targets, _tv_public

GAME = sys.argv[1] if len(sys.argv) > 1 else "holdem_tr_b2"
N_WORKERS = int(sys.argv[2]) if len(sys.argv) > 2 else 8
RHOS = [float(r) for r in os.environ.get("GS_RHOS", "0.1,0.3,0.5").split(",")]
DELTA = 0.05
N_PERTURB = int(os.environ.get("GS_N_PERTURB", "15"))
N_LEAKY = int(os.environ.get("GS_N_LEAKY", "15"))
MAX_TARGETS = int(os.environ.get("GS_MAX_TARGETS", "8"))
PERTURB_SIGMA = 0.06
TOL = 1e-3


LP_TIMEOUT = float(os.environ.get("GS_LP_TIMEOUT", "90"))
SEED = 2026


_SF = None
_VREF = None
_BASES: list = []
_TARGETS: list = []


def _simplex_noise(dist, sigma, rng, floor=1e-3):
    """Compute simplex noise for the run public twin sweep workflow."""
    new = [p * (2.718281828 ** rng.gauss(0.0, sigma)) for p in dist]
    new = [max(floor, v) for v in new]
    s = sum(new)
    return [v / s for v in new]


def _perturb_equilibrium(eq, sigma, rng):
    """Compute perturb equilibrium for the run public twin sweep workflow."""
    return {label: _simplex_noise(dist, sigma, rng) for label, dist in eq.items()}


def _random_leak(eq, actions, rng):
    """Compute random leak for the run public twin sweep workflow."""
    river = rng.random() < 0.5
    rank_thr = rng.choice([0, 7, 9, 11])
    weight = rng.uniform(0.3, 0.6)
    target_act = "f" if rng.random() < 0.7 else "c"
    out = {label: list(dist) for label, dist in eq.items()}
    for label, dist in eq.items():
        hole, hist = label.split("|", 1)
        acts = actions[label]
        if target_act not in acts:
            continue
        is_river = "/" in hist
        if is_river != river:
            continue
        if max(_RANK[hole[0]], _RANK[hole[2]]) < rank_thr:
            continue
        ti = acts.index(target_act)
        new = [p * (1.0 - weight) for p in dist]
        new[ti] += weight
        s = sum(new)
        out[label] = [p / s for p in new] if s > 1e-12 else list(dist)
    return out


_RANK = {r: i for i, r in enumerate("23456789TJQKA")}


def _masses(behavior, targets):
    """Compute masses for the run public twin sweep workflow."""
    yr = list(Opponent(name="b", behavior=behavior, game=GAME).realization())
    out = []
    for _hist, i1, i2 in targets:
        out.append((yr[i1.parent_seq], yr[i2.parent_seq]))
    return out


def _g(behavior, rho):
    """Compute g for the run public twin sweep workflow."""
    return (
        safety_constrained_best_response(
            behavior, v_ref=_VREF, eps_safe=rho, game=GAME
        ).value
        - _VREF
    )


def _init(sf, vref, bases, targets):
    """Initialize process-local state for parallel experiment workers."""
    global _SF, _VREF, _BASES, _TARGETS
    _SF, _VREF, _BASES, _TARGETS = sf, vref, bases, targets


def _task(key):
    """Execute one independently reproducible parallel task."""
    if key[0] == "gbar":
        _kind, base_id, rho = key
        _bid, _ens, _label, behavior, _m = _BASES[base_id]
        return (key, _g(behavior, rho), None)
    _kind, base_id, rho, t_idx, side = key
    _bid, _ens, _label, behavior, masses_list = _BASES[base_id]
    _hist, i1, i2 = _TARGETS[t_idx]
    m1, m2 = masses_list[t_idx]
    eps = DELTA * min(m1, m2)
    beh = _perturbed(behavior, i1, m1, i2, m2, eps, +1.0 if side > 0 else -1.0)
    if beh is None:
        return (key, None, eps)
    return (key, _g(beh, rho), eps)


def main() -> None:
    """Run the command-line entry point."""
    global _SF, _VREF, _BASES, _TARGETS
    t0 = time.time()
    rng = random.Random(SEED)
    sf = compile_game(GAME, 1)
    vref = solve_blueprint(GAME, method="lp").value
    eq = holdem_equilibrium_opponent(GAME).behavior
    actions = {info.label: [a for a, _ in info.children] for info in sf.info_sets}
    _SF, _VREF = sf, vref

    yr_eq = list(Opponent(name="eq", behavior=eq, game=GAME).realization())
    cands = _pick_targets(sf, eq, yr_eq)
    targets = []
    for _gap, _depth, hist, i1, m1, i2, m2 in cands:
        epsp = 0.05 * min(m1, m2)
        bp = _perturbed(eq, i1, m1, i2, m2, epsp, +1.0)
        bm = _perturbed(eq, i1, m1, i2, m2, epsp, -1.0)
        if bp is None or bm is None:
            continue
        yrp = list(Opponent(name="p", behavior=bp, game=GAME).realization())
        yrm = list(Opponent(name="m", behavior=bm, game=GAME).realization())
        if _tv_public(sf, yrp, yrm) > 1e-9:
            continue

        targets.append((hist, i1, i2))
    targets = targets[:MAX_TARGETS]

    bases = []
    bases.append((0, "E0_eq", "equilibrium", eq, _masses(eq, targets)))
    for k in range(N_PERTURB):
        b = _perturb_equilibrium(eq, PERTURB_SIGMA, rng)
        bases.append((len(bases), "EA_perturb", f"perturb{k}", b, _masses(b, targets)))
    for k in range(N_LEAKY):
        b = _random_leak(eq, actions, rng)
        bases.append((len(bases), "EB_leaky", f"leaky{k}", b, _masses(b, targets)))

    _BASES, _TARGETS = bases, targets
    print(
        f"# gamma sweep on {GAME}: v_ref={vref:.4f}, {len(targets)} twin targets, "
        f"{len(bases)} bases ({N_PERTURB} perturb + {N_LEAKY} leaky), rhos={RHOS}",
        flush=True,
    )

    tasks = [("gbar", b[0], rho) for b in bases for rho in RHOS]
    tasks += [
        ("kink", b[0], rho, ti, side)
        for b in bases
        for rho in RHOS
        for ti in range(len(targets))
        for side in (+1, -1)
    ]
    n_lp = len(tasks)
    print(
        f"# {n_lp} LP solves on {N_WORKERS} workers, per-LP timeout {LP_TIMEOUT:.0f}s",
        flush=True,
    )

    os.makedirs("results", exist_ok=True)
    ckpt_path = f"results/gamma_sweep_{GAME}.jsonl"
    results = {}
    if os.path.exists(ckpt_path):
        with open(ckpt_path) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                results[tuple(rec["k"])] = (rec["g"], rec["eps"])
        print(f"# resumed {len(results)} cached LPs from {ckpt_path}", flush=True)
    todo = [t for t in tasks if t not in results]

    n_timeout = 0
    done = len(results)
    ctx = mp.get_context("spawn")
    with open(ckpt_path, "a") as ckpt_fh:
        with ProcessPool(
            max_workers=N_WORKERS,
            initializer=_init,
            initargs=(sf, vref, bases, targets),
            context=ctx,
        ) as pool:
            futures = {
                pool.schedule(_task, args=(key,), timeout=LP_TIMEOUT): key
                for key in todo
            }
            for fut in as_completed(futures):
                key = futures[fut]
                try:
                    _key, g, eps = fut.result()
                except FuturesTimeoutError:
                    g, eps, n_timeout = None, None, n_timeout + 1
                except Exception:
                    g, eps = None, None
                results[key] = (g, eps)
                ckpt_fh.write(json.dumps({"k": list(key), "g": g, "eps": eps}) + "\n")
                ckpt_fh.flush()
                done += 1
                if done % 25 == 0:
                    print(
                        f"  {done}/{n_lp} LPs  ({time.time() - t0:.0f}s, "
                        f"{n_timeout} timeouts)",
                        flush=True,
                    )

    gbar = {(k[1], k[2]): v[0] for k, v in results.items() if k[0] == "gbar"}
    kink = {(k[1], k[2], k[3], k[4]): v for k, v in results.items() if k[0] == "kink"}

    rows = []
    for bid, ens, label, _beh, _m in bases:
        for rho in RHOS:
            gb = gbar.get((bid, rho))
            for ti, (hist, _i1, _i2) in enumerate(targets):
                gp, eps = kink.get((bid, rho, ti, +1), (None, None))
                gm, _ = kink.get((bid, rho, ti, -1), (None, None))
                if gb is None or gp is None or gm is None or not eps:
                    rows.append(
                        {
                            "base": label,
                            "ensemble": ens,
                            "rho": rho,
                            "target": hist,
                            "c_eps": None,
                            "boundary": True,
                        }
                    )
                    continue
                c_eps = (gp + gm - 2 * gb) / (2 * eps)
                rows.append(
                    {
                        "base": label,
                        "ensemble": ens,
                        "rho": rho,
                        "target": hist,
                        "g_base": gb,
                        "c_eps": c_eps,
                        "boundary": False,
                    }
                )

    summary = {}
    for ens in ("E0_eq", "EA_perturb", "EB_leaky"):
        for rho in RHOS:
            vals = [
                r["c_eps"]
                for r in rows
                if r["ensemble"] == ens and r["rho"] == rho and not r["boundary"]
            ]
            n_bd = sum(
                1
                for r in rows
                if r["ensemble"] == ens and r["rho"] == rho and r["boundary"]
            )
            if vals:
                vals_sorted = sorted(vals)
                pos = sum(1 for v in vals if v > TOL)
                summary[f"{ens}|{rho}"] = {
                    "n": len(vals),
                    "n_boundary": n_bd,
                    "frac_pos": pos / len(vals),
                    "median_c_eps": vals_sorted[len(vals_sorted) // 2],
                    "max_c_eps": vals_sorted[-1],
                    "min_c_eps": vals_sorted[0],
                }

    out = {
        "game": GAME,
        "v_ref": vref,
        "rhos": RHOS,
        "delta": DELTA,
        "tol": TOL,
        "n_perturb": N_PERTURB,
        "n_leaky": N_LEAKY,
        "perturb_sigma": PERTURB_SIGMA,
        "seed": SEED,
        "targets": [t[0] for t in targets],
        "summary": summary,
        "rows": rows,
        "n_timeout": n_timeout,
        "lp_timeout_s": LP_TIMEOUT,
        "elapsed_s": time.time() - t0,
    }
    os.makedirs("results", exist_ok=True)
    path = f"results/gamma_sweep_{GAME}.json"
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2)

    print(f"\n# Fraction with c_eps>{TOL} (kink positive), by ensemble x rho:")
    print(
        f"{'ensemble':<12}{'rho':>5}{'n':>5}{'bd':>4}{'frac+':>8}{'median':>9}{'max':>9}"
    )
    for ens in ("E0_eq", "EA_perturb", "EB_leaky"):
        for rho in RHOS:
            s = summary.get(f"{ens}|{rho}")
            if s:
                print(
                    f"{ens:<12}{rho:>5.1f}{s['n']:>5}{s['n_boundary']:>4}"
                    f"{s['frac_pos']:>8.2f}{s['median_c_eps']:>9.4f}{s['max_c_eps']:>9.4f}"
                )
    print(f"\n# wrote {path}  ({time.time() - t0:.0f}s total)")


if __name__ == "__main__":
    main()
