"""Evaluate the reveal controllability experiment. See Experiments and supplementary Certification at the Unbucketed River."""

from __future__ import annotations

import collections
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
import time
from typing import Any

from safe_observation.opponents import Opponent, holdem_equilibrium_opponent
from safe_observation.sequence_form import compile as compile_game
from safe_observation.solvers import robust_safe_response_linear, solve_blueprint
from scripts.poker.evaluate_public_routing import _build_gate_suite

GAME = sys.argv[1] if len(sys.argv) > 1 else "holdem_tr_b2"
RHO = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
NPROC = int(sys.argv[3]) if len(sys.argv) > 3 else 12
MASS_TOL = 1e-4
SHORT_TOL = 1e-5
OPPONENTS = ["river_overfold_strong", "turn_overfold_w50", "revealed_call_strong"]
AUDIT = Path(f"results/residual_audit_{GAME}.cells.jsonl")
OUT = Path(f"results/reveal_controllability_{GAME}.json")
CKPT = Path(f"results/reveal_controllability_{GAME}.cells.jsonl")

_W: dict[str, Any] = {}


def _fold_indices(sf1) -> dict[str, int]:
    """Compute fold indices for the evaluate reveal controllability workflow."""
    out: dict[str, int] = {}
    for info in sf1.info_sets:
        acts = [a for a, _ in info.children]
        if "f" in acts:
            out[info.label] = acts.index("f")
    return out


def _line_of(label: str) -> str:
    """Compute line of for the evaluate reveal controllability workflow."""
    return label.split("|", 1)[1] if "|" in label else label


def _build() -> dict[str, Any]:
    """Build the configured game or confidence object."""
    sf1 = compile_game(GAME, 1)
    v_ref = solve_blueprint(GAME, method="lp").value
    fold_idx = _fold_indices(sf1)
    eqb = holdem_equilibrium_opponent(GAME).behavior
    actions = {i.label: [a for a, _ in i.children] for i in sf1.info_sets}
    suite = _build_gate_suite(GAME, actions, eqb)

    child2la: dict[int, tuple[str, str]] = {}
    for info in sf1.info_sets:
        for _a, (act, child) in enumerate(info.children):
            child2la[child] = (_line_of(info.label), act)
    audit = [json.loads(line) for line in AUDIT.open()]
    Vaud = {r["opp"]: r["R"] for r in audit if r["kind"] == "V"}
    single: dict[str, list[tuple[int, float]]] = collections.defaultdict(list)
    for r in audit:
        if r["kind"] == "single" and r.get("R") is not None:
            single[r["opp"]].append((int(r["payload"]), Vaud[r["opp"]] - r["R"]))

    opp: dict[str, Any] = {}
    for name in OPPONENTS:
        beh = suite[name]
        y_star = list(Opponent(name=name, behavior=beh, game=GAME).realization())

        child_val: dict[int, float] = {}
        line_children: dict[tuple[str, str], list[int]] = collections.defaultdict(list)
        for info in sf1.info_sets:
            fi = fold_idx.get(info.label)
            if y_star[info.parent_seq] <= MASS_TOL:
                continue
            for _a, (act, child) in enumerate(info.children):
                if fi is not None and _a == fi:
                    continue
                child_val[child] = y_star[child]
                line_children[(_line_of(info.label), act)].append(child)

        agg: dict[tuple[str, str], float] = collections.defaultdict(float)
        for c, s in single[name]:
            if s > SHORT_TOL:
                agg[child2la[c]] += s
        ranked = [la for la, _ in sorted(agg.items(), key=lambda x: -x[1])]
        opp[name] = {
            "y_star": y_star,
            "child_val": child_val,
            "line_children": {k: v for k, v in line_children.items()},
            "ranked_lines": ranked,
            "agg_shortfall": dict(agg),
        }
    return {"sf1_n": len(sf1.info_sets), "v_ref": v_ref, "opp": opp}


def _init_worker() -> None:
    """Initialize process-local state for a parallel worker."""
    if os.environ.get("SAFE_OBSERVATION_HIGHS_THREADS") is None:
        os.environ["SAFE_OBSERVATION_HIGHS_THREADS"] = "1"
    _W.update(_build())


def _solve(pins: list[tuple[int, float]]) -> float:
    """Solve the configured optimization problem."""
    entries: list[tuple[int, int, float]] = []
    h: list[float] = []
    meta: list[tuple[str, int]] = []
    row = 0
    for child, val in pins:
        entries.append((row, child, 1.0))
        h.append(val)
        meta.append((f"pin{child}", 0))
        row += 1
        entries.append((row, child, -1.0))
        h.append(-val)
        meta.append((f"pin{child}", 0))
        row += 1
    res = robust_safe_response_linear(
        {}, {}, entries, h, v_ref=_W["v_ref"], eps_safe=RHO, game=GAME, row_meta=meta
    )
    return float(res.robust_value)


def _cell(task: tuple[str, int]) -> dict[str, Any]:
    """Run one independently reproducible experiment cell."""
    name, m = task
    rec = _W["opp"][name]
    child_val: dict[int, float] = rec["child_val"]
    t0 = time.time()
    if m == -1:
        pins = [(c, v) for c, v in child_val.items()]
    elif m == 0:
        pins = []
    else:
        keep: set[int] = set()
        for la in rec["ranked_lines"][:m]:
            keep.update(rec["line_children"][la])
        pins = [(c, child_val[c]) for c in keep]
    R = _solve(pins)
    return {
        "opp": name,
        "m": m,
        "R": R,
        "n_pins": len(pins),
        "wall_s": round(time.time() - t0, 2),
    }


def main() -> None:
    """Run the command-line entry point."""
    t0 = time.time()
    meta = _build()
    tasks: list[tuple[str, int]] = []
    for name in OPPONENTS:
        nlines = len(meta["opp"][name]["ranked_lines"])
        ms = sorted({0, 1, 2, 3, 4, 5, 6, 8, 10, 12, nlines, -1} | {nlines})
        ms = [m for m in ms if m == -1 or m == 0 or m <= nlines]
        for m in ms:
            tasks.append((name, m))

    done: set[tuple[str, int]] = set()
    raw: list[dict] = []
    if CKPT.exists():
        for line in CKPT.open():
            if line.strip():
                r = json.loads(line)
                raw.append(r)
                done.add((r["opp"], r["m"]))
    todo = [t for t in tasks if t not in done]
    nproc = min(len(todo), NPROC) if todo else 1
    print(
        f"reveal controllability  game={GAME} rho={RHO}  tasks={len(tasks)} todo={len(todo)} "
        f"nproc={nproc}",
        flush=True,
    )
    for name in OPPONENTS:
        print(
            f"  {name}: {len(meta['opp'][name]['ranked_lines'])} value-relevant lines",
            flush=True,
        )

    if todo:
        ctx = mp.get_context("spawn")
        with (
            CKPT.open("a") as fh,
            ctx.Pool(nproc, initializer=_init_worker) as pool,
        ):
            for cell in pool.imap_unordered(_cell, todo):
                fh.write(json.dumps(cell) + "\n")
                fh.flush()
                raw.append(cell)
                print(
                    f"  done {cell['opp']:22s} m={cell['m']:>3} "
                    f"R={cell['R']:+.4f} pins={cell['n_pins']} wall={cell['wall_s']}s",
                    flush=True,
                )

    vref = meta["v_ref"]
    curves: dict[str, Any] = {}
    for name in OPPONENTS:
        cells = {c["m"]: c["R"] for c in raw if c["opp"] == name}
        V = cells.get(-1)
        none = cells.get(0)
        nlines = len(meta["opp"][name]["ranked_lines"])
        span = (V - none) if (V is not None and none is not None) else None
        pts = []
        for m in sorted(k for k in cells if k >= 0):
            R = cells[m]
            frac = (R - none) / span if span and span > 1e-9 else 0.0
            pts.append({"m": m, "R": R, "recovered_frac": frac})
        curves[name] = {
            "V": V,
            "none": none,
            "vref": vref,
            "n_value_relevant_lines": nlines,
            "points": pts,
        }
        print(f"\n## {name}  V={V:+.4f} none={none:+.4f} VR-lines={nlines}")
        for p in pts:
            print(
                f"   m={p['m']:>3} lines forced -> R={p['R']:+.4f}  "
                f"recovered={100 * p['recovered_frac']:5.1f}% of span"
            )

    OUT.write_text(
        json.dumps(
            {"game": GAME, "rho": RHO, "curves": curves, "wall_s": time.time() - t0},
            indent=2,
        )
    )
    print(f"\nwrote {OUT}  wall={time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
