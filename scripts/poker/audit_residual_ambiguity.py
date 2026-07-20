""

from __future__ import annotations

from concurrent.futures import TimeoutError as FuturesTimeoutError, as_completed
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
import time
from typing import Any

from pebble import ProcessPool

from safe_observation.opponents import Opponent, holdem_equilibrium_opponent
from safe_observation.sequence_form import compile as compile_game
from safe_observation.solvers import (
    robust_safe_response_linear,
    solve_blueprint,
)
from scripts.poker.evaluate_public_routing import _build_gate_suite

GAME = sys.argv[1] if len(sys.argv) > 1 else "holdem_tr_b2"
RHO = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
_pos = [a for a in sys.argv[3:] if not a.startswith("--")]
NPROC = int(_pos[0]) if len(_pos) > 0 else 12
CELL_TIMEOUT = float(_pos[1]) if len(_pos) > 1 else 600.0
COUNT_ONLY = "--count" in sys.argv


OPPONENTS = [
    "tr_equilibrium",
    "river_overfold_strong",
    "revealed_call_strong",
    "turn_overfold_w50",
]
TOPK = [1, 2, 3, 5, 8]
MASS_TOL = 1e-4
PIN_TOL = 1e-9
SHORT_TOL = 1e-5

_W: dict[str, Any] = {}


def _fold_indices(sf1) -> dict[str, int]:
    out: dict[str, int] = {}
    for info in sf1.info_sets:
        acts = [a for a, _ in info.children]
        if "f" in acts:
            out[info.label] = acts.index("f")
    return out


def _reached_continuations(sf1, fold_idx, y_star):
    ""
    audited = []
    n_total = 0
    n_flow_null = 0
    for info in sf1.info_sets:
        parent = y_star[info.parent_seq]
        fi = fold_idx.get(info.label)
        has_fold = fi is not None
        for a, (_act, child) in enumerate(info.children):
            if has_fold and a == fi:
                continue
            n_total += 1
            if parent <= MASS_TOL or not has_fold:
                n_flow_null += 1
                continue
            audited.append((info.label, a, child, y_star[child]))
    return audited, n_total, n_flow_null


def _pin_rows(pins):
    ""
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
    return entries, h, meta


def _init() -> None:
    if os.environ.get("SAFE_OBSERVATION_HIGHS_THREADS") is None:
        os.environ["SAFE_OBSERVATION_HIGHS_THREADS"] = "1"
    sf1 = compile_game(GAME, 1)
    v_ref = solve_blueprint(GAME, method="lp").value
    fold_idx = _fold_indices(sf1)
    eqb = holdem_equilibrium_opponent(GAME).behavior
    actions = {i.label: [a for a, _ in i.children] for i in sf1.info_sets}
    suite = _build_gate_suite(GAME, actions, eqb)
    opp: dict[str, Any] = {}
    for name in OPPONENTS:
        beh = suite[name]
        y_star = list(Opponent(name=name, behavior=beh, game=GAME).realization())
        audited, n_total, n_flow_null = _reached_continuations(sf1, fold_idx, y_star)

        child_val: dict[int, float] = {}
        for info in sf1.info_sets:
            fi = fold_idx.get(info.label)
            for a, (_act, child) in enumerate(info.children):
                if fi is not None and a == fi:
                    continue
                if y_star[info.parent_seq] > MASS_TOL:
                    child_val[child] = y_star[child]
        opp[name] = {
            "y_star": y_star,
            "audited": audited,
            "child_val": child_val,
            "n_total": n_total,
            "n_flow_null": n_flow_null,
        }
    _W.update(sf1=sf1, v_ref=v_ref, opp=opp)


def _solve(pins) -> float:
    entries, h, meta = _pin_rows(pins)
    res = robust_safe_response_linear(
        {}, {}, entries, h, v_ref=_W["v_ref"], eps_safe=RHO, game=GAME, row_meta=meta
    )
    return float(res.robust_value)


def _cell(task: tuple[str, str, str]) -> dict[str, Any]:
    ""
    name, kind, payload = task
    rec = _W["opp"][name]
    child_val: dict[int, float] = rec["child_val"]
    all_children = list(child_val)
    t0 = time.time()

    if kind == "V":
        pins = [(c, child_val[c]) for c in all_children]
    elif kind == "single":
        drop = int(payload)
        pins = [(c, child_val[c]) for c in all_children if c != drop]
    elif kind == "none":
        pins = []
    elif kind == "drop_top":
        drop = {int(x) for x in payload.split(",") if x}
        pins = [(c, child_val[c]) for c in all_children if c not in drop]
    elif kind == "keep_top":
        keep = {int(x) for x in payload.split(",") if x}
        pins = [(c, child_val[c]) for c in all_children if c in keep]
    else:
        raise ValueError(f"unknown kind {kind!r}")

    R = _solve(pins)
    return {
        "opp": name,
        "kind": kind,
        "payload": payload,
        "R": R,
        "n_pins": len(pins),
        "n_free": len(all_children) - len(pins),
        "wall_s": round(time.time() - t0, 2),
    }


def _key(task: tuple[str, str, str]) -> tuple[str, str, str]:
    return task


def _run_stage(pending, ckpt_fh, done, total, label):
    rows: list[dict[str, Any]] = []
    with ProcessPool(
        max_workers=NPROC, initializer=_init, context=mp.get_context("spawn")
    ) as pool:
        futures = {
            pool.schedule(_cell, args=(t,), timeout=CELL_TIMEOUT): t for t in pending
        }
        for fut in as_completed(futures):
            task = futures[fut]
            try:
                r = fut.result()
            except FuturesTimeoutError:
                r = {
                    "opp": task[0],
                    "kind": task[1],
                    "payload": task[2],
                    "R": None,
                    "n_pins": -1,
                    "n_free": -1,
                    "wall_s": CELL_TIMEOUT,
                    "status": "lp_timeout",
                }
            except Exception as exc:
                r = {
                    "opp": task[0],
                    "kind": task[1],
                    "payload": task[2],
                    "R": None,
                    "n_pins": -1,
                    "n_free": -1,
                    "wall_s": 0.0,
                    "status": f"error:{type(exc).__name__}",
                }
            rows.append(r)
            ckpt_fh.write(json.dumps(r) + "\n")
            ckpt_fh.flush()
            done += 1
            if done % 10 == 0:
                print(f"    ... [{label}] {done}/{total} cells", flush=True)
    return rows, done


def main() -> None:
    print(f"# Residual-width audit on {GAME}  rho={RHO}  nproc={NPROC}", flush=True)
    _init()
    v_ref = _W["v_ref"]
    opp = _W["opp"]
    print(f"v_ref={v_ref:.6f}  opponents={len(OPPONENTS)}", flush=True)
    for name in OPPONENTS:
        rec = opp[name]
        print(
            f"  {name:<24} non-fold cont: {rec['n_total']:>5}  "
            f"flow-null (no fold/zero-mass): {rec['n_flow_null']:>5}  "
            f"audited: {len(rec['audited']):>5}",
            flush=True,
        )

    stage1: list[tuple[str, str, str]] = []
    for name in OPPONENTS:
        stage1.append((name, "V", ""))
        stage1.append((name, "none", ""))
        for _lbl, _a, child, _val in opp[name]["audited"]:
            stage1.append((name, "single", str(child)))

    n1 = len(stage1)
    print(f"\nStage 1 cells (V + none + single-drop): {n1}", flush=True)
    if COUNT_ONLY:
        n2_est = len(OPPONENTS) * (2 * len(TOPK))
        print(f"Stage 2 cells (drop_top + keep_top, est): {n2_est}", flush=True)
        print(f"TOTAL est: {n1 + n2_est} cells", flush=True)
        return

    ckpt = Path(f"results/residual_audit_{GAME}.cells.jsonl")
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    done_keys: set[tuple[str, str, str]] = set()
    all_rows: list[dict[str, Any]] = []
    if ckpt.exists():
        for line in ckpt.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            all_rows.append(r)
            done_keys.add((r["opp"], r["kind"], r["payload"]))
        print(f"  resuming: {len(done_keys)} cells already checkpointed", flush=True)

    ckpt_fh = open(ckpt, "a")
    t_start = time.time()
    pending1 = [t for t in stage1 if _key(t) not in done_keys]
    rows1, done = _run_stage(
        pending1, ckpt_fh, len(all_rows), n1 + len(OPPONENTS) * 2 * len(TOPK), "S1"
    )
    all_rows.extend(rows1)
    print(f"  Stage 1 done in {time.time() - t_start:.0f}s", flush=True)

    V = {r["opp"]: r["R"] for r in all_rows if r["kind"] == "V" and r["R"] is not None}
    single: dict[str, list[tuple[float, int]]] = {n: [] for n in OPPONENTS}
    for r in all_rows:
        if r["kind"] == "single" and r["R"] is not None and r["opp"] in V:
            sh = V[r["opp"]] - r["R"]
            single[r["opp"]].append((sh, int(r["payload"])))
    for n in OPPONENTS:
        single[n].sort(reverse=True)

    stage2: list[tuple[str, str, str]] = []
    for name in OPPONENTS:
        ranked = [c for _sh, c in single[name]]
        for k in TOPK:
            if k > len(ranked):
                continue
            topk = ranked[:k]
            csv = ",".join(str(c) for c in topk)
            stage2.append((name, "drop_top", csv))
            stage2.append((name, "keep_top", csv))

    pending2 = [t for t in stage2 if _key(t) not in done_keys]
    total = len(all_rows) + len(pending2)
    print(
        f"\nStage 2 cells (multi-drop): {len(stage2)} ({len(pending2)} pending)",
        flush=True,
    )
    t2 = time.time()
    rows2, done = _run_stage(pending2, ckpt_fh, len(all_rows), total, "S2")
    all_rows.extend(rows2)
    ckpt_fh.close()
    print(f"  Stage 2 done in {time.time() - t2:.0f}s", flush=True)

    _report(all_rows, V, single)


def _report(all_rows, V, single) -> None:
    v_ref = _W["v_ref"]

    def by(opp, kind, payload=""):
        return next(
            (
                r
                for r in all_rows
                if r["opp"] == opp
                and r["kind"] == kind
                and r["payload"] == payload
                and r["R"] is not None
            ),
            None,
        )

    print(
        "\n== full river audit: residual safe-payoff width on ambiguous targets ==",
        flush=True,
    )
    print(
        f"{'opponent':<24}{'V-vref':>9}{'#cont':>7}{'#val-rel':>9}"
        f"{'maxΔ':>9}{'medΔ':>9}{'sumΔ':>9}{'Vrec%':>7}",
        flush=True,
    )
    for name in OPPONENTS:
        if name not in V:
            print(f"{name:<24}  (no V cell)", flush=True)
            continue
        shorts = [sh for sh, _c in single[name]]
        vr = [sh for sh in shorts if sh > SHORT_TOL]
        none_r = by(name, "none")
        R_none = none_r["R"] if none_r else float("nan")
        expl = V[name] - v_ref
        maxd = max(shorts) if shorts else 0.0
        med = sorted(shorts)[len(shorts) // 2] if shorts else 0.0
        ssum = sum(vr)

        vrec = (R_none - v_ref) / expl * 100 if expl > 1e-9 else float("nan")
        print(
            f"{name:<24}{expl:>9.4f}{len(shorts):>7}{len(vr):>9}"
            f"{maxd:>9.4f}{med:>9.4f}{ssum:>9.4f}{vrec:>7.1f}",
            flush=True,
        )

    print("\n-- residual-width curves (shortfall V - R_M) --", flush=True)
    for name in OPPONENTS:
        if name not in V:
            continue
        ssum = sum(sh for sh, _c in single[name] if sh > SHORT_TOL)
        print(
            f"\n  {name}   (V-vref={V[name] - v_ref:.4f}, "
            f"sum single value-rel Δ={ssum:.4f})",
            flush=True,
        )
        print(f"    {'k':>3} {'drop_top Δ':>12} {'keep_top Vrec%':>15}", flush=True)
        for k in TOPK:
            csv = ",".join(str(c) for _sh, c in single[name][:k])
            dr = by(name, "drop_top", csv)
            kr = by(name, "keep_top", csv)
            drop_sh = (V[name] - dr["R"]) if dr else float("nan")
            none_r = by(name, "none")
            R_none = none_r["R"] if none_r else float("nan")
            expl = V[name] - v_ref
            keep_rec = (
                (kr["R"] - R_none) / (V[name] - R_none) * 100
                if kr and (V[name] - R_none) > 1e-9
                else float("nan")
            )
            print(f"    {k:>3} {drop_sh:>12.4f} {keep_rec:>15.1f}", flush=True)

    print("\n== verdict ==", flush=True)
    for name in OPPONENTS:
        if name not in V:
            continue
        rec = _W["opp"][name]
        n_total = rec["n_total"]
        n_flow_null = rec["n_flow_null"]

        n_aud_null = sum(1 for sh, _c in single[name] if sh <= SHORT_TOL)
        n_null = n_flow_null + n_aud_null
        frac_null = n_null / n_total * 100 if n_total else float("nan")
        maxd = max((sh for sh, _c in single[name]), default=0.0)
        expl = V[name] - v_ref
        ratio = maxd / expl * 100 if expl > 1e-9 else 0.0
        print(
            f"  {name:<24} payoff-null {frac_null:5.1f}% of {n_total} continuations "
            f"({n_flow_null} by flow + {n_aud_null} audited); "
            f"worst single drop = {ratio:4.1f}% of exploitation",
            flush=True,
        )
    print(
        "  small worst-drop + high payoff-null fraction => weakening buys scope "
        "(under-forcing is cheap).",
        flush=True,
    )


if __name__ == "__main__":
    main()
