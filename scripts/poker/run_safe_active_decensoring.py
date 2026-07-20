""

from concurrent.futures import TimeoutError as FuturesTimeoutError, as_completed
import json
import multiprocessing as mp
import os
from pathlib import Path
import statistics
import sys
from typing import Any

from pebble import ProcessPool

from safe_observation import native
from safe_observation.confidence import (
    OpponentEvidenceStore,
    OpponentEvidenceStore as _Store,
    empirical_bernstein_interval,
    hoeffding_interval,
)
from safe_observation.opponents import Opponent, holdem_equilibrium_opponent
from safe_observation.payoff import build as build_payoff
from safe_observation.sequence_form import compile as compile_game
from safe_observation.solvers import (
    agent_showdown_reach,
    opponent_reach_weights,
    robust_safe_response_linear,
    robust_safe_response_probe,
    robust_safe_response_public,
    safety_constrained_best_response,
    safety_verifier,
    solve_blueprint,
)
from scripts.poker.estimate_identifiable_value import (
    _fold_action_indices,
    _population_public_intervals,
)
from scripts.poker.evaluate_public_routing import _build_gate_suite
from scripts.poker.run_active_decensoring import _empirical_event_constraints
from scripts.poker.run_turn_river_methods import _perturb, _top_rank

GAME = sys.argv[1] if len(sys.argv) > 1 else "holdem_tr_b2"
RHO = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
N_GRID = [
    int(x)
    for x in (
        sys.argv[3].split(",") if len(sys.argv) > 3 else ["1000", "10000", "100000"]
    )
]
SEEDS = (
    [2026 + i for i in range(int(sys.argv[4]))]
    if len(sys.argv) > 4
    else [2026 + i for i in range(10)]
)
NPROC = int(sys.argv[5]) if len(sys.argv) > 5 else 10


CELL_TIMEOUT = (
    float(sys.argv[6])
    if len(sys.argv) > 6
    else float(os.environ.get("SAD_CELL_TIMEOUT", "600"))
)
DELTA = 0.1
METHOD = "empirical_bernstein"
BETA = 1e6
TOL = 1e-6
MODES = ["cpub", "passive", "random", "sad", "oracle_target", "oracle"]
LEAKS = [
    "tr_equilibrium",
    "river_overfold_w80",
    "turn_overfold_w70",
    "revealed_call_strong",
]


ABLATION = os.environ.get("SAD_ABLATION") == "1"
if ABLATION:
    MODES = [
        "cpub",
        "kappa",
        "dpub",
        "dpub_kappa",
        "dpub_lambda",
        "random",
        "oracle_target",
        "oracle",
    ]


if os.environ.get("SAD_CONC") == "1":
    LEAKS = ["tr_equilibrium", "conc_line"]
ACTIVE_PROBE_MODES = (
    ("random", "kappa", "dpub", "dpub_kappa", "dpub_lambda", "oracle_target")
    if ABLATION
    else ("random", "sad", "oracle_target")
)


_W: dict[str, Any] = {}


def _is_river(label: str) -> bool:
    return "/" in (label.split("|", 1)[1] if "|" in label else "")


def _try_solve(fn):
    ""
    try:
        return fn()
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException:
        return None


def _build_probe_behavior(weights: dict[str, float]) -> dict[str, list[float]]:
    ""
    sf0 = _W["sf0"]
    pr = robust_safe_response_probe(
        _W["triv_iv"],
        _W["cont_beh"],
        weights,
        v_ref=_W["v_ref"],
        eps_safe=RHO,
        beta=BETA,
        rho=0.0,
        game=GAME,
    )
    return sf0.behavior_from_realization(list(pr.realization))


def _init(
    probes: dict[str, Any] | None = None, kappa: dict[str, float] | None = None
) -> None:
    sf0 = compile_game(GAME, 0)
    sf1 = compile_game(GAME, 1)
    payoff = build_payoff(GAME)
    bp = solve_blueprint(GAME, method="lp")
    v_ref = bp.value
    omega_bp = opponent_reach_weights(bp.realization, game=GAME)
    fold_idx = _fold_action_indices(sf1)
    groups = OpponentEvidenceStore.for_game(GAME).public_groups()
    info_by = {i.label: i for i in sf1.info_sets}
    eqb = holdem_equilibrium_opponent(GAME).behavior
    y_eq = list(Opponent(name="eq", behavior=eqb, game=GAME).realization())
    bp_behavior = sf0.behavior_from_realization(bp.realization)
    triv_iv = {i.label: [(0.0, 1.0)] * len(i.children) for i in sf1.info_sets}
    cont_beh = {}
    for info in sf1.info_sets:
        n = len(info.children)
        fi = fold_idx.get(info.label)
        if fi is None:
            cont_beh[info.label] = [1.0 / n] * n
        else:
            nf = [k for k in range(n) if k != fi]
            cont_beh[info.label] = [
                (1.0 / len(nf)) if k in nf else 0.0 for k in range(n)
            ]
    actions = {i.label: [a for a, _ in i.children] for i in sf1.info_sets}
    suite = _build_gate_suite(GAME, actions, eqb)

    if os.environ.get("SAD_CONC") == "1":
        cheap = {"cc/7sc", "cc/Ahc"}
        deep = {"cpac/7sc", "cpac/Ahc", "cppac/7sc", "cpp", "cpa"}
        conc = _perturb(
            eqb,
            actions,
            lambda hole, hist, acts: (
                hist in cheap and _top_rank(hole) >= 11 and "c" in acts
            ),
            "c",
            0.6,
        )
        conc = _perturb(
            conc,
            actions,
            lambda hole, hist, acts: (
                hist in deep and _top_rank(hole) >= 11 and "f" in acts
            ),
            "f",
            0.7,
        )
        suite["conc_line"] = conc

    pi_hat = {
        info.label: y_eq[info.parent_seq]
        for info in sf1.info_sets
        if _is_river(info.label)
    }
    _W.update(
        sf0=sf0,
        sf1=sf1,
        payoff=payoff,
        v_ref=v_ref,
        floor=v_ref - RHO,
        omega_bp=omega_bp,
        fold_idx=fold_idx,
        groups=groups,
        info_by=info_by,
        y_eq=y_eq,
        bp_behavior=bp_behavior,
        triv_iv=triv_iv,
        cont_beh=cont_beh,
        suite=suite,
        probes=probes or {},
        kappa=kappa or {},
        pi_hat=pi_hat,
    )


def _public_anomaly_weights(y_ref: list[float]) -> dict[str, float]:
    ""
    sf1, omega_bp, y_eq = _W["sf1"], _W["omega_bp"], _W["y_eq"]
    by_hist: dict[str, list[Any]] = {}
    for info in sf1.info_sets:
        h = info.label.split("|", 1)[1] if "|" in info.label else ""
        by_hist.setdefault(h, []).append(info)

    anomaly: dict[str, float] = {}
    for h, infos in by_hist.items():
        nA = len(infos[0].children)
        num_ref = [0.0] * nA
        num_eq = [0.0] * nA
        den_ref = den_eq = 0.0
        for info in infos:
            w = omega_bp.get(info.label, 0.0)
            if w <= 0.0:
                continue
            den_ref += w * y_ref[info.parent_seq]
            den_eq += w * y_eq[info.parent_seq]
            for a, (_ac, child) in enumerate(info.children):
                num_ref[a] += w * y_ref[child]
                num_eq[a] += w * y_eq[child]
        if den_ref <= 1e-12 or den_eq <= 1e-12:
            continue
        tv = 0.5 * sum(
            abs(num_ref[a] / den_ref - num_eq[a] / den_eq) for a in range(nA)
        )
        if tv > 1e-6:
            anomaly[h] = tv

    weights: dict[str, float] = {}
    for info in sf1.info_sets:
        h = info.label.split("|", 1)[1] if "|" in info.label else ""
        if "/" not in h:
            continue
        score = sum(tv for hk, tv in anomaly.items() if h.startswith(hk))
        if score > 1e-6:
            weights[info.label] = score
    return weights


def _leak_weights(y_star: list[float]) -> dict[str, float]:
    ""
    sf1, fold_idx, y_eq = _W["sf1"], _W["fold_idx"], _W["y_eq"]
    w: dict[str, float] = {}
    for info in sf1.info_sets:
        if not _is_river(info.label):
            continue
        fi = fold_idx.get(info.label)
        dev = sum(
            abs(y_star[child] - y_eq[child])
            for a, (_ac, child) in enumerate(info.children)
            if not (fi is not None and a == fi)
        )
        if dev > 1e-6:
            w[info.label] = dev
    return w


def _kappa_weights() -> dict[str, float]:
    ""
    return {i: k for i, k in _W["kappa"].items() if k > 1e-9}


def _dpub_kappa_weights(y_ref: list[float]) -> dict[str, float]:
    ""
    d = _public_anomaly_weights(y_ref)
    k = _W["kappa"]
    return {i: d[i] * k.get(i, 0.0) for i in d if k.get(i, 0.0) > 1e-9}


def _dpub_lambda_weights(y_ref: list[float]) -> dict[str, float]:
    ""
    d = _public_anomaly_weights(y_ref)
    k = _W["kappa"]
    pi = _W["pi_hat"]
    return {
        i: d[i] * k.get(i, 0.0) * pi.get(i, 0.0)
        for i in d
        if k.get(i, 0.0) > 1e-9 and pi.get(i, 0.0) > 1e-12
    }


def _kappa_task(label: str) -> tuple[str, float]:
    ""
    pr = _try_solve(
        lambda: robust_safe_response_probe(
            _W["triv_iv"],
            _W["cont_beh"],
            {label: 1.0},
            v_ref=_W["v_ref"],
            eps_safe=RHO,
            beta=BETA,
            rho=0.0,
            game=GAME,
        )
    )
    if pr is None:
        return label, 0.0
    x = list(pr.realization)
    return label, float(opponent_reach_weights(x, game=GAME).get(label, 0.0))


def _probe_task(task: tuple[str, str]) -> tuple[str, Any, Any]:
    ""
    import random as _random

    mode, name = task
    behavior = _W["suite"][name]
    y_star = list(Opponent(name=name, behavior=behavior, game=GAME).realization())
    if mode == "random":
        rng = _random.Random(hash(name) & 0xFFFF)
        weights = {
            i.label: rng.random() for i in _W["sf1"].info_sets if _is_river(i.label)
        }
    elif mode in ("sad", "dpub"):
        weights = _public_anomaly_weights(y_star)
    elif mode == "kappa":
        weights = _kappa_weights()
    elif mode == "dpub_kappa":
        weights = _dpub_kappa_weights(y_star)
    elif mode == "dpub_lambda":
        weights = _dpub_lambda_weights(y_star)
    elif mode == "oracle_target":
        weights = _leak_weights(y_star)
    else:
        raise ValueError(mode)
    if not weights:
        return f"{mode}|{name}", None, None
    beh = _build_probe_behavior(weights)
    sv = safety_verifier(_W["sf0"].realization_from_behavior(beh), game=GAME).value
    return f"{mode}|{name}", beh, sv


def _empirical_mass_constraints(sf1, show, sd_reach, fold_idx, episodes, delta, method):
    ""

    pins: list[tuple[Any, int, int, float]] = []
    for info in sf1.info_sets:
        row = sd_reach.get(info.label)
        if row is None:
            continue
        fi = fold_idx.get(info.label)
        for a, (_act, child) in enumerate(info.children):
            if fi is not None and a == fi:
                continue
            if a >= len(row):
                continue
            w_sd, committing = row[a]
            if committing and w_sd > 1e-12:
                pins.append((info, a, child, w_sd))
    n_pairs = len(pins)
    per = delta / max(1, n_pairs)

    entries: list[tuple[int, int, float]] = []
    h: list[float] = []
    meta: list[tuple[str, int]] = []
    row_i = 0
    for info, a, child, w_sd in pins:
        cnt = show.get(info.label, [0] * len(info.children))
        mhat = cnt[a] / episodes
        if method == "empirical_bernstein":
            lo, hi = empirical_bernstein_interval(mhat, episodes, per)
        else:
            lo, hi = hoeffding_interval(mhat, episodes, per)
        if hi < w_sd:
            entries.append((row_i, child, w_sd))
            h.append(hi)
            meta.append((info.label, a))
            row_i += 1
        if lo > 0.0:
            entries.append((row_i, child, -w_sd))
            h.append(-lo)
            meta.append((info.label, a))
            row_i += 1
    return entries, h, meta, n_pairs


def _deploy(mode: str, name: str, episodes: int, seed: int) -> dict[str, Any]:
    sf1, payoff, v_ref = _W["sf1"], _W["payoff"], _W["v_ref"]
    groups, omega_bp, fold_idx = _W["groups"], _W["omega_bp"], _W["fold_idx"]
    info_by = _W["info_by"]
    behavior = _W["suite"][name]
    y_star = list(Opponent(name=name, behavior=behavior, game=GAME).realization())

    if mode == "oracle":
        orc = safety_constrained_best_response(
            behavior, v_ref=v_ref, eps_safe=RHO, game=GAME
        )
        x = list(orc.realization)
        realized = payoff.bilinear(x, y_star) - v_ref
        sv = safety_verifier(x, game=GAME).value
        return {
            "certified": realized,
            "realized": realized,
            "safety": sv,
            "reveal_budget": 0,
        }

    if mode == "cpub":
        pub = _population_public_intervals(groups, info_by, y_star, omega_bp)
        core = robust_safe_response_public(
            groups, pub, v_ref=v_ref, eps_safe=RHO, game=GAME, weights=omega_bp
        )
        x = list(core.realization)
        return {
            "certified": core.robust_value - v_ref,
            "realized": payoff.bilinear(x, y_star) - v_ref,
            "safety": safety_verifier(x, game=GAME).value,
            "reveal_budget": 0,
        }

    if mode == "passive":
        agent_behavior = _W["bp_behavior"]
    else:
        probe = _W["probes"].get(f"{mode}|{name}")
        agent_behavior = probe if probe is not None else _W["bp_behavior"]
    _pay, show, fold = native.simulate_showdown(
        GAME, agent_behavior, behavior, episodes, seed
    )
    ev_public = _Store.for_game(GAME)
    for label, c in show.items():
        ev_public.record(label, c)
    for label, c in fold.items():
        ev_public.record(label, c)
    pub_intervals = ev_public.public_intervals(DELTA, method=METHOD)

    agent_real = _W["sf0"].realization_from_behavior(agent_behavior)
    omega = opponent_reach_weights(agent_real, game=GAME)
    if mode == "passive":
        ev_point = _Store.for_game(GAME)
        for label, c in show.items():
            ev_point.record(label, c)
        ev_entries, ev_h, ev_meta, n_pairs = _empirical_event_constraints(
            sf1, ev_point, omega, fold_idx, DELTA, METHOD
        )
    else:
        sd_reach = agent_showdown_reach(agent_real, game=GAME)
        ev_entries, ev_h, ev_meta, n_pairs = _empirical_mass_constraints(
            sf1, show, sd_reach, fold_idx, episodes, DELTA, METHOD
        )

    box_infeasible = False
    pub_infeasible = False
    resp = _try_solve(
        lambda: robust_safe_response_linear(
            groups,
            pub_intervals,
            ev_entries,
            ev_h,
            v_ref=v_ref,
            eps_safe=RHO,
            game=GAME,
            weights=omega,
            row_meta=ev_meta,
        )
    )
    if resp is None:
        box_infeasible = True
        resp = _try_solve(
            lambda: robust_safe_response_public(
                groups,
                pub_intervals,
                v_ref=v_ref,
                eps_safe=RHO,
                game=GAME,
                weights=omega,
            )
        )
    if resp is None:
        pub_infeasible = True
        pop = _population_public_intervals(groups, info_by, y_star, omega_bp)
        resp = robust_safe_response_public(
            groups, pop, v_ref=v_ref, eps_safe=RHO, game=GAME, weights=omega_bp
        )
    x = list(resp.realization)
    return {
        "certified": resp.robust_value - v_ref,
        "realized": payoff.bilinear(x, y_star) - v_ref,
        "safety": safety_verifier(x, game=GAME).value,
        "reveal_budget": episodes,
        "n_event_pairs": n_pairs,
        "box_infeasible": box_infeasible,
        "pub_infeasible": pub_infeasible,
    }


def _cell(task: tuple[str, str, int, int]) -> dict[str, Any]:
    mode, name, episodes, seed = task
    r = _deploy(mode, name, episodes, seed)
    r.update(
        mode=mode,
        opponent=name,
        episodes=episodes,
        seed=seed,
        violation=r["safety"] < _W["floor"] - TOL,
    )
    return r


def _timeout_row(task: tuple[str, str, int, int], reason: str) -> dict[str, Any]:
    ""
    mode, name, episodes, seed = task
    return {
        "mode": mode,
        "opponent": name,
        "episodes": episodes,
        "seed": seed,
        "certified": float("nan"),
        "realized": float("nan"),
        "safety": float("nan"),
        "reveal_budget": episodes,
        "box_infeasible": False,
        "lp_timeout": True,
        "reason": reason,
        "violation": False,
    }


def _load_or_compute_kappa(river_labels: list[str]) -> dict[str, float]:
    ""
    path = Path(f"results/kappa_cache_{GAME}_rho{RHO}.json")
    cached: dict[str, float] = {}
    if path.exists():
        cached = {k: float(v) for k, v in json.loads(path.read_text()).items()}
    todo = [lbl for lbl in river_labels if lbl not in cached]
    if not todo:
        return cached
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(path) + ".tmp")

    def _flush() -> None:
        tmp.write_text(json.dumps(cached))
        tmp.replace(path)

    total = len(river_labels)
    done = len(cached)
    n_timeout = 0
    since_flush = 0
    with ProcessPool(
        max_workers=NPROC,
        initializer=_init,
        context=mp.get_context("spawn"),
    ) as pool:
        futures = {
            pool.schedule(_kappa_task, args=(lbl,), timeout=CELL_TIMEOUT): lbl
            for lbl in todo
        }
        for fut in as_completed(futures):
            lbl = futures[fut]
            try:
                _lbl, k = fut.result()
            except FuturesTimeoutError:
                k, n_timeout = 0.0, n_timeout + 1
            except Exception:
                k = 0.0
            cached[lbl] = k
            done += 1
            since_flush += 1
            if since_flush >= 25:
                _flush()
                since_flush = 0
                print(f"    capacity setup: {done}/{total} LPs", flush=True)
    _flush()
    if n_timeout:
        print(
            f"    capacity setup: {n_timeout} LP(s) hit the per-LP budget "
            f"(recorded kappa=0)",
            flush=True,
        )
    return cached


def main() -> None:
    print(
        f"Safe Active De-censoring  game={GAME}  rho={RHO}  N={N_GRID}  "
        f"seeds={len(SEEDS)}  modes={MODES}  ({NPROC} workers)",
        flush=True,
    )

    kappa: dict[str, float] | None = None
    if ABLATION:
        sf1_p = compile_game(GAME, 1)
        river_labels = [i.label for i in sf1_p.info_sets if _is_river(i.label)]
        kappa = _load_or_compute_kappa(river_labels)
        n_pos = sum(1 for v in kappa.values() if v > 1e-9)
        print(
            f"  capacity setup: cached for {len(kappa)} targets ({n_pos} with kappa>0)",
            flush=True,
        )

    probe_tasks = [(m, n) for m in ACTIVE_PROBE_MODES for n in LEAKS]
    probes: dict[str, Any] = {}
    probe_safety: dict[str, Any] = {}
    with mp.Pool(NPROC, initializer=_init, initargs=(None, kappa)) as pool:
        for key, beh, sv in pool.imap_unordered(_probe_task, probe_tasks, chunksize=1):
            probes[key] = beh
            probe_safety[key] = sv
    n_real = sum(1 for b in probes.values() if b is not None)
    print(
        f"  probe construction: {n_real}/{len(probe_tasks)} policies computed"
        f"  (rest fall back to blueprint: no signal)",
        flush=True,
    )
    min_probe_safe = min(
        (s for s in probe_safety.values() if s is not None), default=float("nan")
    )
    print(
        f"  min probe safety value = {min_probe_safe:+.4f}  (floor = {(-RHO):+.4f} rel v_ref)",
        flush=True,
    )

    tasks: list[tuple[str, str, int, int]] = []
    for name in LEAKS:
        for mode in MODES:
            if mode in ("cpub", "oracle"):
                tasks.append((mode, name, 0, 0))
            else:
                for n in N_GRID:
                    for s in SEEDS:
                        tasks.append((mode, name, n, s))

    out_path = Path(os.environ.get("SAD_OUT", f"results/sad_deploy_{GAME}.json"))
    ckpt = out_path.with_suffix(".cells.jsonl")
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    done_keys: set[tuple[str, str, int, int]] = set()
    if ckpt.exists():
        for line in ckpt.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            rows.append(r)
            done_keys.add(
                (r["mode"], r["opponent"], int(r["episodes"]), int(r["seed"]))
            )
        print(f"  resuming: {len(done_keys)} cells already checkpointed", flush=True)
    pending = [t for t in tasks if t not in done_keys]

    cell_timeout = CELL_TIMEOUT
    total = len(tasks)
    done = len(rows)

    ckpt_fh = open(ckpt, "a")
    with ProcessPool(
        max_workers=NPROC,
        initializer=_init,
        initargs=(probes, kappa),
        context=mp.get_context("spawn"),
    ) as pool:
        futures = {
            pool.schedule(_cell, args=(t,), timeout=cell_timeout): t for t in pending
        }
        for fut in as_completed(futures):
            task = futures[fut]
            try:
                r = fut.result()
            except FuturesTimeoutError:
                r = _timeout_row(task, "lp_timeout")
            except Exception as exc:
                r = _timeout_row(task, f"error:{type(exc).__name__}")
            rows.append(r)
            ckpt_fh.write(json.dumps(r) + "\n")
            ckpt_fh.flush()
            done += 1
            if done % 25 == 0:
                print(f"    ... {done}/{total} cells", flush=True)
    ckpt_fh.close()

    cpub = {r["opponent"]: r for r in rows if r["mode"] == "cpub"}
    oracle = {r["opponent"]: r for r in rows if r["mode"] == "oracle"}

    def cap(realized: float, name: str) -> float:
        lo = cpub[name]["realized"]
        hi = oracle[name]["realized"]
        return (realized - lo) / (hi - lo) if hi - lo > 1e-9 else 0.0

    summary: dict[str, Any] = {}
    print(
        f"\n  {'opponent':<20}{'mode':<14}{'N':>7}{'cert':>8}{'real':>8}{'capture':>8}{'viol':>6}{'infeas':>6}",
        flush=True,
    )
    for name in LEAKS:
        for mode in MODES:
            sub = [r for r in rows if r["opponent"] == name and r["mode"] == mode]
            if not sub:
                continue
            by_n: dict[int, list[dict[str, Any]]] = {}
            for r in sub:
                by_n.setdefault(r["episodes"], []).append(r)
            for n, group in sorted(by_n.items()):
                ok = [r for r in group if not r.get("lp_timeout")]
                n_to = sum(bool(r.get("lp_timeout")) for r in group)
                cert = (
                    statistics.mean(r["certified"] for r in ok) if ok else float("nan")
                )
                real = (
                    statistics.mean(r["realized"] for r in ok) if ok else float("nan")
                )
                viol = sum(bool(r.get("violation")) for r in ok)
                infeas = sum(r.get("box_infeasible", False) for r in ok)
                capture = cap(real, name) if ok else float("nan")
                summary[f"{name}|{mode}|{n}"] = {
                    "certified": cert,
                    "realized": real,
                    "capture": capture,
                    "violations": viol,
                    "box_infeasible": infeas,
                    "lp_timeout": n_to,
                    "n_cells": len(group),
                }
                inf_str = (
                    f"{infeas:>4}/{n_to:<3}"
                    if mode not in ("cpub", "oracle")
                    else "   .   "
                )
                print(
                    f"  {name:<20}{mode:<14}{n:>7}{cert:>+8.3f}{real:>+8.3f}{capture:>8.2f}{viol:>6} {inf_str}",
                    flush=True,
                )

    total_viol = sum(bool(r.get("violation")) for r in rows)
    total_infeas = sum(r.get("box_infeasible", False) for r in rows)
    total_timeout = sum(bool(r.get("lp_timeout")) for r in rows)
    print(
        f"\n  TOTAL floor violations across all {len(rows)} cells = {total_viol}"
        f"  (must be 0 -- structural by Thm 4)",
        flush=True,
    )
    print(
        f"  TOTAL naive-box infeasibilities (C_pub ∩ C_event empty) = {total_infeas}"
        f"  (MNAR pathology, Thm 1; deployer falls back to C_pub)",
        flush=True,
    )
    print(
        f"  TOTAL lp_timeout cells (degenerate robust LP abandoned) = {total_timeout}"
        f"  (excluded from value means; raise SAD_CELL_TIMEOUT to reduce)",
        flush=True,
    )

    out = out_path
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "game": GAME,
                "rho": RHO,
                "n_grid": N_GRID,
                "seeds": SEEDS,
                "delta": DELTA,
                "method": METHOD,
                "modes": MODES,
                "leaks": LEAKS,
                "total_violations": total_viol,
                "summary": summary,
                "rows": rows,
            },
            indent=2,
        )
    )
    print(f"\nwrote {out}", flush=True)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
