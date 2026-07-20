""

import json
import multiprocessing as mp
import os
from pathlib import Path
import statistics
import sys
from typing import Any

from safe_observation.opponents import Opponent, holdem_equilibrium_opponent
from safe_observation.sequence_form import compile as compile_game
from safe_observation.solvers import (
    opponent_reach_weights,
    robust_safe_response_probe,
    safety_verifier,
    solve_blueprint,
)
from scripts.poker.estimate_identifiable_value import _fold_action_indices
from scripts.poker.evaluate_public_routing import _build_gate_suite

GAME = sys.argv[1] if len(sys.argv) > 1 else "holdem_tr_b2"
LEAK = sys.argv[2] if len(sys.argv) > 2 else "revealed_call_strong"
TOPK = int(sys.argv[3]) if len(sys.argv) > 3 else 80
RHO_GRID = [
    float(x)
    for x in (
        sys.argv[4].split(",")
        if len(sys.argv) > 4
        else ["0", "0.05", "0.1", "0.25", "0.5"]
    )
]
NPROC = int(sys.argv[5]) if len(sys.argv) > 5 else 10
EPS = float(sys.argv[6]) if len(sys.argv) > 6 else 0.05
BETA = 1e6
TOL = 1e-9


_W: dict[str, Any] = {}


def _init() -> None:
    sf1 = compile_game(GAME, 1)
    fold_idx = _fold_action_indices(sf1)
    bp = solve_blueprint(GAME, method="lp")
    v_ref = bp.value
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
    _W.update(sf1=sf1, v_ref=v_ref, triv_iv=triv_iv, cont_beh=cont_beh)


def _kappa(task: tuple[str, float]) -> tuple[str, float, float, float]:
    ""
    label, rho = task
    pr = robust_safe_response_probe(
        _W["triv_iv"],
        _W["cont_beh"],
        {label: 1.0},
        v_ref=_W["v_ref"],
        eps_safe=rho,
        beta=BETA,
        rho=0.0,
        game=GAME,
    )
    x = list(pr.realization)
    sv = safety_verifier(x, game=GAME).value
    floor = _W["v_ref"] - rho
    if sv < floor - 1e-6:
        return label, rho, 0.0, sv
    om = opponent_reach_weights(x, game=GAME)
    return label, rho, float(om.get(label, 0.0)), sv


def _quantile(xs: list[float], q: float) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    i = max(0, min(len(s) - 1, int(round(q * (len(s) - 1)))))
    return s[i]


def main() -> None:
    sf1 = compile_game(GAME, 1)
    bp = solve_blueprint(GAME, method="lp")
    v_ref = bp.value
    omega_bp = opponent_reach_weights(bp.realization, game=GAME)
    fold_idx = _fold_action_indices(sf1)
    eqb = holdem_equilibrium_opponent(GAME).behavior
    actions = {i.label: [a for a, _ in i.children] for i in sf1.info_sets}
    suite = _build_gate_suite(GAME, actions, eqb)
    y_eq = list(Opponent(name="eq", behavior=eqb, game=GAME).realization())
    y_star = list(Opponent(name=LEAK, behavior=suite[LEAK], game=GAME).realization())

    def is_river(label: str) -> bool:
        return "/" in (label.split("|", 1)[1] if "|" in label else "")

    carriers: list[tuple[str, float, float]] = []
    for info in sf1.info_sets:
        if not is_river(info.label):
            continue
        fi = fold_idx.get(info.label)
        dev = 0.0
        for a, (_ac, child) in enumerate(info.children):
            if fi is not None and a == fi:
                continue
            dev += abs(y_star[child] - y_eq[child])
        if dev > 1e-6:
            pi_star = y_star[info.parent_seq]
            carriers.append((info.label, dev, pi_star))
    carriers.sort(key=lambda t: t[1], reverse=True)
    targets = carriers[:TOPK]
    target_labels = [t[0] for t in targets]
    dev_by = {t[0]: t[1] for t in targets}
    pi_by = {t[0]: t[2] for t in targets}

    print(
        f"# E-KAPPA capacity frontier  game={GAME}  leak={LEAK}  rho_grid={RHO_GRID}",
        flush=True,
    )
    print(
        f"  v_ref={v_ref:+.5f}  leak-carrying river infosets={len(carriers)}  using top-{len(targets)} by value"
        f"  (eps={EPS}, {NPROC} workers)",
        flush=True,
    )

    tasks = [(lbl, rho) for rho in RHO_GRID for lbl in target_labels]
    results: dict[tuple[str, float], float] = {}
    with mp.Pool(NPROC, initializer=_init) as pool:
        done = 0
        for label, rho, kappa, _sv in pool.imap_unordered(_kappa, tasks, chunksize=4):
            results[(label, rho)] = kappa
            done += 1
            if done % 50 == 0:
                print(f"    ... {done}/{len(tasks)} probes", flush=True)

    passive_kappa = {lbl: float(omega_bp.get(lbl, 0.0)) for lbl in target_labels}

    per_rho: dict[str, Any] = {}
    print(
        f"\n  {'rho':>6}{'med kappa':>11}{'p10 kappa':>11}{'minNZ':>10}{'vw kappa':>10}"
        f"{'med lambda':>12}{'medN_cert':>11}{'frac>1e-3':>11}",
        flush=True,
    )
    for rho in RHO_GRID:
        kappas = [results[(lbl, rho)] for lbl in target_labels]
        lambdas = [results[(lbl, rho)] * pi_by[lbl] for lbl in target_labels]
        nz = [k for k in kappas if k > TOL]
        wsum = sum(dev_by.values())
        vw = (
            sum(results[(lbl, rho)] * dev_by[lbl] for lbl in target_labels) / wsum
            if wsum
            else 0.0
        )
        med_lam = statistics.median(lambdas) if lambdas else 0.0
        med_ncert = (1.0 / (med_lam * EPS * EPS)) if med_lam > TOL else float("inf")
        frac_practical = (
            sum(1 for k in kappas if k > 1e-3) / len(kappas) if kappas else 0.0
        )
        per_rho[f"{rho}"] = {
            "median_kappa": statistics.median(kappas) if kappas else 0.0,
            "p10_kappa": _quantile(kappas, 0.10),
            "min_nonzero_kappa": min(nz) if nz else 0.0,
            "value_weighted_kappa": vw,
            "median_lambda": med_lam,
            "median_N_cert": med_ncert,
            "frac_kappa_gt_1e-3": frac_practical,
            "n_targets": len(kappas),
        }
        ncert_str = (
            f"{med_ncert:>11.0f}" if med_ncert != float("inf") else f"{'inf':>11}"
        )
        print(
            f"  {rho:>6.3f}{statistics.median(kappas):>11.4f}{_quantile(kappas, 0.10):>11.4f}"
            f"{(min(nz) if nz else 0.0):>10.4f}{vw:>10.4f}{med_lam:>12.5f}{ncert_str}"
            f"{frac_practical:>11.2f}",
            flush=True,
        )

    med_passive = (
        statistics.median(list(passive_kappa.values())) if passive_kappa else 0.0
    )
    print(
        f"\n  passive (blueprint) median reach over same targets = {med_passive:.4f}"
        f"  (rho=0 capacity vs this shows the safe-probe headroom)",
        flush=True,
    )

    out = Path(os.environ.get("KSW_OUT", f"results/kappa_rho_sweep_{GAME}.json"))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "game": GAME,
                "leak": LEAK,
                "rho_grid": RHO_GRID,
                "eps": EPS,
                "n_carriers": len(carriers),
                "n_targets": len(targets),
                "passive_median_kappa": med_passive,
                "per_infoset": {
                    lbl: {
                        "dev": dev_by[lbl],
                        "pi_star": pi_by[lbl],
                        "kappa": {f"{rho}": results[(lbl, rho)] for rho in RHO_GRID},
                        "passive_kappa": passive_kappa[lbl],
                    }
                    for lbl in target_labels
                },
                "per_rho": per_rho,
            },
            indent=2,
        )
    )
    print(f"\nwrote {out}", flush=True)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
