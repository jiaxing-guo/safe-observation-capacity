"""Run the active de-censoring experiment. See Safe Active De-censoring and supplementary Algorithms."""

import json
import os
from pathlib import Path
import statistics
from typing import Any

from safe_observation import native
from safe_observation.confidence import OpponentEvidenceStore
from safe_observation.opponents import Opponent, holdem_equilibrium_opponent
from safe_observation.payoff import build as build_payoff
from safe_observation.sequence_form import compile as compile_game
from safe_observation.solvers import (
    opponent_reach_weights,
    robust_safe_response_linear,
    robust_safe_response_public,
    safety_constrained_best_response,
    safety_verifier,
    solve_blueprint,
)
from scripts.poker.estimate_identifiable_value import _fold_action_indices
from scripts.poker.evaluate_finite_sample_routing import _empirical_signal, _pf_eq_turn
from scripts.poker.evaluate_public_routing import _build_gate_suite

GAME = os.environ.get("ONL_GAME", "holdem_tr_b2")
RHO = float(os.environ.get("ONL_RHO", "0.5"))
DELTA = float(os.environ.get("ONL_DELTA", "0.1"))
METHOD = os.environ.get("ONL_METHOD", "empirical_bernstein")
GRID = [int(x) for x in os.environ.get("ONL_GRID", "1000,10000,100000").split(",")]
SEEDS = [
    int(x) for x in os.environ.get("ONL_SEEDS", "2026,2027,2028,2029,2030").split(",")
]
SIGMA_K = float(os.environ.get("ONL_SIGMA_K", "4.0"))
N_SHARDS = int(os.environ.get("ONL_N_SHARDS", "1"))
SHARD_IDX = int(os.environ.get("ONL_SHARD_IDX", "0"))
MERGE = os.environ.get("ONL_MERGE", "0") == "1"
TOL = 1e-6


def _out_path(shard: int | None = None) -> Path:
    """Compute out path for the run active de-censoring workflow."""
    base = os.environ.get("ONL_OUT")
    if base:
        return Path(base)
    suffix = "" if shard is None else f"_shard{shard:03d}"
    return Path(f"results/active_decensoring_{GAME}{suffix}.json")


def _empirical_event_constraints(sf1, ev_point, omega, fold_idx, delta, method):
    """Compute empirical event constraints for the run active de-censoring workflow."""
    reached = [
        info
        for info in sf1.info_sets
        if omega.get(info.label, 0.0) > 0.0 and ev_point.visits(info.label) > 0
    ]
    n_pairs = 0
    for info in reached:
        fi = fold_idx.get(info.label)
        n_pairs += sum(1 for a in range(len(info.children)) if a != fi)
    per = delta / max(1, n_pairs)

    entries: list[tuple[int, int, float]] = []
    h: list[float] = []
    meta: list[tuple[str, int]] = []
    row = 0
    for info in reached:
        box = ev_point.interval(info.label, per, method)
        fi = fold_idx.get(info.label)
        for a, (_act, child) in enumerate(info.children):
            if fi is not None and a == fi:
                continue
            lo, hi = box[a]
            if hi < 1.0:
                entries.append((row, child, 1.0))
                entries.append((row, info.parent_seq, -hi))
                h.append(0.0)
                meta.append((info.label, a))
                row += 1
            if lo > 0.0:
                entries.append((row, child, -1.0))
                entries.append((row, info.parent_seq, lo))
                h.append(0.0)
                meta.append((info.label, a))
                row += 1
    return entries, h, meta, n_pairs


def _observation_covers(sf1, ev_point, omega, fold_idx, delta, method, y_star) -> bool:
    """Compute observation covers for the run active de-censoring workflow."""
    reached = [
        info
        for info in sf1.info_sets
        if omega.get(info.label, 0.0) > 0.0 and ev_point.visits(info.label) > 0
    ]
    n_pairs = 0
    for info in reached:
        fi = fold_idx.get(info.label)
        n_pairs += sum(1 for a in range(len(info.children)) if a != fi)
    per = delta / max(1, n_pairs)
    for info in reached:
        box = ev_point.interval(info.label, per, method)
        parent = y_star[info.parent_seq]
        fi = fold_idx.get(info.label)
        for a, (_act, child) in enumerate(info.children):
            if fi is not None and a == fi:
                continue
            lo, hi = box[a]
            v = y_star[child]
            if v < lo * parent - 1e-9 or v > hi * parent + 1e-9:
                return False
    return True


def _cell_keys() -> list[tuple[str, int]]:
    """Compute cell keys for the run active de-censoring workflow."""
    suite_names = list(_suite_cache().keys())
    cells = [(name, n) for name in suite_names for n in GRID]
    return [c for k, c in enumerate(cells) if k % N_SHARDS == SHARD_IDX]


_SUITE: dict[str, Any] | None = None


def _suite_cache() -> dict[str, Any]:
    """Compute suite cache for the run active de-censoring workflow."""
    global _SUITE
    if _SUITE is None:
        sf1 = compile_game(GAME, 1)
        actions = {info.label: [a for a, _ in info.children] for info in sf1.info_sets}
        eq = holdem_equilibrium_opponent(GAME).behavior
        _SUITE = _build_gate_suite(GAME, actions, eq)
    return _SUITE


def main() -> None:
    """Run the command-line entry point."""
    if MERGE:
        _merge()
        return

    game = GAME
    print(
        f"# C_obs ONLINE finite-sample loop  game={game}  rho={RHO}  delta={DELTA}  "
        f"method={METHOD}  grid={GRID}  seeds={len(SEEDS)}  shard={SHARD_IDX}/{N_SHARDS}",
        flush=True,
    )
    payoff = build_payoff(game)
    sf0 = compile_game(game, 0)
    sf1 = compile_game(game, 1)
    info_by_label = {info.label: info for info in sf1.info_sets}
    bp = solve_blueprint(game, method="lp")
    assert bp.realization is not None
    v_ref = bp.value
    agent_behavior = sf0.behavior_from_realization(bp.realization)
    omega = opponent_reach_weights(bp.realization, game=game)
    fold_idx = _fold_action_indices(sf1)
    groups = OpponentEvidenceStore.for_game(game).public_groups()
    eq = holdem_equilibrium_opponent(game).behavior
    y_eq = list(Opponent(name="eq", behavior=eq, game=game).realization())
    pf_eq_turn = _pf_eq_turn(groups, info_by_label, y_eq, omega, fold_idx)
    floor = v_ref - RHO
    suite = _suite_cache()

    cells = _cell_keys()
    print(
        f"  v_ref={v_ref:+.5f}  floor={floor:+.4f}  cells_this_shard={len(cells)}",
        flush=True,
    )

    rows: list[dict[str, Any]] = []
    for name, episodes in cells:
        behavior = suite[name]
        y_star = list(Opponent(name=name, behavior=behavior, game=game).realization())
        oracle = safety_constrained_best_response(
            behavior, v_ref=v_ref, eps_safe=RHO, game=game
        )
        oracle_gain = payoff.bilinear(list(oracle.realization), y_star) - v_ref

        for seed in SEEDS:
            ev_point = OpponentEvidenceStore.for_game(game)
            ev_public = OpponentEvidenceStore.for_game(game)
            _pay, show, fold = native.simulate_showdown(
                game, agent_behavior, behavior, episodes, seed
            )
            for label, c in show.items():
                ev_point.record(label, c)
                ev_public.record(label, c)
            for label, c in fold.items():
                ev_public.record(label, c)

            d_hat = _empirical_signal(ev_public, pf_eq_turn, episodes)
            pub_intervals = ev_public.public_intervals(DELTA, method=METHOD)
            ev_entries, ev_h, ev_meta, n_pairs = _empirical_event_constraints(
                sf1, ev_point, omega, fold_idx, DELTA, METHOD
            )

            core = robust_safe_response_public(
                groups,
                pub_intervals,
                v_ref=v_ref,
                eps_safe=RHO,
                game=game,
                weights=omega,
            )
            obs = robust_safe_response_linear(
                groups,
                pub_intervals,
                ev_entries,
                ev_h,
                v_ref=v_ref,
                eps_safe=RHO,
                game=game,
                weights=omega,
                row_meta=ev_meta,
            )
            core_gain = payoff.bilinear(list(core.realization), y_star) - v_ref
            obs_gain = payoff.bilinear(list(obs.realization), y_star) - v_ref
            core_safe = safety_verifier(list(core.realization), game=game).value
            obs_safe = safety_verifier(list(obs.realization), game=game).value
            covered = _observation_covers(
                sf1, ev_point, omega, fold_idx, DELTA, METHOD, y_star
            )
            rows.append(
                {
                    "opponent": name,
                    "episodes": episodes,
                    "seed": seed,
                    "d_hat": d_hat,
                    "core_gain": core_gain,
                    "obs_gain": obs_gain,
                    "oracle_gain": oracle_gain,
                    "obs_minus_core": obs_gain - core_gain,
                    "core_safety": core_safe,
                    "obs_safety": obs_safe,
                    "floor": floor,
                    "core_violation": core_safe < floor - TOL,
                    "obs_violation": obs_safe < floor - TOL,
                    "observation_covered": covered,
                    "n_event_pairs": n_pairs,
                }
            )
        print(
            f"  done {name:<22} N-cells; last Delta_hat avg="
            f"{statistics.mean(r['obs_minus_core'] for r in rows[-len(SEEDS) :]):+.3f}",
            flush=True,
        )

    out = _out_path(SHARD_IDX if N_SHARDS > 1 else None)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "game": game,
        "rho": RHO,
        "delta": DELTA,
        "method": METHOD,
        "grid": GRID,
        "seeds": SEEDS,
        "sigma_k": SIGMA_K,
        "shard_idx": SHARD_IDX,
        "n_shards": N_SHARDS,
        "rows": rows,
    }
    if N_SHARDS == 1:
        payload["summary"] = _summarize(rows)
        _print_summary(payload["summary"])
    out.write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {out}", flush=True)


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize the supplied experiment records."""
    by_n: dict[int, list[dict[str, Any]]] = {}
    for r in rows:
        by_n.setdefault(r["episodes"], []).append(r)

    summary: dict[str, Any] = {"per_n": {}}
    for n, group in sorted(by_n.items()):
        eq_dhat = [r["d_hat"] for r in group if r["opponent"] == "tr_equilibrium"]
        floor_mean = statistics.mean(eq_dhat) if eq_dhat else 0.0
        floor_std = statistics.stdev(eq_dhat) if len(eq_dhat) > 1 else 0.0
        tau = floor_mean + SIGMA_K * floor_std

        opp_names = sorted({r["opponent"] for r in group})
        always_core = always_obs = gate_val = oracle_tot = 0.0
        engaged = []
        for name in opp_names:
            cells = [r for r in group if r["opponent"] == name]
            mc = statistics.mean(r["core_gain"] for r in cells)
            mo = statistics.mean(r["obs_gain"] for r in cells)
            mor = statistics.mean(r["oracle_gain"] for r in cells)
            md = statistics.mean(r["d_hat"] for r in cells)
            always_core += mc
            always_obs += mo
            oracle_tot += mor
            if md > tau:
                engaged.append(name)
                gate_val += mo
            else:
                gate_val += mc
        violations = sum(r["core_violation"] or r["obs_violation"] for r in group)
        covered = sum(1 for r in group if r["observation_covered"])
        denom = always_obs - always_core
        summary["per_n"][str(n)] = {
            "tau": tau,
            "floor_mean": floor_mean,
            "floor_std": floor_std,
            "solve_rate": len(engaged) / len(opp_names) if opp_names else 0.0,
            "engaged": engaged,
            "always_core": always_core,
            "always_obs": always_obs,
            "gate_value": gate_val,
            "oracle": oracle_tot,
            "gate_capture_of_max": (gate_val - always_core) / denom
            if abs(denom) > 1e-9
            else 0.0,
            "safety_violation_rate": violations / len(group) if group else 0.0,
            "observation_coverage": covered / len(group) if group else 1.0,
            "target_coverage": 1.0 - DELTA,
        }
    return summary


def _print_summary(summary: dict[str, Any]) -> None:
    """Compute print summary for the run active de-censoring workflow."""
    print("\n  per-N deployment summary:", flush=True)
    for n, s in summary["per_n"].items():
        print(
            f"    N={int(n):>7}  solve_rate={s['solve_rate']:.2f}  "
            f"gate_capture_of_max={s['gate_capture_of_max']:+.3f}  "
            f"viol_rate={s['safety_violation_rate']:.3f}  "
            f"observation_cov={s['observation_coverage']:.3f} (target {s['target_coverage']:.2f})",
            flush=True,
        )


def _merge() -> None:
    """Merge the supplied records into this aggregate."""
    shards = sorted(Path("results").glob(f"active_decensoring_{GAME}_shard*.json"))
    if not shards:
        raise SystemExit(f"no shard files for game={GAME} to merge")
    rows: list[dict[str, Any]] = []
    meta: dict[str, Any] = {}
    for path in shards:
        blob = json.loads(path.read_text())
        rows.extend(blob["rows"])
        meta = {
            k: blob[k]
            for k in ("game", "rho", "delta", "method", "grid", "seeds", "sigma_k")
        }
    summary = _summarize(rows)
    out = _out_path(None)
    out.write_text(
        json.dumps(
            {**meta, "n_shards_merged": len(shards), "summary": summary, "rows": rows},
            indent=2,
        )
    )
    print(f"merged {len(shards)} shards ({len(rows)} cells) -> {out}", flush=True)
    _print_summary(summary)


if __name__ == "__main__":
    main()
