""

import json
import os
from pathlib import Path
import random
import signal
from typing import Any

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
from scripts.poker.estimate_identifiable_value import (
    _fold_action_indices,
    _population_event_constraints,
    _population_public_intervals,
)
from scripts.poker.evaluate_public_routing import _pearson, _spearman
from scripts.poker.evaluate_reveal_routing import _fold_pub_signal, _gate_curve, _knee
from scripts.poker.run_turn_river_methods import _perturb, _top_rank

GAME = os.environ.get("RPOP_GAME", "holdem_tr_b2")
RHO = float(os.environ.get("RPOP_RHO", "0.5"))
N_POP = int(os.environ.get("RPOP_N", "200"))
BASE_SEED = int(os.environ.get("RPOP_SEED", "2026"))
MAX_MAG = float(os.environ.get("RPOP_MAXMAG", "0.8"))
WITH_ORACLE = os.environ.get("RPOP_WITH_ORACLE", "0") == "1"
N_SHARDS = int(os.environ.get("RPOP_N_SHARDS", "1"))
SHARD_IDX = int(os.environ.get("RPOP_SHARD_IDX", "0"))
MERGE = os.environ.get("RPOP_MERGE", "0") == "1"


OPP_TIMEOUT = int(os.environ.get("RPOP_OPP_TIMEOUT", "0"))


class _OpponentTimeout(Exception):
    ""


def _alarm(seconds: int):
    ""

    class _Guard:
        def __enter__(self):
            if seconds > 0 and hasattr(signal, "SIGALRM"):

                def _handler(signum, frame):
                    raise _OpponentTimeout

                self._prev = signal.signal(signal.SIGALRM, _handler)
                signal.alarm(seconds)
            return self

        def __exit__(self, *exc):
            if seconds > 0 and hasattr(signal, "SIGALRM"):
                signal.alarm(0)
                signal.signal(signal.SIGALRM, self._prev)
            return False

    return _Guard()


def _out_path(shard: int | None = None) -> Path:
    base = os.environ.get("RPOP_OUT")
    if base:
        return Path(base)
    suffix = "" if shard is None else f"_shard{shard:03d}"
    return Path(f"results/opponent_population_{GAME}{suffix}.json")


def _random_opponent(
    rng: random.Random, eq: dict[str, list[float]], actions: dict[str, list[str]]
) -> tuple[dict[str, list[float]], dict[str, Any]]:
    ""
    line = rng.choice(["turn", "river", "any"])
    target = rng.choice(["f", "f", "c"])
    strength = rng.choice(["all", "strong", "weak"])
    mag = rng.uniform(0.05, MAX_MAG)

    def pick(hole: str, hist: str, acts: list[str]) -> bool:
        is_river = "/" in hist
        if line == "turn" and is_river:
            return False
        if line == "river" and not is_river:
            return False
        top = _top_rank(hole)
        if strength == "strong" and top < 9:
            return False
        if strength == "weak" and top >= 9:
            return False
        return target in acts

    behavior = _perturb(eq, actions, pick, target, mag)
    profile = {"line": line, "target": target, "strength": strength, "mag": mag}
    return behavior, profile


def _shard_indices(n: int, n_shards: int, shard_idx: int) -> list[int]:
    ""
    return [i for i in range(n) if i % n_shards == shard_idx]


def _merge() -> None:
    shards = sorted(Path("results").glob(f"opponent_population_{GAME}_shard*.json"))
    if not shards:
        raise SystemExit(f"no shard files for game={GAME} to merge")
    rows: list[dict[str, Any]] = []
    meta: dict[str, Any] = {}
    for path in shards:
        blob = json.loads(path.read_text())
        rows.extend(blob["rows"])
        meta = {k: blob[k] for k in ("game", "rho", "n_pop", "base_seed")}
    rows.sort(key=lambda r: r["index"])
    solved = [r for r in rows if not r.get("skipped")]
    n_skipped = len(rows) - len(solved)
    summary = _summarize(solved)
    out = _out_path(None)
    out.write_text(
        json.dumps(
            {
                **meta,
                "n_shards_merged": len(shards),
                "n_skipped": n_skipped,
                "summary": summary,
                "rows": rows,
            },
            indent=2,
        )
    )
    print(
        f"merged {len(shards)} shards ({len(solved)} solved, {n_skipped} skipped) "
        f"-> {out}",
        flush=True,
    )
    _print_summary(summary)


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "n": 0,
            "pearson_De_Delta": 0.0,
            "spearman_De_Delta": 0.0,
            "available_lift": 0.0,
            "max_delta": 0.0,
            "mean_delta": 0.0,
            "frac_safe": 1.0,
            "gate_knee_90": None,
            "gate_curve": [],
        }
    de = [r["D_e"] for r in rows]
    delta = [r["obs_minus_core"] for r in rows]
    curve = _gate_curve(de, delta)
    knee = _knee(curve, target=0.9)
    n_safe = sum(1 for r in rows if r["min_safety"] >= r["floor"] - 1e-7)
    return {
        "n": len(rows),
        "pearson_De_Delta": _pearson(de, delta),
        "spearman_De_Delta": _spearman(de, delta),
        "available_lift": sum(delta),
        "max_delta": max(delta) if delta else 0.0,
        "mean_delta": (sum(delta) / len(delta)) if delta else 0.0,
        "frac_safe": n_safe / len(rows) if rows else 1.0,
        "gate_knee_90": knee,
        "gate_curve": curve,
    }


def _print_summary(summary: dict[str, Any]) -> None:
    print(
        f"\n  population n={summary['n']}  "
        f"pearson(D_e,Delta)={summary['pearson_De_Delta']:+.3f}  "
        f"spearman={summary['spearman_De_Delta']:+.3f}",
        flush=True,
    )
    print(
        f"  available_lift={summary['available_lift']:+.3f}  "
        f"max_delta={summary['max_delta']:+.3f}  "
        f"mean_delta={summary['mean_delta']:+.3f}  "
        f"frac_safe={summary['frac_safe']:.3f}",
        flush=True,
    )
    knee = summary["gate_knee_90"]
    if knee:
        print(
            f"  gate knee (capture>=90%): solve_rate={knee['solve_rate']:.2f}  "
            f"captured={knee['captured_frac']:+.3f}  tau={knee['tau']:.5f}",
            flush=True,
        )


def main() -> None:
    if MERGE:
        _merge()
        return

    game = GAME
    print(
        f"# C_obs RANDOM-POPULATION boundary test  game={game}  rho={RHO}  "
        f"n_pop={N_POP}  shard={SHARD_IDX}/{N_SHARDS}",
        flush=True,
    )
    payoff = build_payoff(game)
    sf1 = compile_game(game, 1)
    info_by_label = {info.label: info for info in sf1.info_sets}
    bp = solve_blueprint(game, method="lp")
    assert bp.realization is not None
    v_ref = bp.value
    omega = opponent_reach_weights(bp.realization, game=game)
    fold_idx = _fold_action_indices(sf1)
    actions = {info.label: [a for a, _ in info.children] for info in sf1.info_sets}
    groups = OpponentEvidenceStore.for_game(game).public_groups()
    eq = holdem_equilibrium_opponent(game).behavior
    y_eq = list(Opponent(name="eq", behavior=eq, game=game).realization())
    floor = v_ref - RHO

    indices = _shard_indices(N_POP, N_SHARDS, SHARD_IDX)
    print(
        f"  sequences={sf1.num_sequences} infosets={len(sf1.info_sets)} "
        f"public_states={len(groups)} v_ref={v_ref:+.5f}  this_shard={len(indices)}",
        flush=True,
    )

    rows: list[dict[str, Any]] = []
    out = _out_path(SHARD_IDX if N_SHARDS > 1 else None)
    out.parent.mkdir(parents=True, exist_ok=True)

    def _write_shard(final: bool) -> None:
        payload: dict[str, Any] = {
            "game": game,
            "rho": RHO,
            "n_pop": N_POP,
            "base_seed": BASE_SEED,
            "shard_idx": SHARD_IDX,
            "n_shards": N_SHARDS,
            "complete": final,
            "rows": rows,
        }
        if final and N_SHARDS == 1:
            payload["summary"] = _summarize([r for r in rows if not r.get("skipped")])
        out.write_text(json.dumps(payload, indent=2))

    n_skipped = 0
    for i in indices:
        rng = random.Random(BASE_SEED + i)
        behavior, profile = _random_opponent(rng, eq, actions)
        try:
            with _alarm(OPP_TIMEOUT):
                y_star = list(
                    Opponent(name=f"rp{i}", behavior=behavior, game=game).realization()
                )

                pub_intervals = _population_public_intervals(
                    groups, info_by_label, y_star, omega
                )
                obs_pub_intervals = _population_public_intervals(
                    groups,
                    info_by_label,
                    y_star,
                    omega,
                    fold_only=True,
                    fold_idx=fold_idx,
                )
                event_entries, event_h, event_meta = _population_event_constraints(
                    sf1, y_star, omega, fold_idx
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
                    obs_pub_intervals,
                    event_entries,
                    event_h,
                    v_ref=v_ref,
                    eps_safe=RHO,
                    game=game,
                    weights=omega,
                    row_meta=event_meta,
                )
                responses = [core, obs]
                oracle_gain = None
                if WITH_ORACLE:
                    oracle = safety_constrained_best_response(
                        behavior, v_ref=v_ref, eps_safe=RHO, game=game
                    )
                    responses.append(oracle)
                    oracle_gain = (
                        payoff.bilinear(list(oracle.realization), y_star) - v_ref
                    )
                core_gain = payoff.bilinear(list(core.realization), y_star) - v_ref
                obs_gain = payoff.bilinear(list(obs.realization), y_star) - v_ref
                d_e = _fold_pub_signal(
                    sf1,
                    y_star,
                    y_eq,
                    omega,
                    fold_idx,
                    groups,
                    info_by_label,
                    restrict="turn",
                )
                d_e_all = _fold_pub_signal(
                    sf1, y_star, y_eq, omega, fold_idx, groups, info_by_label
                )
                min_safety = min(
                    safety_verifier(list(r.realization), game=game).value
                    for r in responses
                )
        except _OpponentTimeout:
            n_skipped += 1
            rows.append({"index": i, "profile": profile, "skipped": True})
            _write_shard(final=False)
            print(
                f"  [SKIP {n_skipped}] i={i:<5} {profile['line']:<5} "
                f"{profile['target']} mag={profile['mag']:.2f} "
                f"(exceeded {OPP_TIMEOUT}s)",
                flush=True,
            )
            continue

        row = {
            "index": i,
            "profile": profile,
            "core": core_gain,
            "obs": obs_gain,
            "oracle": oracle_gain,
            "obs_minus_core": obs_gain - core_gain,
            "D_e": d_e,
            "D_e_all": d_e_all,
            "min_safety": min_safety,
            "floor": floor,
        }
        rows.append(row)
        _write_shard(final=False)
        if len(rows) % 10 == 0 or len(rows) == len(indices):
            print(
                f"  [{len(rows):>4}/{len(indices)}] i={i:<5} {profile['line']:<5} "
                f"{profile['target']} {profile['strength']:<6} mag={profile['mag']:.2f}"
                f"  Delta={row['obs_minus_core']:+.3f}  D_e={d_e:.4f}",
                flush=True,
            )

    _write_shard(final=True)
    if n_skipped:
        print(f"  ({n_skipped} opponents skipped on timeout)", flush=True)
    if N_SHARDS == 1:
        _print_summary(_summarize([r for r in rows if not r.get("skipped")]))
    print(f"\nwrote {out}", flush=True)


if __name__ == "__main__":
    main()
