""

import json
import os
from pathlib import Path
import re
import tempfile
from time import perf_counter
from typing import Any

from safe_observation.confidence import OpponentEvidenceStore
from safe_observation.opponents import Opponent
from safe_observation.sequence_form import compile as compile_game
from safe_observation.solvers import (
    opponent_reach_weights,
    robust_safe_response_public,
    robust_safe_response_public_cutting_plane,
    solve_blueprint,
)
from scripts.poker.estimate_identifiable_value import _population_public_intervals
from scripts.poker.run_turn_river_methods import _build_opponents

GAME = os.environ.get("SOLVER_TRACE_GAME", "holdem_tr_b2")
RHO = float(os.environ.get("SOLVER_TRACE_RHO", "0.5"))
OPPONENT = os.environ.get("SOLVER_TRACE_OPPONENT", "tr_turn_overfold")
MAX_ITERS = int(os.environ.get("SOLVER_TRACE_MAX_ITERS", "120"))
TOL = float(os.environ.get("SOLVER_TRACE_TOL", "1e-7"))
OUT = Path(
    os.environ.get("SOLVER_TRACE_OUT", f"results/robust_solver_trace_{GAME}.json")
)

_ITER_RE = re.compile(
    r"\+\s*([0-9.]+)s\].*?:iter\s+iter=(\d+)\s+master=([-0-9.eE]+)\s+"
    r"robust=([-0-9.eE]+)\s+safety=([-0-9.eE]+)"
)


def _capture_stderr(fn):
    ""
    old_fd = os.dup(2)
    tf = tempfile.NamedTemporaryFile("w+", delete=False)
    raised = False
    result: Any = None
    try:
        os.dup2(tf.fileno(), 2)
        try:
            result = fn()
        except Exception as exc:
            result = exc
            raised = True
        finally:
            os.dup2(old_fd, 2)
            os.close(old_fd)
        tf.flush()
        tf.seek(0)
        text = tf.read()
    finally:
        tf.close()
        os.unlink(tf.name)
    return result, text, raised


def main() -> None:
    print(
        f"# FA3 solver trace  game={GAME}  opponent={OPPONENT}  rho={RHO}  "
        f"max_iters={MAX_ITERS}",
        flush=True,
    )
    sf1 = compile_game(GAME, 1)
    info_by = {i.label: i for i in sf1.info_sets}
    bp = solve_blueprint(GAME, method="lp")
    v_ref = bp.value
    omega = opponent_reach_weights(bp.realization, game=GAME)
    groups = OpponentEvidenceStore.for_game(GAME).public_groups()
    actions = {i.label: [a for a, _ in i.children] for i in sf1.info_sets}
    opponents = _build_opponents(GAME, actions)
    behavior = opponents[OPPONENT]
    y_star = list(Opponent(name=OPPONENT, behavior=behavior, game=GAME).realization())
    pub = _population_public_intervals(groups, info_by, y_star, omega)

    t0 = perf_counter()
    mono = robust_safe_response_public(
        groups, pub, v_ref=v_ref, eps_safe=RHO, game=GAME, weights=omega
    )
    mono_wall = perf_counter() - t0
    print(
        f"  monolithic: robust_value={mono.robust_value:+.6f}  wall={mono_wall:.2f}s",
        flush=True,
    )

    prev = os.environ.get("SAFE_OBSERVATION_TIMERS")
    os.environ["SAFE_OBSERVATION_TIMERS"] = "1"
    t0 = perf_counter()

    prev = os.environ.get("SAFE_OBSERVATION_TIMERS")
    os.environ["SAFE_OBSERVATION_TIMERS"] = "1"
    t0 = perf_counter()
    try:
        cp_result, captured, raised = _capture_stderr(
            lambda: robust_safe_response_public_cutting_plane(
                groups,
                pub,
                v_ref=v_ref,
                eps_safe=RHO,
                game=GAME,
                weights=omega,
                max_iters=MAX_ITERS,
                tol=TOL,
            )
        )
    finally:
        if prev is None:
            os.environ.pop("SAFE_OBSERVATION_TIMERS", None)
        else:
            os.environ["SAFE_OBSERVATION_TIMERS"] = prev
    cp_wall = perf_counter() - t0

    converged = not raised
    if raised:
        msg = str(cp_result)
        mm = re.search(r"last master ([-0-9.eE]+).*?last robust ([-0-9.eE]+)", msg)
        cp_value = float(mm.group(2)) if mm else None
        cp_master = float(mm.group(1)) if mm else None
        print(
            f"  cutting-plane did NOT converge in {MAX_ITERS} iters (expected): {msg[:80]}",
            flush=True,
        )
    else:
        cp_value = cp_result.robust_value
        cp_master = None

    iters: list[dict[str, Any]] = []
    for m in _ITER_RE.finditer(captured):
        wall, it, master, robust, safety = m.groups()
        iters.append(
            {
                "iter": int(it),
                "wall_s": float(wall),
                "master_ub": float(master),
                "robust_lb": float(robust),
                "safety": float(safety),
                "gap": float(master) - float(robust),
            }
        )

    final_gap = iters[-1]["gap"] if iters else None
    print(
        f"  cutting-plane: robust_value={cp_value}  wall={cp_wall:.2f}s  "
        f"iters_logged={len(iters)}  converged={converged}  final_gap={final_gap}",
        flush=True,
    )
    if iters:
        print(
            f"  first iter: master={iters[0]['master_ub']:+.4f} "
            f"robust={iters[0]['robust_lb']:+.4f} gap={iters[0]['gap']:.4f}  |  "
            f"last iter: master={iters[-1]['master_ub']:+.4f} "
            f"robust={iters[-1]['robust_lb']:+.4f} gap={iters[-1]['gap']:.4f}",
            flush=True,
        )

        if len(iters) >= 2:
            d_early = iters[1]["wall_s"] - iters[0]["wall_s"]
            d_late = iters[-1]["wall_s"] - iters[-2]["wall_s"]
            print(
                f"  per-iter wall: early~{d_early:.3f}s  late~{d_late:.3f}s",
                flush=True,
            )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps(
            {
                "game": GAME,
                "opponent": OPPONENT,
                "rho": RHO,
                "v_ref": v_ref,
                "max_iters": MAX_ITERS,
                "tol": TOL,
                "monolithic": {"robust_value": mono.robust_value, "wall_s": mono_wall},
                "cutting_plane": {
                    "robust_value": cp_value,
                    "master_at_stop": cp_master,
                    "wall_s": cp_wall,
                    "iters_logged": len(iters),
                    "final_gap": final_gap,
                    "converged": converged,
                    "speedup_vs_monolithic": cp_wall / mono_wall
                    if mono_wall > 0
                    else None,
                },
                "iters": iters,
            },
            indent=2,
        )
    )
    print(f"\nwrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
