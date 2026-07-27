"""Run the online reach experiment. See Experiments and supplementary Certification at the Unbucketed River."""

import json
import os
from pathlib import Path

from safe_observation import native
from safe_observation.confidence import (
    OpponentEvidenceStore,
    empirical_bernstein_halfwidth,
)
from safe_observation.experiments.online import _censored_em_behavior
from safe_observation.opponents import Opponent, holdem_structured_opponent_suite
from safe_observation.payoff import build as build_payoff
from safe_observation.sequence_form import compile as compile_game
from safe_observation.solvers import (
    opponent_reach_weights,
    robust_safe_response_envelope,
    robust_safe_response_public,
    safety_constrained_best_response,
    safety_verifier,
    solve_blueprint,
)

try:
    from run_turn_river_methods import (
        _build_opponents as _build_turn_river_opponents,
    )
except ImportError:
    _build_turn_river_opponents = None

GAME = os.environ.get("OR_GAME", "holdem")
OPPONENT = os.environ.get("OR_OPP", "board_public_call")
ROUNDS = int(os.environ.get("OR_ROUNDS", "40"))
EPISODES = int(os.environ.get("OR_EPISODES", "3000"))
SEED = int(os.environ.get("OR_SEED", "2026"))
DELTA = 0.1
METHOD = "empirical_bernstein"
RHO_CAP = 0.5
SEED_ROUNDS = int(os.environ.get("OR_SEED_ROUNDS", "10"))
OUT = Path(os.environ.get("OR_OUT", "results/online_reach_holdem.json"))


def _opponent_for_game(game: str, opponent_name: str, sf1):
    """Compute opponent for game for the run online reach workflow."""
    if game.startswith("holdem_tr"):
        if _build_turn_river_opponents is None:
            raise RuntimeError("turn-river opponent builder is unavailable")
        actions = {
            info.label: [action for action, _ in info.children]
            for info in sf1.info_sets
        }
        opponents = _build_turn_river_opponents(game, actions)
        if opponent_name not in opponents:
            raise KeyError(
                f"unknown turn-river opponent {opponent_name!r}; choices={sorted(opponents)}"
            )
        return Opponent(
            name=opponent_name, behavior=opponents[opponent_name], game=game
        )
    return holdem_structured_opponent_suite(game)[opponent_name]


def _fold_idx(sf1) -> dict[str, int]:
    """Compute fold idx for the run online reach workflow."""
    out: dict[str, int] = {}
    for info in sf1.info_sets:
        for i, (a, _) in enumerate(info.children):
            if a == "f":
                out[info.label] = i
                break
    return out


def _accumulate_reach(omega_acc: dict[str, float], omega_t: dict[str, float]) -> None:
    """Compute accumulate reach for the run online reach workflow."""
    for label, w in omega_t.items():
        omega_acc[label] = omega_acc.get(label, 0.0) + w


def _showdown_boxes(ev_point, omega_acc, fold_idx):
    """Compute showdown boxes for the run online reach workflow."""
    labels = list(ev_point.labels)
    n_boxes = sum(
        1
        for lbl in labels
        if omega_acc.get(lbl, 0.0) > 0.0
        for i in range(len(ev_point.counts(lbl)))
        if i != fold_idx.get(lbl)
    )
    delta_box = DELTA / max(1, n_boxes)
    boxes: dict[str, list[tuple[float, float]]] = {}
    for lbl in labels:
        w = omega_acc.get(lbl, 0.0)
        counts = ev_point.counts(lbl)
        k = len(counts)
        total = float(sum(counts))
        fi = fold_idx.get(lbl)
        if w <= 0.0 or total <= 0.0:
            boxes[lbl] = [(0.0, 1.0)] * k
            continue
        row: list[tuple[float, float]] = []
        for a in range(k):
            if a == fi:
                row.append((0.0, 1.0))
                continue

            phat = counts[a] / total
            var = phat * (1.0 - phat)
            half = empirical_bernstein_halfwidth(int(total), var, delta_box)
            row.append((max(0.0, phat - half), min(1.0, phat + half)))
        boxes[lbl] = row
    return boxes


def _solve_arm(arm, groups, pub_intervals, boxes, omega_acc, v_ref, game):
    """Solve arm for the run online reach workflow."""
    if arm == "core":
        resp = robust_safe_response_public(
            groups,
            pub_intervals,
            v_ref=v_ref,
            eps_safe=RHO_CAP,
            game=game,
            weights=omega_acc,
        )
        return {
            "realization": resp.realization,
            "cert_gain": resp.robust_value - v_ref,
            "solver": "core",
            "box_scale": None,
        }

    scale = 1.0
    for _ in range(10):
        infl = {
            lbl: [
                (
                    max(0.0, lo - (scale - 1.0) * 0.05),
                    min(1.0, hi + (scale - 1.0) * 0.05),
                )
                for (lo, hi) in rows
            ]
            for lbl, rows in boxes.items()
        }
        try:
            resp = robust_safe_response_envelope(
                groups,
                pub_intervals,
                infl,
                v_ref=v_ref,
                eps_safe=RHO_CAP,
                game=game,
                weights=omega_acc,
            )
            return {
                "realization": resp.realization,
                "cert_gain": resp.robust_value - v_ref,
                "solver": "obs",
                "box_scale": scale,
            }
        except ValueError:
            scale += 1.0

    resp = robust_safe_response_public(
        groups,
        pub_intervals,
        v_ref=v_ref,
        eps_safe=RHO_CAP,
        game=game,
        weights=omega_acc,
    )
    return {
        "realization": resp.realization,
        "cert_gain": resp.robust_value - v_ref,
        "solver": "fallback_core",
        "box_scale": scale,
    }


def main() -> None:
    """Run the command-line entry point."""
    game = GAME
    payoff = build_payoff(game)
    sf0 = compile_game(game, 0)
    sf1 = compile_game(game, 1)
    bp = solve_blueprint(game, method="lp")
    assert bp.realization is not None
    x_bp = list(bp.realization)
    v_ref = bp.value
    fold_idx = _fold_idx(sf1)

    opp = _opponent_for_game(game, OPPONENT, sf1)
    y_star = list(opp.realization())

    oracle = safety_constrained_best_response(
        opp.behavior, v_ref=v_ref, eps_safe=RHO_CAP, game=game
    )
    oracle_gain = payoff.bilinear(list(oracle.realization), y_star) - v_ref
    om_bp = opponent_reach_weights(x_bp, game=game)
    om_oracle = opponent_reach_weights(list(oracle.realization), game=game)
    exploit_labels = {
        lbl
        for lbl in om_oracle
        if om_oracle.get(lbl, 0.0) > 5e-3
        and om_bp.get(lbl, 0.0) < 0.2 * om_oracle.get(lbl, 0.0)
    }

    print(
        f"# online adaptive-reach toy  game={game}  opp={OPPONENT}  "
        f"rounds={ROUNDS} eps/rd={EPISODES}  oracle_gain={oracle_gain:+.3f}\n"
        f"# exploit info sets (off-blueprint-reach): {len(exploit_labels)}\n",
        flush=True,
    )

    arms = ["core", "obs", "em", "obs_seed"]
    state = {
        arm: {
            "ev_point": OpponentEvidenceStore.for_game(game),
            "ev_public": OpponentEvidenceStore.for_game(game),
            "omega_acc": {},
            "x_prev": list(x_bp),
        }
        for arm in arms
    }
    history: dict[str, list[dict]] = {arm: [] for arm in arms}

    print(
        f"{'rd':>3} | " + " | ".join(f"{arm:>20}" for arm in arms),
        flush=True,
    )
    for rd in range(ROUNDS):
        line = f"{rd:>3} | "
        for arm in arms:
            st = state[arm]
            ab = sf0.behavior_from_realization(st["x_prev"])
            _p, show, fold = native.simulate_showdown(
                game, ab, opp.behavior, EPISODES, SEED + rd
            )
            for lbl, c in show.items():
                st["ev_point"].record(lbl, c)
                st["ev_public"].record(lbl, c)
            for lbl, c in fold.items():
                st["ev_public"].record(lbl, c)
            _accumulate_reach(
                st["omega_acc"], opponent_reach_weights(st["x_prev"], game=game)
            )

            groups = st["ev_public"].public_groups()
            pub_intervals = st["ev_public"].public_intervals(DELTA, method=METHOD)

            if arm == "em" or (arm == "obs_seed" and rd < SEED_ROUNDS):
                p_em = _censored_em_behavior(
                    game, st["ev_point"], st["ev_public"], st["omega_acc"]
                )
                x = list(
                    safety_constrained_best_response(
                        p_em, v_ref=v_ref, eps_safe=RHO_CAP, game=game
                    ).realization
                )
                cert_gain = None
                solver = "em"
                box_scale = None
            elif arm == "core":
                solved = _solve_arm(
                    "core",
                    groups,
                    pub_intervals,
                    None,
                    st["omega_acc"],
                    v_ref,
                    game,
                )
                x = list(solved["realization"])
                cert_gain = solved["cert_gain"]
                solver = solved["solver"]
                box_scale = solved["box_scale"]
            else:
                boxes = _showdown_boxes(st["ev_point"], st["omega_acc"], fold_idx)
                solved = _solve_arm(
                    "obs",
                    groups,
                    pub_intervals,
                    boxes,
                    st["omega_acc"],
                    v_ref,
                    game,
                )
                x = list(solved["realization"])
                cert_gain = solved["cert_gain"]
                solver = solved["solver"]
                box_scale = solved["box_scale"]

            realized = payoff.bilinear(x, y_star) - v_ref
            exploit_reach = sum(st["omega_acc"].get(lbl, 0.0) for lbl in exploit_labels)
            min_safety = safety_verifier(x, game=game).value
            history[arm].append(
                {
                    "round": rd,
                    "realized": realized,
                    "cert_gain": cert_gain,
                    "exploit_reach": exploit_reach,
                    "min_safety": min_safety,
                    "solver": solver,
                    "box_scale": box_scale,
                }
            )
            st["x_prev"] = x
            line += f"{realized:+.3f} r={exploit_reach:5.2f} | "
        if rd % 5 == 0 or rd == ROUNDS - 1:
            print(line, flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps(
            {
                "game": game,
                "opponent": OPPONENT,
                "rounds": ROUNDS,
                "episodes": EPISODES,
                "v_ref": v_ref,
                "oracle_gain": oracle_gain,
                "seed_rounds": SEED_ROUNDS,
                "n_exploit_labels": len(exploit_labels),
                "history": history,
            },
            indent=2,
        )
    )

    print("\n=== final (mean of last 5 rounds) ===", flush=True)
    for arm in arms:
        tail = history[arm][-5:]
        rmean = sum(h["realized"] for h in tail) / len(tail)
        cert_vals = [h["cert_gain"] for h in tail if h.get("cert_gain") is not None]
        cmean = sum(cert_vals) / len(cert_vals) if cert_vals else None
        ereach = history[arm][-1]["exploit_reach"]
        msafe = min(h["min_safety"] for h in history[arm])
        cstr = "n/a" if cmean is None else f"{cmean:+.3f}"
        print(
            f"  {arm:<10} realized={rmean:+.3f}  cert={cstr:>7}  exploit_reach={ereach:6.2f}  "
            f"min_safety={msafe:+.3f}",
            flush=True,
        )
    print(f"\noracle_gain={oracle_gain:+.3f}", flush=True)
    print(
        "READ: if obs exploit_reach stays ~0 and realized ~ core while em grows both,"
        " a robust-only agent cannot self-bootstrap off-reach value -> selector needed"
        " (Case 2). If obs_seed (explore-then-certify) reaches oracle, explore-with-em /"
        " certify-with-obs is viable (rescues a single-method story).",
        flush=True,
    )
    print(f"wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
