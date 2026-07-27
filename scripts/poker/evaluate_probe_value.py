"""Evaluate the probe value experiment. See Experiments and supplementary Certification at the Unbucketed River."""

from __future__ import annotations

from collections import defaultdict
import json
import statistics as st
import sys

from safe_observation.opponents import Opponent, holdem_equilibrium_opponent
from safe_observation.payoff import build as build_payoff
from safe_observation.sequence_form import compile as compile_game
from safe_observation.solvers import opponent_reach_weights, solve_blueprint
from scripts.poker.run_turn_river_methods import _perturb, _top_rank

GAME = "holdem_tr_b2"
LEAK = sys.argv[1] if len(sys.argv) > 1 else "both"


def _leak(eq, actions, which):
    """Compute leak for the evaluate probe value workflow."""

    def turn(h, hist, acts):
        """Compute turn for the evaluate probe value workflow."""
        return "/" not in hist and "f" in acts and _top_rank(h) >= 9

    def river(h, hist, acts):
        """Compute river for the evaluate probe value workflow."""
        return "/" in hist and "f" in acts and _top_rank(h) >= 9

    out = eq
    if which in ("turn", "both"):
        out = _perturb(out, actions, turn, "f", 0.6)
    if which in ("river", "both"):
        out = _perturb(out, actions, river, "f", 0.5)
    return out


def main() -> None:
    """Run the command-line entry point."""
    sf1 = compile_game(GAME, 1)
    payoff = build_payoff(GAME)
    bp = solve_blueprint(GAME, method="lp")
    x_bp = list(bp.realization)
    eq = holdem_equilibrium_opponent(GAME).behavior
    actions = {i.label: [a for a, _ in i.children] for i in sf1.info_sets}
    y_eq = list(Opponent(name="eq", behavior=eq, game=GAME).realization())
    omega_bp = opponent_reach_weights(x_bp, game=GAME)

    leak = _leak(eq, actions, LEAK)
    y_leak = list(Opponent(name="leak", behavior=leak, game=GAME).realization())

    stake = payoff.matvec_at_x(x_bp)

    by_pub = defaultdict(list)
    for info in sf1.info_sets:
        hist = info.label.split("|", 1)[1]
        v_h = sum(
            (y_leak[info.children[a][1]] - y_eq[info.children[a][1]])
            * stake[info.children[a][1]]
            for a in range(len(info.children))
        )
        w_h = omega_bp.get(info.label, 0.0)
        by_pub[hist].append((info.label, w_h, v_h))

    states = []
    for hist, hands in by_pub.items():
        wv = [(w, v) for _lbl, w, v in hands if w > 1e-12]
        if len(wv) < 2:
            continue
        resolved = sum(w * abs(v) for w, v in wv)
        censored = abs(sum(w * v for w, v in wv))
        evpi = resolved - censored
        if resolved <= 1e-12:
            continue
        states.append(
            {
                "pub": hist,
                "depth": hist.count("/"),
                "n_hands": len(wv),
                "resolved": resolved,
                "censored": censored,
                "evpi": evpi,
                "evpi_frac": evpi / resolved,
            }
        )

    states.sort(key=lambda s: -s["evpi"])
    tot_resolved = sum(s["resolved"] for s in states)
    tot_censored = sum(s["censored"] for s in states)
    tot_evpi = sum(s["evpi"] for s in states)
    print(
        f"# Probe VoI: leak={LEAK}, {len(states)} public states with >=2 reached hands\n"
    )
    print(f"total resolved value   (perfect info):   {tot_resolved:.4f}")
    print(f"total censored value   (public only):    {tot_censored:.4f}")
    print(
        f"total EVPI (de-censoring value)      :    {tot_evpi:.4f}  "
        f"({100 * tot_evpi / tot_resolved:.0f}% of resolved value is hidden by censoring)\n"
    )

    by_d = defaultdict(lambda: [0.0, 0.0, 0.0])
    for s in states:
        by_d[s["depth"]][0] += s["resolved"]
        by_d[s["depth"]][1] += s["censored"]
        by_d[s["depth"]][2] += s["evpi"]
    print(
        "by street depth (0=turn line, 1=river line): resolved | censored | EVPI | EVPI-frac"
    )
    for d in sorted(by_d):
        r, c, e = by_d[d]
        print(f"  depth {d}: {r:.4f} | {c:.4f} | {e:.4f} | {e / r:.2f}")

    print("\ntop-6 states by EVPI:")
    for s in states[:6]:
        print(
            f"  {s['pub']:<16} hands={s['n_hands']:2d}  resolved={s['resolved']:.4f}  "
            f"censored={s['censored']:.4f}  EVPI={s['evpi']:.4f} ({s['evpi_frac']:.0%})"
        )

    out = {
        "game": GAME,
        "leak": LEAK,
        "n_states": len(states),
        "total_resolved": tot_resolved,
        "total_censored": tot_censored,
        "total_evpi": tot_evpi,
        "evpi_fraction": tot_evpi / tot_resolved if tot_resolved > 0 else 0.0,
        "states": states,
    }
    with open(f"results/probe_voi_{LEAK}.json", "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\n# wrote results/probe_voi_{LEAK}.json")
    _ = st


if __name__ == "__main__":
    main()
