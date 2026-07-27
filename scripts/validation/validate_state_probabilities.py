"""Validate public-state reach and action probabilities."""

import json
import tempfile

from safe_observation import native
from safe_observation.confidence import OpponentEvidenceStore
from safe_observation.opponents import (
    _holdem_p2_actions,
    holdem_showdown_opponent_suite,
)
from safe_observation.sequence_form import compile as compile_game
from safe_observation.solvers import solve_blueprint

GAME = "holdem"
EPISODES = 40000
SEED = 2026
OPPS = [
    "equilibrium",
    "near_equilibrium",
    "censored_fold",
    "overfold",
    "low_reach_leak",
    "calling_station",
    "maniac",
]
WANT = {
    "censored_fold": "kappa0",
    "overfold": "kappa0",
    "low_reach_leak": "kappa0",
    "calling_station": "point",
    "maniac": "point",
}


def per_state_fold(opp_behavior, bp_behavior, fold_idx):
    """Compute per state fold for the validate state probabilities workflow."""
    ev = OpponentEvidenceStore.for_game(GAME)
    _p, show, fold = native.simulate_showdown(
        GAME, bp_behavior, opp_behavior, EPISODES, SEED
    )
    for label, c in show.items():
        ev.record(label, c)
    for label, c in fold.items():
        ev.record(label, c)
    groups = ev.public_groups()
    pub_counts = ev._public_counts()
    out = {}
    for key, counts in pub_counts.items():
        member = groups[key][0]
        fi = fold_idx.get(member)
        total = float(sum(counts))
        if total <= 0 or fi is None or fi >= len(counts):
            continue
        out[key] = (counts[fi] / total, total)
    return out


def main() -> None:
    """Run the command-line entry point."""
    suite = holdem_showdown_opponent_suite()
    bp = solve_blueprint(GAME, method="lp")
    assert bp.realization is not None
    sf0 = compile_game(GAME, 0)
    bp_behavior = sf0.behavior_from_realization(list(bp.realization))

    actions = _holdem_p2_actions()
    fold_idx = {
        label: (acts.index("f") if "f" in acts else None)
        for label, acts in actions.items()
    }

    eq = per_state_fold(suite["equilibrium"].behavior, bp_behavior, fold_idx)

    report = {}
    for name in OPPS:
        opp = per_state_fold(suite[name].behavior, bp_behavior, fold_idx)
        total_mass = sum(m for _r, m in opp.values())
        states = []
        for key, (rate, mass) in opp.items():
            eq_rate = eq.get(key, (0.0, 0.0))[0]
            excess = rate - eq_rate
            states.append(
                {
                    "state": key,
                    "excess": excess,
                    "reach": mass / total_mass if total_mass else 0.0,
                    "fold_rate": rate,
                    "importance": abs(excess)
                    * (mass / total_mass if total_mass else 0.0),
                }
            )
        states.sort(key=lambda s: s["importance"], reverse=True)

        reachable = [s for s in states if s["reach"] >= 0.01]
        peak = max(
            reachable,
            key=lambda s: abs(s["excess"]),
            default={"excess": 0.0, "state": "-"},
        )

        agg = sum(s["excess"] * s["reach"] for s in states)
        report[name] = {"top": states[:4], "peak": peak, "agg": agg}

    with tempfile.NamedTemporaryFile(
        mode="w", prefix="perstate_validation_", suffix=".json", delete=False
    ) as fh:
        json.dump(report, fh, indent=2)
        report_path = fh.name

    print(f"saved {report_path}")

    print(f"PER-PUBLIC-STATE excess fold rate (eps={EPISODES})")
    print(
        f"{'opponent':17s} {'agg':>7s} {'peak|exc|':>9s}  want    dominant states (excess@reach)"
    )
    print("-" * 92)
    for name in OPPS:
        r = report[name]
        want = WANT.get(name, "either")
        peak = r["peak"]

        dom = r["top"][0] if r["top"] else {"excess": 0.0}
        predict = "kappa0" if dom["excess"] > 0.0 else "point"
        ok = "" if want == "either" else ("OK" if predict == want else "** MISS")
        tops = "  ".join(
            f"{s['state'][:14]}={s['excess']:+.2f}@{s['reach']:.2f}"
            for s in r["top"][:3]
        )
        print(
            f"{name:17s} {r['agg']:+7.3f} {peak['excess']:+9.2f}  {want:6s} {ok:8s} {tops}"
        )


if __name__ == "__main__":
    main()
