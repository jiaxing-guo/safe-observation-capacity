""

import json
import tempfile

from safe_observation import native
from safe_observation.opponents import (
    _holdem_p2_actions,
    holdem_showdown_opponent_suite,
)
from safe_observation.sequence_form import compile as compile_game
from safe_observation.solvers import solve_blueprint

GAME = "holdem"
EPISODES = 20000
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
    "equilibrium": "either",
    "near_equilibrium": "either",
    "censored_fold": "kappa0",
    "overfold": "kappa0",
    "low_reach_leak": "kappa0",
    "calling_station": "point",
    "maniac": "point",
}


def fold_rate(show, fold, fold_idx) -> tuple[float, float]:
    ""
    fold_mass = 0.0
    total = 0.0
    for store in (show, fold):
        for label, counts in store.items():
            total += float(sum(counts))
            fi = fold_idx.get(label)
            if fi is not None and fi < len(counts):
                fold_mass += float(counts[fi])
    return (fold_mass / total if total else 0.0), total


def main() -> None:
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

    eq = suite["equilibrium"]
    _p, eq_show, eq_fold = native.simulate_showdown(
        GAME, bp_behavior, eq.behavior, EPISODES, SEED
    )
    eq_rate, _ = fold_rate(eq_show, eq_fold, fold_idx)

    rows = {}
    for name in OPPS:
        opp = suite[name]
        _p, show, fold = native.simulate_showdown(
            GAME, bp_behavior, opp.behavior, EPISODES, SEED
        )
        rate, mass = fold_rate(show, fold, fold_idx)

        s_mass = sum(sum(c) for c in show.values())
        f_mass = sum(sum(c) for c in fold.values())
        rows[name] = {
            "opp_fold_rate": rate,
            "excess_fold_rate": rate - eq_rate,
            "hand_showdown_rate": s_mass / (s_mass + f_mass)
            if (s_mass + f_mass)
            else 0.0,
            "want": WANT[name],
        }

    with tempfile.NamedTemporaryFile(
        mode="w", prefix="foldrate_validation_", suffix=".json", delete=False
    ) as fh:
        json.dump({"eq_fold_rate": eq_rate, "rows": rows}, fh, indent=2)
        report_path = fh.name

    print(f"saved {report_path}")

    print(
        f"opponent CENSORING-rate discriminator (eq baseline fold rate = {eq_rate:.3f})"
    )
    print(
        f"{'opponent':18s} {'foldrate':>9s} {'excess':>8s} "
        f"{'(old)s_rate':>11s}   want      predict"
    )
    print("-" * 72)
    for name in OPPS:
        r = rows[name]

        predict = "kappa0" if r["excess_fold_rate"] > 0.03 else "point"
        ok = (
            ""
            if r["want"] == "either"
            else ("  OK" if predict == r["want"] else "  ** MISS")
        )
        print(
            f"{name:18s} {r['opp_fold_rate']:9.3f} {r['excess_fold_rate']:+8.3f} "
            f"{r['hand_showdown_rate']:11.3f}   {r['want']:8s}  {predict}{ok}"
        )


if __name__ == "__main__":
    main()
