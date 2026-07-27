"""Run the interval ablation experiment. See Experiments and supplementary Certification at the Unbucketed River."""

import json
import math
from pathlib import Path

from safe_observation import native
from safe_observation.confidence import OpponentEvidenceStore
from safe_observation.opponents import holdem_showdown_opponent_suite
from safe_observation.sequence_form import compile as compile_game
from safe_observation.solvers import opponent_reach_weights

METHODS = ("hoeffding", "empirical_bernstein")

OPPONENTS = ("equilibrium", "censored_fold", "overfold", "low_reach_leak")
DELTA = 0.1
ROUNDS = 60
EPISODES = 200
SEEDS = tuple(range(42, 67))


def _mean_halfwidth(intervals) -> float:
    """Compute mean halfwidth for the run interval ablation workflow."""
    widths = [(hi - lo) / 2.0 for bounds in intervals.values() for (lo, hi) in bounds]
    return sum(widths) / len(widths) if widths else 0.0


def _min_halfwidth(intervals) -> float:
    """Compute min halfwidth for the run interval ablation workflow."""
    widths = [(hi - lo) / 2.0 for bounds in intervals.values() for (lo, hi) in bounds]
    return min(widths) if widths else 0.0


def _inside(game, groups, intervals, weights, y_star) -> bool:
    """Return whether the estimate lies inside the supplied interval."""
    grp = {k: list(v) for k, v in groups.items()}
    payload = {k: [tuple(b) for b in bd] for k, bd in intervals.items()}
    cset = native.ConfidenceSet.from_public(game, grp, payload, weights)
    return cset.nrows == 0 or cset.max_violation(y_star) <= 1e-9


def run_opponent(game, opp, weights) -> dict:
    """Construct the run opponent policy."""
    y_star = list(opp.realization())
    blueprint_behavior = compile_game(game, 0).behavior_from_realization(
        native.blueprint_realization(game, 0)
    )

    final_mean_hw = {m: 0.0 for m in METHODS}
    final_min_hw = {m: 0.0 for m in METHODS}
    never_violated = {m: 0 for m in METHODS}
    final_n = 0.0
    for seed in SEEDS:
        store = OpponentEvidenceStore.for_game(game)
        violated = {m: False for m in METHODS}
        for t in range(1, ROUNDS + 1):
            _pay, show, fold = native.simulate_showdown(
                game, blueprint_behavior, opp.behavior, EPISODES, seed * 100_000 + t
            )
            for label, c in show.items():
                store.record(label, c)
            for label, c in fold.items():
                store.record(label, c)
            groups = store.public_groups()
            delta_t = DELTA * 6.0 / (math.pi**2 * t * t)
            for m in METHODS:
                intervals = store.public_intervals(delta_t, method=m, union_bound=True)
                if not _inside(game, groups, intervals, weights, y_star):
                    violated[m] = True
                if t == ROUNDS:
                    final_mean_hw[m] += _mean_halfwidth(intervals)
                    final_min_hw[m] += _min_halfwidth(intervals)

        final_n += sum(sum(c) for c in store._public_counts().values())
        for m in METHODS:
            if not violated[m]:
                never_violated[m] += 1
    ns = len(SEEDS)
    h_mean, b_mean = final_mean_hw["hoeffding"], final_mean_hw["empirical_bernstein"]
    h_min, b_min = final_min_hw["hoeffding"], final_min_hw["empirical_bernstein"]
    return {
        "opponent": opp.name,
        "mean_public_samples": final_n / ns,
        "methods": {
            m: {
                "mean_halfwidth": final_mean_hw[m] / ns,
                "min_halfwidth": final_min_hw[m] / ns,
                "anytime_coverage": never_violated[m] / ns,
            }
            for m in METHODS
        },
        "mean_bernstein_over_hoeffding": (
            b_mean / h_mean if h_mean > 0 else float("nan")
        ),
        "min_bernstein_over_hoeffding": (b_min / h_min if h_min > 0 else float("nan")),
    }


def main() -> None:
    """Run the command-line entry point."""
    game = "holdem"
    suite = holdem_showdown_opponent_suite(game=game)
    by_name = {o.name: o for o in suite.values()}
    weights = opponent_reach_weights(native.blueprint_realization(game, 0), game=game)
    results = {
        "game": game,
        "monitoring": "showdown",
        "rounds": ROUNDS,
        "episodes_per_round": EPISODES,
        "seeds": list(SEEDS),
        "delta": DELTA,
        "opponents": [],
    }
    for name in OPPONENTS:
        opp = by_name[name]
        print(f"=== interval ablation: {name} ===", flush=True)
        res = run_opponent(game, opp, weights)
        h = res["methods"]["hoeffding"]
        b = res["methods"]["empirical_bernstein"]
        print(
            f"  n={res['mean_public_samples']:.0f}  "
            f"mean hw H={h['mean_halfwidth']:.3f} B={b['mean_halfwidth']:.3f} "
            f"(B/H={res['mean_bernstein_over_hoeffding']:.2f})  | "
            f"min hw H={h['min_halfwidth']:.4f} B={b['min_halfwidth']:.4f} "
            f"(B/H={res['min_bernstein_over_hoeffding']:.2f})  | "
            f"cov H={h['anytime_coverage']:.2f} B={b['anytime_coverage']:.2f}",
            flush=True,
        )
        results["opponents"].append(res)
    out = Path("results/interval_ablation_holdem.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
